"""Interactively fit GRIS positions and persist WCS plus full registered HMI maps."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import sunpy.map
from sunpy.util.metadata import MetaDict
from astropy import units as u
import astropy.coordinates
from astropy.time import Time
from astropy.io import fits
from astropy.wcs.utils import pixel_to_pixel
from aiapy.calibrate import register
from scipy.ndimage import map_coordinates
from scipy.stats import pearsonr
from scipy.optimize import least_squares, minimize
from tqdm import tqdm

from hmi_alignment import load_timestamps, get_closest


def get_upsampled_image(image, factor):
    if factor == 1:
        return image

    ny, nx = image.shape

    # Preserve native pixel centres and footprint under integer upsampling.
    new_ny, new_nx = int(round(ny * factor)), int(round(nx * factor))
    y, x = np.meshgrid((np.arange(new_ny) + 0.5) / factor - 0.5,
                       (np.arange(new_nx) + 0.5) / factor - 0.5,
                       indexing='ij')
    return map_coordinates(np.asarray(image, dtype=float), [y, x],
                           order=1, mode='nearest')


def get_orig_image(data, time, wave, subpixel_accuracy=1):

    if len(data.shape) == 5:
        image = data[time, 0, :, :, wave]
    else:
        image = data[0, :, :, wave]

    image = get_upsampled_image(image, subpixel_accuracy)

    return image


def get_correlation_value(
    region1, region2
):

    region1 = region1[2:, 2:]

    region2 = region2[2:, 2:]

    mask = np.isfinite(region1) & np.isfinite(region2)
    if mask.sum() < 2:
        return np.nan
    corr = np.round(pearsonr(region1[mask].flatten(), region2[mask].flatten()).statistic, 4)

    return corr


def average_downsample(image, factor):
    if factor == 1:
        return image
    h, w = image.shape
    return image[:h//factor*factor, :w//factor*factor].reshape(
        h//factor, factor, w//factor, factor).mean(axis=(1, 3))


def downsample_sunpy_map(map_in, factor):
    
    new_data = average_downsample(map_in.data, factor)

    # Copy and modify the metadata
    new_meta = MetaDict(map_in.meta.copy())
    new_meta['NAXIS1'] = new_data.shape[1]
    new_meta['NAXIS2'] = new_data.shape[0]
    new_meta['CDELT1'] *= factor
    new_meta['CDELT2'] *= factor
    new_meta['CRPIX1'] = (new_meta['CRPIX1'] - 0.5) / factor + 0.5
    new_meta['CRPIX2'] = (new_meta['CRPIX2'] - 0.5) / factor + 0.5

    return sunpy.map.Map(new_data, new_meta)


def gris_map(image, registered, xc, yc):
    """Attach the fitted solar WCS to the native GRIS pixel grid."""
    centre = astropy.coordinates.SkyCoord(xc * u.arcsec, yc * u.arcsec,
                                          frame=registered.coordinate_frame)
    header = sunpy.map.make_fitswcs_header(
        image, centre, scale=[0.135, 0.135] * u.arcsec / u.pix)
    return sunpy.map.Map(image, header)


def get_hmi_submap(registered, image, init_x, init_y, factor=27):
    native = gris_map(image, registered, init_x, init_y)
    header = native.meta.copy()
    for axis in (1, 2):
        header[f'crpix{axis}'] = (header[f'crpix{axis}'] - 0.5) * factor + 0.5
        header[f'cdelt{axis}'] /= factor
        header[f'naxis{axis}'] *= factor
    shape = (image.shape[0] * factor, image.shape[1] * factor)
    grid = sunpy.map.Map(np.zeros(shape), header)
    y, x = np.indices(shape, dtype=float)
    hx, hy = pixel_to_pixel(grid.wcs, registered.wcs, x, y)
    sampled = map_coordinates(np.asarray(registered.data, dtype=float), [hy, hx],
                              order=1, mode='constant', cval=np.nan)
    return sunpy.map.Map(sampled, header)


def align_feature_pairs(image, registered, gris_points, hmi_points,
                        patch_radius=20, search_radius=1.5):
    """Fit translation from two pixel pairs, then maximize local Pearson scores.

    Points are zero-based (x, y) in native GRIS / registered HMI pixels.
    Refinement is bounded to search_radius arcsec around the landmark fit.
    """
    gris_points = np.asarray(gris_points, dtype=float)
    hmi_points = np.asarray(hmi_points, dtype=float)
    if (gris_points.shape != (2, 2) or hmi_points.shape != (2, 2)
            or not np.isfinite([gris_points, hmi_points]).all()):
        raise ValueError("Mark both features in both GRIS and HMI first.")
    if np.linalg.norm(gris_points[1] - gris_points[0]) < 2:
        raise ValueError("Choose two distinct GRIS features, at least 2 pixels apart.")
    if np.linalg.norm(hmi_points[1] - hmi_points[0]) < 0.5:
        raise ValueError("Choose two distinct HMI features.")
    for points, shape in ((gris_points, image.shape), (hmi_points, registered.data.shape)):
        if np.any(points < -0.5) or np.any(points >= np.array(shape[::-1]) - 0.5):
            raise ValueError("Feature markers must lie inside their images.")
    if patch_radius < 2 or search_radius <= 0:
        raise ValueError("Patch radius must be at least 2 pixels and search radius positive.")

    coords = registered.pixel_to_world(hmi_points[:, 0] * u.pix, hmi_points[:, 1] * u.pix)
    initial = np.array([np.mean(coords.Tx.to_value(u.arcsec)
                               - (gris_points[:, 0] - (image.shape[1] - 1) / 2) * 0.135),
                        np.mean(coords.Ty.to_value(u.arcsec)
                               - (gris_points[:, 1] - (image.shape[0] - 1) / 2) * 0.135)])

    def point_residual(centre):
        native = gris_map(image, registered, *centre)
        x, y = pixel_to_pixel(native.wcs, registered.wcs,
                             gris_points[:, 0], gris_points[:, 1])
        return (np.column_stack((x, y)) - hmi_points).ravel()

    fit = least_squares(point_residual, initial, diff_step=1e-4)
    if not fit.success:
        raise ValueError("Could not fit the marked feature positions.")
    seed = fit.x
    patches = []
    for px, py in gris_points:
        x0, x1 = max(0, int(round(px)) - patch_radius), min(image.shape[1], int(round(px)) + patch_radius + 1)
        y0, y1 = max(0, int(round(py)) - patch_radius), min(image.shape[0], int(round(py)) + patch_radius + 1)
        y, x = np.mgrid[y0:y1, x0:x1]
        values = np.asarray(image[y0:y1, x0:x1], dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 25 or np.std(values[valid]) <= 1e-12:
            raise ValueError("Each feature needs a textured patch with at least 25 finite pixels.")
        patches.append((x[valid], y[valid], values[valid]))

    hmi_values = np.asarray(registered.data, dtype=float)

    def scores(centre):
        native = gris_map(image, registered, *centre)
        result = []
        for x, y, values in patches:
            hx, hy = pixel_to_pixel(native.wcs, registered.wcs, x, y)
            sample = map_coordinates(hmi_values, [hy, hx],
                                     order=1, mode='constant', cval=np.nan)
            valid = np.isfinite(sample)
            if valid.sum() < max(25, int(np.ceil(0.8 * len(values)))):
                return None
            a, b = values[valid] - values[valid].mean(), sample[valid] - sample[valid].mean()
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if denominator <= 1e-12:
                return None
            result.append(float(np.dot(a, b) / denominator))
        return result

    def objective(offset):
        correlations = scores(seed + offset)
        return 2.0 if correlations is None else -np.mean(correlations)

    # A coarse bounded search avoids relying on the local gradient at the clicks.
    offsets = np.linspace(-search_radius, search_radius, 9)
    candidates = [(objective([dx, dy]), [dx, dy]) for dx in offsets for dy in offsets]
    best_score, best_offset = min(candidates, key=lambda item: item[0])
    refined = minimize(objective, best_offset, method='Powell',
                       bounds=[(-search_radius, search_radius)] * 2,
                       options={'xtol': 0.001, 'ftol': 1e-6, 'maxiter': 40})
    offset = refined.x if refined.success and refined.fun < best_score else np.array(best_offset)
    final = seed + offset
    correlations = scores(final)
    if correlations is None or min(correlations) < 0.3:
        raise ValueError("Weak feature correlation. Re-mark textured continuum features or enlarge the patches.")
    if np.max(np.abs(offset)) >= search_radius - 0.01:
        raise ValueError("Best match reaches the search boundary. Re-mark the features or enlarge Search.")
    return final, dict(method='two_feature_correlation',
                       landmark_centre_arcsec=seed.tolist(),
                       refinement_arcsec=offset.tolist(),
                       feature_correlations=correlations,
                       landmark_rms_hmi_pixels=float(np.sqrt(np.mean(point_residual(seed) ** 2))),
                       patch_radius_gris_pixels=int(patch_radius), search_radius_arcsec=float(search_radius),
                       gris_points=gris_points.tolist(), hmi_points=hmi_points.tolist())


def animate(
    base_path, filename, hmi_path, hmi_write_path,
    timestamps_path, subpixel_target=0.005, save_crops=False
):

    if not np.isfinite(subpixel_target) or subpixel_target <= 0:
        raise ValueError("subpixel_target must be finite and positive")
    factor = int(round(0.135 / subpixel_target))
    if factor < 1 or not np.isclose(factor * subpixel_target, 0.135):
        raise ValueError("subpixel_target must divide the native 0.135 arcsec scale")
    time = [0]

    init_x = [0]

    init_y = [0]

    wave = [0]

    frame_toggle = [0]

    aia_map_dict = dict()

    def get_aia_map(closest_file):
        aia_map = None

        if closest_file.name not in aia_map_dict:
            hmi_data, hmi_header = fits.getdata(closest_file, ext=1, header=True)

            hmi_map = sunpy.map.Map(hmi_data, hmi_header)

            aia_map = register(hmi_map)

            aia_map_dict[closest_file.name] = aia_map
        else:
            aia_map = aia_map_dict[closest_file.name]

        return aia_map

    val_dict = dict()
    feature_pairs = {}  # Separate marks for each (time, wavelength).
    selected_feature = [None]

    def get_val(time):
        if time in val_dict:
            return val_dict[time]['init_x'],  val_dict[time]['init_y'], val_dict[time]['wave']
        else:
            max_time = max(val_dict)
            init_x = val_dict[max_time]['init_x']
            init_y = val_dict[max_time]['init_y']
            wave = val_dict[max_time]['wave']
            set_val(time, init_x, init_y, wave)
            return init_x, init_y, wave

    def set_val(time, init_x, init_y, wave):

        if time in val_dict:
            val_dict[time].pop('auto_alignment', None)
            val_dict[time]['init_x'] = init_x
            val_dict[time]['init_y'] = init_y
            val_dict[time]['wave'] = wave
        else:
            val_dict[time] = dict()
            val_dict[time]['init_x'] = init_x
            val_dict[time]['init_y'] = init_y
            val_dict[time]['wave'] = wave

    hmi_file_list = list(hmi_path.glob("*.fits"))

    # JSOC filename clocks explicitly labelled TAI must be converted to UTC.
    fits_datetimes = []
    for hmi_file in sorted(hmi_file_list):
        match = re.search(r"(\d{8})_(\d{6})_TAI", hmi_file.name)
        if not match:
            raise ValueError(f"Unrecognized HMI timestamp: {hmi_file.name}")
        dt = datetime.strptime(''.join(match.groups()), "%Y%m%d%H%M%S")
        dt_utc = Time(dt, scale='tai').utc.to_datetime(timezone=timezone.utc)
        fits_datetimes.append((dt_utc, hmi_file))

    data = fits.getdata(base_path / filename, ext=0)
    timestamps = load_timestamps(timestamps_path, data.shape[0] if data.ndim == 5 else 1)
    set_val(0, init_x[0], init_y[0], wave[0])

    closest_datetime, closest_file = get_closest(timestamps[time[0]], fits_datetimes)

    aia_map = get_aia_map(closest_file)

    image = get_orig_image(data, time[0], wave[0])

    upsampled_image = get_upsampled_image(image, factor)

    resampled_submap = get_hmi_submap(
        aia_map, image, init_x[0], init_y[0], factor
    )    

    final_image_1 = [upsampled_image]

    final_image_2 = [resampled_submap.data]

    corr = get_correlation_value(
        upsampled_image, resampled_submap.data
    )

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(9, 11))
    axs = [fig.add_subplot(211, projection=aia_map), fig.add_subplot(212)]
    im0 = aia_map.plot(axes=axs[0], cmap='gray')

    im = axs[1].imshow(final_image_1[0], cmap='gray', origin='lower')

    t_text = fig.text(0.08, 0.475, 't={}'.format(time[0]))
    w_text = fig.text(0.18, 0.475, 'w={}'.format(wave[0]))
    corr_text = fig.text(0.32, 0.475, 'Pearson Corr: {}'.format(corr))

    f1_text = fig.text(
        0.08, 0.445,
        'F1: {}'.format(filename)
    )

    f2_text = fig.text(
        0.08, 0.427,
        'F2: {}'.format(closest_file.name)
    )

    text_image = axs[1].text(
        0.1, 0.9,
        'F1',
        transform=axs[1].transAxes, color='white',
        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
    )

    slider_ax_time = plt.axes([0.1, 0.23, 0.8, 0.03])
    slider_ax_init_x = plt.axes([0.1, 0.18, 0.8, 0.03])
    slider_ax_init_y = plt.axes([0.1, 0.13, 0.8, 0.03])
    slider_ax_wave = plt.axes([0.1, 0.08, 0.8, 0.03])
    button_ax = plt.axes([0.6, 0.28, 0.2, 0.04])
    
    t_max = 0

    if len(data.shape) == 5:
        t_max = data.shape[0] - 1

    time_slider = Slider(
        ax=slider_ax_time,
        label='time',
        valmin=0,
        valmax=t_max,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    init_x_textbox = TextBox(
        ax=slider_ax_init_x,
        label='init_x'
    )

    init_y_textbox = TextBox(
        ax=slider_ax_init_y,
        label='init_y'
    )

    init_x_textbox.set_val("0")

    init_y_textbox.set_val("0")

    if len(data.shape) == 5:
        wmax = data.shape[4] - 1
    else:
        wmax = data.shape[3] - 1
    
    wave_slider = Slider(
        ax=slider_ax_wave,
        label='wave',
        valmin=0,
        valmax=wmax,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    save_button = Button(button_ax, 'Save Data')
    feature_buttons = [Button(plt.axes([0.08 + i * 0.18, 0.33, 0.16, 0.035]), f'Feature {i + 1}')
                       for i in range(2)]
    auto_button = Button(plt.axes([0.44, 0.33, 0.18, 0.035]), 'Auto Align')
    clear_button = Button(plt.axes([0.64, 0.33, 0.18, 0.035]), 'Clear Marks')
    flicker_button = Button(plt.axes([0.08, 0.28, 0.18, 0.04]), 'Resume Flicker')
    patch_box = TextBox(plt.axes([0.37, 0.28, 0.07, 0.035]), 'Patch px ', initial='20')
    search_box = TextBox(plt.axes([0.51, 0.28, 0.07, 0.035]), 'Search ″ ', initial='1.5')
    status = fig.text(0.08, 0.385, 'Select Feature 1, then click it in GRIS (bottom) and HMI (top).', fontsize=8, wrap=True)
    markers = [[ax.plot([], [], '+', color=color, ms=12, mew=2)[0]
                for color in ('tab:orange', 'tab:cyan')] for ax in axs]
    labels = [[ax.text(0, 0, str(i + 1), color=color, visible=False)
               for i, color in enumerate(('tab:orange', 'tab:cyan'))] for ax in axs]

    def draw_marks():
        pairs = feature_pairs.get((time[0], wave[0]), {})
        for ax_index, side in enumerate(('hmi', 'gris')):
            for index in range(2):
                point = pairs.get(index, {}).get(side)
                visible = point is not None and selected_feature[0] is not None
                markers[ax_index][index].set_visible(visible)
                labels[ax_index][index].set_visible(visible)
                if visible:
                    x, y = point
                    if side == 'gris':
                        x, y = (x + 0.5) * factor - 0.5, (y + 0.5) * factor - 0.5
                    markers[ax_index][index].set_data([x], [y])
                    labels[ax_index][index].set_position((x, y))

    def show_gris():
        timer.stop()
        im.set_data(final_image_1[0])
        im.set_clim(np.nanmin(final_image_1[0]), np.nanmax(final_image_1[0]))
        text_image.set_text('GRIS — mark features')
        draw_marks()
        fig.canvas.draw_idle()

    def choose_feature(index):
        selected_feature[0] = index
        status.set_text(f'Feature {index + 1}: click its centre in GRIS and HMI. Click again to replace a mark.')
        show_gris()

    def resume_flicker(event):
        selected_feature[0] = None
        draw_marks()
        status.set_text('Flicker preview. Select a feature button to mark points again.')
        timer.start()
        fig.canvas.draw_idle()

    def clear_marks(event):
        feature_pairs.pop((time[0], wave[0]), None)
        val_dict[time[0]].pop('auto_alignment', None)
        choose_feature(0)

    def auto_align(event):
        pairs = feature_pairs.get((time[0], wave[0]), {})
        if any(side not in pairs.get(i, {}) for i in range(2) for side in ('gris', 'hmi')):
            status.set_text('Mark Feature 1 and Feature 2 in both images before Auto Align.')
            fig.canvas.draw_idle()
            return
        timer.stop()
        status.set_text('Matching feature patches…')
        fig.canvas.draw_idle()
        try:
            radius = int(patch_box.text)
            search = float(search_box.text)
            if not np.isfinite(search):
                raise ValueError('Search must be finite.')
            _, source = get_closest(timestamps[time[0]], fits_datetimes)
            centre, report = align_feature_pairs(
                get_orig_image(data, time[0], wave[0]), get_aia_map(source),
                [pairs[i]['gris'] for i in range(2)], [pairs[i]['hmi'] for i in range(2)],
                patch_radius=radius, search_radius=search)
        except ValueError as error:
            status.set_text(str(error))
            fig.canvas.draw_idle()
            return
        init_x[0], init_y[0] = map(float, centre)
        set_val(time[0], init_x[0], init_y[0], wave[0])
        val_dict[time[0]]['auto_alignment'] = report
        selected_feature[0] = None
        update_timer()
        c1, c2 = report['feature_correlations']
        status.set_text(f'Aligned: centre ({centre[0]:.3f}, {centre[1]:.3f}) arcsec; correlations {c1:.3f}, {c2:.3f}.')
        fig.canvas.draw_idle()

    def prepare_flicker_callback(frame_toggle, im, axs, fig, image1, image2, text_image):
        def flicker_callback(*args):

            if frame_toggle[0] == 0:
                frame = image1
                text_image.set_text('F1')
            else:
                frame = image2
                text_image.set_text('F2')
    
            im.set_array(frame)
            mn, mx = np.nanmin(frame) * 0.9, np.nanmax(frame) * 1.1
            im.set_clim(mn, mx)
            axs.draw_artist(axs.patch)
            axs.draw_artist(im)
            axs.draw_artist(text_image)
            fig.canvas.blit(axs.bbox)
            frame_toggle[0] = 1 - frame_toggle[0]
        return flicker_callback

    flicker_callback = prepare_flicker_callback(frame_toggle, im, axs[1], fig, final_image_1[0], final_image_2[0], text_image)

    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(flicker_callback)
    timer.start()

    def update_timer():

        u_init_x, u_init_y, u_wave = get_val(time[0])

        init_x[0], init_y[0], wave[0] = u_init_x, u_init_y, u_wave

        init_x_textbox.set_val(str(u_init_x))

        init_y_textbox.set_val(str(u_init_y))

        closest_datetime, closest_file = get_closest(timestamps[time[0]], fits_datetimes)

        aia_map = get_aia_map(closest_file)

        image = get_orig_image(data, time[0], u_wave)

        upsampled_image = get_upsampled_image(image, factor)

        resampled_submap = get_hmi_submap(
            aia_map, image, u_init_x, u_init_y, factor
        )

        final_image_1[0] = upsampled_image

        final_image_2[0] = resampled_submap.data

        corr = get_correlation_value(
            upsampled_image, resampled_submap.data
        )

        flicker_callback = prepare_flicker_callback(
            frame_toggle, im, axs[1], fig,
            final_image_1[0], final_image_2[0], text_image
        )


        t_text.set_text('t={}'.format(time[0]))
        w_text.set_text('w={}'.format(wave[0]))
        corr_text.set_text('Pearson Corr: {}'.format(corr))
        f2_text.set_text(
            'F2: {}'.format(closest_file.name)
        )

        axs[0].reset_wcs(aia_map.wcs)
        im0.set_extent((-0.5, aia_map.data.shape[1] - 0.5, -0.5, aia_map.data.shape[0] - 0.5))
        im0.set_array(aia_map.data)
        mn, mx = np.nanmin(aia_map.data) * 0.9, np.nanmax(aia_map.data) * 1.1
        im0.set_clim(mn, mx)
        axs[0].draw_artist(axs[0].patch)
        axs[0].draw_artist(im0)
        fig.canvas.draw_idle()

        timer.stop()
        timer.callbacks = []
        timer.add_callback(flicker_callback)
        draw_marks()
        if selected_feature[0] is None:
            timer.start()
        else:
            show_gris()

    def do_save(event):
        missing = [i for i in range(t_max + 1) if i not in val_dict]
        if missing:
            print(f"Please align the remaining frame indices: {missing}")
            return
        hmi_write_path.mkdir(parents=True, exist_ok=True)
        registered_dir = hmi_write_path / 'registered'
        gris_dir = hmi_write_path / 'gris_wcs'
        registered_dir.mkdir(exist_ok=True)
        gris_dir.mkdir(exist_ok=True)
        records, saved = [], set()
        for a_t in tqdm(range(t_max + 1), desc="Saving alignment"):
            xc, yc, wavelength = get_val(a_t)
            matched_time, source = get_closest(timestamps[a_t], fits_datetimes)
            registered = get_aia_map(source)
            registered_path = registered_dir / source.name
            if source not in saved:
                registered.save(registered_path, overwrite=True)
                saved.add(source)
            image = get_orig_image(data, a_t, wavelength)
            aligned = gris_map(image, registered, xc, yc)
            aligned.meta['grisdate'] = timestamps[a_t].isoformat()
            aligned.meta['grisidx'] = a_t
            aligned.meta['griswave'] = wavelength
            gris_path = gris_dir / f'gris_{a_t:04d}.hdr'
            aligned.fits_header.totextfile(gris_path, overwrite=True)
            crop_name = None
            if save_crops:
                crop = downsample_sunpy_map(
                    get_hmi_submap(registered, image, xc, yc, factor), factor)
                crop_name = f'gris_{a_t:04d}_{source.name}'
                crop.save(hmi_write_path / crop_name, overwrite=True)
            records.append(dict(
                frame_index=a_t, series_index=a_t + 1,
                timestamp_utc=timestamps[a_t].isoformat(),
                hmi_timestamp_utc=matched_time.isoformat(),
                hmi_minus_gris_seconds=(matched_time - timestamps[a_t]).total_seconds(),
                source_hmi=str(source.resolve()),
                registered_hmi=str(registered_path.relative_to(hmi_write_path)),
                gris_header=str(gris_path.relative_to(hmi_write_path)),
                crop=crop_name, centre_arcsec=[xc, yc], wavelength_index=wavelength,
                auto_alignment=val_dict[a_t].get('auto_alignment')))
        manifest = dict(schema_version=2, source_gris=str((base_path / filename).resolve()),
                        timestamps_csv=str(Path(timestamps_path).resolve()),
                        native_scale_arcsec=0.135, frames=records)
        (hmi_write_path / 'alignment.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def update_wave(val):
        selected_feature[0] = None
        status.set_text('Wavelength changed. Select feature buttons to mark this image.')
        wave[0] = int(val)
        set_val(
            time[0],
            init_x[0],
            init_y[0],
            wave[0]
        )
        update_timer()

    def update_time(val):
        selected_feature[0] = None
        status.set_text('Frame changed. Select feature buttons to mark this frame.')
        time[0] = int(val)
        update_timer()

    def on_click_im0(event):
        toolbar = getattr(fig.canvas, 'toolbar', None)
        if event.button != 1 or (toolbar is not None and toolbar.mode):
            return
        if selected_feature[0] is not None and event.inaxes in axs:
            side = 'hmi' if event.inaxes == axs[0] else 'gris'
            x, y = float(event.xdata), float(event.ydata)
            if side == 'gris':
                x, y = (x + 0.5) / factor - 0.5, (y + 0.5) / factor - 0.5
            pairs = feature_pairs.setdefault((time[0], wave[0]), {})
            pairs.setdefault(selected_feature[0], {})[side] = [x, y]
            val_dict[time[0]].pop('auto_alignment', None)
            count = sum(len(pair) for pair in pairs.values())
            status.set_text(f'{count}/4 marks placed. Select Feature 1 or 2 to edit; Auto Align when all four are marked.')
            show_gris()
            return
        if event.inaxes != axs[0]:
            return

        _, source = get_closest(timestamps[time[0]], fits_datetimes)
        coord = get_aia_map(source).pixel_to_world(event.xdata * u.pix, event.ydata * u.pix)
        x, y = coord.Tx.to_value(u.arcsec), coord.Ty.to_value(u.arcsec)

        x, y = np.round(x, 2), np.round(y, 2)
        init_x[0] = x
        init_y[0] = y

        set_val(
            time[0],
            init_x[0],
            init_y[0],
            wave[0]
        )

        update_timer()

    def handle_enter(event):
        if event.key == "enter" and event.inaxes in (slider_ax_init_x, slider_ax_init_y):
            text_x = init_x_textbox.text
            text_y = init_y_textbox.text
            init_x[0] = np.round(float(text_x), 2)
            init_y[0] = np.round(float(text_y), 2)
            set_val(
                time[0],
                init_x[0],
                init_y[0],
                wave[0]
            )
            update_timer()

    wave_slider.on_changed(update_wave)
    time_slider.on_changed(update_time)
    for index, button in enumerate(feature_buttons):
        button.on_clicked(lambda event, index=index: choose_feature(index))
    auto_button.on_clicked(auto_align)
    clear_button.on_clicked(clear_marks)
    flicker_button.on_clicked(resume_flicker)
    save_button.on_clicked(do_save)
    fig.canvas.mpl_connect('button_press_event', on_click_im0)
    fig.canvas.mpl_connect("key_press_event", handle_enter)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.50, top=0.96, wspace=0.0, hspace=0.2)

    # plt.ion()          # turn on interactive mode
    plt.show()  # non-blocking show


if __name__ == '__main__':

    plt.switch_backend('QtAgg')

    base_path = Path('/mn/stornext/d9/data/harshm/GRISData')

    filename = '25Apr25ARM1-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    hmi_path = base_path / 'SDO' / 'HMI' / 'Continuum'

    hmi_write_path = base_path / 'aligned_SDO' / 'HMI' / 'Continuum'

    timestamps_path = Path(__file__).with_name('serie_timestamps.csv')
    animate(base_path, filename, hmi_path, hmi_write_path, timestamps_path,
            subpixel_target=0.005, save_crops=False)
