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

    fig = plt.figure(figsize=(7, 9))
    axs = [fig.add_subplot(211, projection=aia_map), fig.add_subplot(212)]
    im0 = aia_map.plot(axes=axs[0], cmap='gray')

    im = axs[1].imshow(final_image_1[0], cmap='gray', origin='lower')

    t_text = axs[1].text(
        0.02, 1.5,
        't={}'.format(time[0]),
        transform=axs[1].transAxes
    )

    w_text = axs[1].text(
        0.02, 1.3,
        'w={}'.format(wave[0]),
        transform=axs[1].transAxes
    )

    corr_text = axs[1].text(
        0.3, 1.03,
        'Pearson Corr: {}'.format(corr),
        transform=axs[1].transAxes
    )

    f1_text = axs[1].text(
        -0.1, -0.2,
        'F1: {}'.format(filename),
        transform=axs[1].transAxes
    )

    f2_text = axs[1].text(
        -0.1, -0.4,
        'F2: {}'.format(closest_file.name),
        transform=axs[1].transAxes
    )

    text_image = axs[1].text(
        0.1, 0.9,
        'F1',
        transform=axs[1].transAxes
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

        final_image_1 = [upsampled_image]

        final_image_2 = [resampled_submap.data]

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
        axs[0].draw_artist(t_text)
        axs[0].draw_artist(w_text)
        axs[0].draw_artist(corr_text)
        axs[0].draw_artist(f2_text)
        fig.canvas.draw_idle()

        timer.stop()
        timer.callbacks = []
        timer.add_callback(flicker_callback)
        timer.start()

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
                crop=crop_name, centre_arcsec=[xc, yc], wavelength_index=wavelength))
        manifest = dict(schema_version=2, source_gris=str((base_path / filename).resolve()),
                        timestamps_csv=str(Path(timestamps_path).resolve()),
                        native_scale_arcsec=0.135, frames=records)
        (hmi_write_path / 'alignment.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def update_wave(val):
        wave[0] = int(val)
        set_val(
            time[0],
            init_x[0],
            init_y[0],
            wave[0]
        )
        update_timer()

    def update_time(val):
        time[0] = int(val)
        update_timer()

    def on_click_im0(event):
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
        if event.key == "enter":
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
    save_button.on_clicked(do_save)
    fig.canvas.mpl_connect('button_press_event', on_click_im0)
    fig.canvas.mpl_connect("key_press_event", handle_enter)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.4, top=0.99, wspace=0.0, hspace=0.2)

    # plt.ion()          # turn on interactive mode
    plt.show()  # non-blocking show


if __name__ == '__main__':

    plt.switch_backend('QtAgg')

    base_path = Path('/mnt/f/GRIS')

    filename = '25Apr25ARM1-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    hmi_path = base_path / 'SDO' / 'HMI' / 'Continuum'

    hmi_write_path = base_path / 'aligned_SDO' / 'HMI' / 'Continuum'

    timestamps_path = Path(__file__).with_name('serie_timestamps.csv')
    animate(base_path, filename, hmi_path, hmi_write_path, timestamps_path,
            subpixel_target=0.005, save_crops=False)
