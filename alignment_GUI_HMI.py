import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import sunpy.io
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import matplotlib
from scipy.stats import pearsonr
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from scipy.signal import fftconvolve
import scipy
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
from scipy.ndimage import zoom
from datetime import datetime
from aiapy.calibrate import register
from datetime import timedelta, timezone
from astropy import units as u
import astropy.coordinates
import sys
import time
from sunpy.util.metadata import MetaDict


plt.switch_backend('QtAgg')


def get_upsampled_image(image, factor):
    if factor == 1:
        return image

    ny, nx = image.shape

    # Original grid
    y = np.arange(ny)
    x = np.arange(nx)

    # Spline interpolator
    spline = RectBivariateSpline(y, x, image, kx=1, ky=1)

    # New grid
    new_ny = int(np.round(ny * factor))
    new_nx = int(np.round(nx * factor))
    y_new = np.linspace(0, ny - 1, new_ny)
    x_new = np.linspace(0, nx - 1, new_nx)

    # Evaluate the spline on new grid
    zoomed_image = spline(y_new, x_new)

    return zoomed_image


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

    mask = ~np.isnan(region1) & ~np.isnan(region2)
    corr = np.round(pearsonr(region1[mask].flatten(), region2[mask].flatten()).statistic, 4)

    return corr


def average_downsample(image, factor):
    if factor == 1:
        return image
    h, w = image.shape
    return image[:h//factor*factor, :w//factor*factor].reshape(
        h//factor, factor, w//factor, factor).mean(axis=(1, 3))


def align_and_merge_ca_he_data_streaming(
        base_path, filename1, filename2, subpixel_accuracy=1,
        offset_1_y=0, offset_1_x=0, offset_2_y=0, offset_2_x=0
):
    data1, _ = sunpy.io.read_file(base_path / filename1)[0]
    data2, _ = sunpy.io.read_file(base_path / filename2)[0]

    print("Read data...")

    is_5d = (len(data1.shape) == 5)
    if is_5d:
        T1, P1, H1, W1, C1 = data1.shape
        T2, P2, H2, W2, C2 = data2.shape
    else:
        T1, H1, W1, C1 = data1.shape
        T2, H2, W2, C2 = data2.shape
        P1 = P2 = 1  # Dummy polarization axis

    # Consistency check for T and P
    assert T1 == T2 and P1 == P2, "Mismatch in time or polarization dimensions"

    # Final canvas size (upsampled)
    up_H1 = H1 * subpixel_accuracy
    up_W1 = W1 * subpixel_accuracy
    up_H2 = H2 * subpixel_accuracy
    up_W2 = W2 * subpixel_accuracy

    canvas_H = max(offset_1_y + up_H1, offset_2_y + up_H2)
    canvas_W = max(offset_1_x + up_W1, offset_2_x + up_W2)

    final_H = canvas_H // subpixel_accuracy
    final_W = canvas_W // subpixel_accuracy

    print(f"Output shape after alignment and downsampling: ({final_H}, {final_W})")

    # Preallocate
    if is_5d:
        aligned_1 = np.zeros((T1, P1, final_H, final_W, C1), dtype=np.float32)
        aligned_2 = np.zeros((T2, P2, final_H, final_W, C2), dtype=np.float32)
    else:
        aligned_1 = np.zeros((T1, final_H, final_W, C1), dtype=np.float32)
        aligned_2 = np.zeros((T2, final_H, final_W, C2), dtype=np.float32)

    canvas1 = np.empty((canvas_H, canvas_W), dtype=np.float32)
    canvas2 = np.empty((canvas_H, canvas_W), dtype=np.float32)

    total_steps = T1 * P1 * (C1 + C2)
    with tqdm(total=total_steps, desc="Aligning frames") as pbar:
        for t in range(T1):
            for p in range(P1):
                for c in range(C1):
                    img1 = data1[t, p, :, :, c] if is_5d else data1[t, :, :, c]
                    img1_up = get_upsampled_image(img1, subpixel_accuracy)
                    canvas1.fill(np.median(img1_up))
                    canvas1[offset_1_y:offset_1_y + img1_up.shape[0],
                            offset_1_x:offset_1_x + img1_up.shape[1]] = img1_up
                    img1_down = average_downsample(canvas1, subpixel_accuracy)
                    if is_5d:
                        aligned_1[t, p, :, :, c] = img1_down
                    else:
                        aligned_1[t, :, :, c] = img1_down
                    pbar.update(1)

                for c in range(C2):
                    img2 = data2[t, p, :, :, c] if is_5d else data2[t, :, :, c]
                    img2_up = get_upsampled_image(img2, subpixel_accuracy)
                    canvas2.fill(np.median(img2_up))
                    canvas2[offset_2_y:offset_2_y + img2_up.shape[0],
                            offset_2_x:offset_2_x + img2_up.shape[1]] = img2_up
                    img2_down = average_downsample(canvas2, subpixel_accuracy)
                    if is_5d:
                        aligned_2[t, p, :, :, c] = img2_down
                    else:
                        aligned_2[t, :, :, c] = img2_down
                    pbar.update(1)

    print("Saving aligned, downsampled data...")

    sunpy.io.write_file(
        base_path / f'{filename1}_aligned_downsampled_streamed.fits',
        aligned_1,
        dict(),
        overwrite=True
    )
    sunpy.io.write_file(
        base_path / f'{filename2}_aligned_downsampled_streamed.fits',
        aligned_2,
        dict(),
        overwrite=True
    )

    print("Done and saved.")


def get_closest(exact_date, cadence, time, fits_datetimes):
    target_datetime = exact_date + timedelta(seconds=cadence * time)

    closest = min(fits_datetimes, key=lambda x: abs(x[0] - target_datetime))

    closest_datetime, closest_file = closest

    return closest_datetime, closest_file


def extract_submap_with_metadata(aia_map, xc, yc, dx, dy, onx, ony, in_scale=0.6, out_scale=0.005):

    data = aia_map.data
    ny, nx = data.shape


    crpix1 = aia_map.meta['crpix1']  # 1-based
    crpix2 = aia_map.meta['crpix2']

    # Input arcsec grid
    x_vals = (np.arange(nx) - (crpix1 - 1)) * in_scale
    y_vals = (np.arange(ny) - (crpix2 - 1)) * in_scale

    # Construct output coordinate arrays WITHOUT exceeding given half-widths
    x_start = xc - dx
    x_end   = xc + dx
    y_start = yc - dy
    y_end   = yc + dy


    x_new = np.arange(xc-dx, xc+dx, out_scale)[0:onx]
    y_new = np.arange(yc-dy, yc+dy, out_scale)[0:ony]

    data[np.where(np.isnan(data))] = 0
    # Interpolate
    rbs = RectBivariateSpline(y_vals, x_vals, data, kx=1, ky=1)
    sub_data = rbs(y_new, x_new)

    # Update metadata
    new_meta = MetaDict(aia_map.meta.copy())
    new_meta['naxis1'] = len(x_new)
    new_meta['naxis2'] = len(y_new)
    new_meta['crpix1'] = np.round((xc - x_new[0]) / out_scale, 4) + 1
    new_meta['crpix2'] = np.round((yc - y_new[0]) / out_scale, 4) + 1
    new_meta['crval1'] = xc
    new_meta['crval2'] = yc
    new_meta['cdelt1'] = out_scale
    new_meta['cdelt2'] = out_scale

    return sunpy.map.Map(sub_data, new_meta)


def get_hmi_submap(
    aia_map, image, init_x, init_y
):

    spread_x = image.shape[1] * 0.135 / 2

    spread_y = image.shape[0] * 0.135 / 2

    onx = int(spread_x * 2 / 0.005)

    ony = int(spread_y * 2 / 0.005)

    resampled_submap = extract_submap_with_metadata(aia_map, init_x, init_y, spread_x, spread_y, onx, ony, in_scale=0.6, out_scale=0.005)

    return resampled_submap


def animate(
    base_path, filename, hmi_path,
    exact_date, cadence=44, subpixel_target=0.005
):

    time = [0]

    init_x = [0]

    init_y = [0]

    wave = [0]

    frame_toggle = [0]

    aia_map_dict = dict()

    def get_aia_map(closest_file):
        aia_map = None

        if closest_file.name not in aia_map_dict:
            hmi_data, hmi_header = sunpy.io.read_file(closest_file)[1]

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
            return 0, 0, 0

    def set_val(time, init_x, init_y, wave):
        print ("setting for time: {}".format(time))
        if time in val_dict:
            val_dict[time]['init_x'] = init_x
            val_dict[time]['init_y'] = init_y
            val_dict[time]['wave'] = wave
        else:
            val_dict[time] = dict()
            val_dict[time]['init_x'] = init_x
            val_dict[time]['init_y'] = init_y
            val_dict[time]['wave'] = wave

        print (val_dict)

    hmi_file_list = list(hmi_path.glob("*.fits"))

    # Extract UTC datetime from filenames
    fits_datetimes = []
    for hmi_file in hmi_file_list:
        name = hmi_file.name
        try:
            # Extract date and time from filename
            date_str = name.split('.')[2] + name.split('.')[3]  # e.g., '20250425' + '090815'
            date_str = date_str[:-5]
            dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
            dt_utc = dt.replace(tzinfo=timezone.utc)  # make it timezone-aware in UTC
            fits_datetimes.append((dt_utc, hmi_file))
        except Exception as e:
            print (e)  # Skip malformed filenames
            sys.exit(-1)

    data, _ = sunpy.io.read_file(base_path / filename)[0]

    closest_datetime, closest_file = get_closest(exact_date, cadence, time[0], fits_datetimes)

    aia_map = get_aia_map(closest_file)

    image = get_orig_image(data, time[0], wave[0])

    upsampled_image = get_upsampled_image(image, 0.135/0.005)

    resampled_submap = get_hmi_submap(
        aia_map, image, init_x[0], init_y[0]
    )    

    final_image_1 = [upsampled_image]

    final_image_2 = [resampled_submap.data]

    corr = get_correlation_value(
        upsampled_image, resampled_submap.data
    )

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 1, figsize=(7, 9))

    extent = [
        (1 - aia_map.meta['crpix1']) * 0.6,  (aia_map.data.shape[0] - aia_map.meta['crpix1']) * 0.6,
        (1 - aia_map.meta['crpix1']) * 0.6,  (aia_map.data.shape[0] - aia_map.meta['crpix1']) * 0.6
    ]

    im0 = axs[0].imshow(aia_map.data, cmap='gray', origin='lower', extent=extent)

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

    def prepare_flicker_callback(frame_toggle, im, fig, image1, image2, text_image):
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
            fig.canvas.draw_idle()
            frame_toggle[0] = 1 - frame_toggle[0]
        return flicker_callback

    flicker_callback = prepare_flicker_callback(frame_toggle, im, fig, final_image_1[0], final_image_2[0], text_image)

    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(flicker_callback)
    timer.start()

    def update_timer():

        u_init_x, u_init_y, u_wave = get_val(time[0])

        closest_datetime, closest_file = get_closest(exact_date, cadence, time[0], fits_datetimes)

        aia_map = get_aia_map(closest_file)

        image = get_orig_image(data, time[0], u_wave)

        upsampled_image = get_upsampled_image(image, 0.135/0.005)

        resampled_submap = get_hmi_submap(
            aia_map, image, u_init_x, u_init_y
        )    

        final_image_1 = [upsampled_image]

        final_image_2 = [resampled_submap.data]

        corr = get_correlation_value(
            upsampled_image, resampled_submap.data
        )

        flicker_callback = prepare_flicker_callback(
            frame_toggle, im, fig,
            final_image_1[0], final_image_2[0], text_image
        )
        timer.stop()
        timer.callbacks = []
        timer.add_callback(flicker_callback)
        timer.start()
        t_text.set_text('t={}'.format(time[0]))
        w_text.set_text('w={}'.format(wave[0]))
        corr_text.set_text('Pearson Corr: {}'.format(corr))

    def do_save(event):
        # align_and_merge_ca_he_data_streaming(
        #     base_path, filename1, filename2, subpixel_accuracy,
        #     offset_1_y[0], offset_1_x[0], offset_2_y[0], offset_2_x[0]
        # )
        pass
    
    def update_wave(val):
        wave[0] = val
        set_val(
            time[0],
            init_x[0],
            init_y[0],
            wave[0]
        )
        update_timer()

    def update_time(val):
        time[0] = val
        update_timer()

    # def update_init_x(val):
    #     init_x[0] = np.round(float(val), 2)
    #     set_val(
    #         time[0],
    #         init_x[0],
    #         init_y[0],
    #         wave[0]
    #     )
    #     update_timer()

    # def update_init_y(val):
    #     init_y[0] = np.round(float(val), 2)
    #     set_val(
    #         time[0],
    #         init_x[0],
    #         init_y[0],
    #         wave[0]
    #     )
    #     update_timer()

    def on_click_im0(event):
        if event.inaxes != axs[0]:
            return

        x, y = event.xdata, event.ydata

        x, y = np.round(x, 2), np.round(y, 2)
        init_x[0] = x
        init_y[0] = y

        init_x_textbox.set_val(str(x))

        init_y_textbox.set_val(str(y))

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
    # init_x_textbox.on_submit(update_init_x)
    # init_y_textbox.on_submit(update_init_y)
    fig.canvas.mpl_connect('button_press_event', on_click_im0)
    fig.canvas.mpl_connect("key_press_event", handle_enter)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.4, top=0.99, wspace=0.0, hspace=0.2)

    plt.show()


if __name__ == '__main__':

    exact_date = '2025-04-25T09:08:00+0000'

    base_path = Path('/mnt/f/GRIS')

    filename = '25Apr25ARM1-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    hmi_path = base_path / 'SDO' / 'HMI' / 'Continuum'

    exact_date = '2025-04-25T09:08:00+0000'

    exact_date = datetime.strptime(exact_date, '%Y-%m-%dT%H:%M:%S%z')

    animate(base_path, filename, hmi_path, exact_date, cadence=44, subpixel_target=0.005)
