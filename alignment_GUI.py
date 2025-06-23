import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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


plt.switch_backend('QtAgg')


def get_upsampled_image(image, factor):
    if factor == 1:
        return image
    """
    Zooms a 2D image using spline interpolation.

    Parameters:
    -----------
    image : 2D numpy array
        Input image to be zoomed.
    factor : float
        Zoom factor (>1 for upsampling, <1 for downsampling).

    Returns:
    --------
    zoomed_image : 2D numpy array
        Resampled image with new shape.
    """
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


def get_orig_images(data1, data2, time, wave1, wave2, subpixel_accuracy=1):

    if len(data1.shape) == 5:
        image1 = data1[time, 0, :, :, wave1]

        image2 = data2[time, 0, :, :, wave2]
    else:
        image1 = data1[0, :, :, wave1]

        image2 = data2[0, :, :, wave2]

    image1 = get_upsampled_image(image1, subpixel_accuracy)

    image2 = get_upsampled_image(image2, subpixel_accuracy)

    return image1, image2


def get_final_images(
    data1, data2, time,
    offset_1_x, offset_1_y,
    offset_2_x, offset_2_y,
    wave1, wave2,
    subpixel_accuracy=1
):

    image1, image2 = get_orig_images(data1, data2, time, wave1, wave2, subpixel_accuracy)

    fill_value_1 = np.median(image1)

    fill_value_2 = np.median(image2)

    final_image_1 = np.ones(
        shape=(
            max(
                image1.shape[0] + offset_1_y,
                image2.shape[0] + offset_2_y
            ),
            max(
                image1.shape[1] + offset_1_x,
                image2.shape[1] + offset_2_x
            )
        )
    ) * fill_value_1


    final_image_2 = np.ones(
        shape=(
            max(
                image1.shape[0] + offset_1_y,
                image2.shape[0] + offset_2_y
            ),
            max(
                image1.shape[1] + offset_1_x,
                image2.shape[1] + offset_2_x
            )
        )
    ) * fill_value_2

    final_image_1[0 + offset_1_y: image1.shape[0] + offset_1_y, 0 + offset_1_x: image1.shape[1] + offset_1_x] = image1

    final_image_2[0 + offset_2_y: image2.shape[0] + offset_2_y, 0 + offset_2_x: image2.shape[1] + offset_2_x] = image2

    final_image_1 = final_image_1 / np.nanmax(np.abs(final_image_1))

    final_image_2 = final_image_2 / np.nanmax(np.abs(final_image_2))

    return final_image_1, final_image_2



def get_overlapping_coordinates(image1, image2, offset_1_x, offset_1_y, offset_2_x, offset_2_y):
    # Top-left of each image in final canvas
    i1_y0, i1_x0 = offset_1_y, offset_1_x
    i2_y0, i2_x0 = offset_2_y, offset_2_x

    # Bottom-right (exclusive)
    i1_y1, i1_x1 = i1_y0 + image1.shape[0], i1_x0 + image1.shape[1]
    i2_y1, i2_x1 = i2_y0 + image2.shape[0], i2_x0 + image2.shape[1]

    # Compute intersection (overlapping box)
    overlap_y0 = max(i1_y0, i2_y0)
    overlap_x0 = max(i1_x0, i2_x0)
    overlap_y1 = min(i1_y1, i2_y1)
    overlap_x1 = min(i1_x1, i2_x1)

    return overlap_y0, overlap_x0, overlap_y1, overlap_x1


def get_correlation_value(
    data1, data2,
    time, wave1, wave2,
    offset_1_x, offset_1_y,
    offset_2_x, offset_2_y,
    final_image_1, final_image_2,
    subpixel_accuracy=1
):
    image1, image2 = get_orig_images(data1, data2, time, wave1, wave2, subpixel_accuracy)

    overlap_y0, overlap_x0, overlap_y1, overlap_x1 = get_overlapping_coordinates(
        image1, image2, offset_1_x, offset_1_y, offset_2_x, offset_2_y
    )

    region1 = final_image_1[overlap_y0:overlap_y1, overlap_x0:overlap_x1]
    region2 = final_image_2[overlap_y0:overlap_y1, overlap_x0:overlap_x1]

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


def animate(base_path, filename1, filename2, subpixel_accuracy):

    time = [0]

    offset_1_x = [0]

    offset_1_y = [0]

    offset_2_x = [0]

    offset_2_y = [0]

    wave1 = [0]

    wave2 = [0]

    frame_toggle = [0]

    data1, _ = sunpy.io.read_file(base_path / filename1)[0]

    data2, _ = sunpy.io.read_file(base_path / filename2)[0]

    a, b = get_final_images(
        data1, data2, time[0],
        offset_1_x[0], offset_1_y[0],
        offset_2_x[0], offset_2_y[0],
        wave1[0], wave2[0],
        subpixel_accuracy
    )

    final_image_1 = [a]

    final_image_2 = [b]

    corr = get_correlation_value(
        data1, data2,
        time[0], wave1[0], wave2[0],
        offset_1_x[0], offset_1_y[0],
        offset_2_x[0], offset_2_y[0],
        final_image_1[0], final_image_2[0],
        subpixel_accuracy
    )

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(1, 1, figsize=(7, 9))

    im = axs.imshow(final_image_1[0], cmap='gray', origin='lower')

    t_text = axs.text(
        0.05, 1.4,
        't={}'.format(time[0]),
        transform=axs.transAxes
    )

    w1_text = axs.text(
        0.15, 1.4,
        'w1={}'.format(wave1[0]),
        transform=axs.transAxes
    )

    w2_text = axs.text(
        0.25, 1.4,
        'w2={}'.format(wave2[0]),
        transform=axs.transAxes
    )

    o1x_text = axs.text(
        0.05, 1.3,
        'o1x={}'.format(offset_1_x[0]),
        transform=axs.transAxes
    )

    o1y_text = axs.text(
        0.15, 1.3,
        'o1y={}'.format(offset_1_y[0]),
        transform=axs.transAxes
    )

    o2x_text = axs.text(
        0.05, 1.2,
        'o2x={}'.format(offset_2_x[0]),
        transform=axs.transAxes
    )

    o2y_text = axs.text(
        0.15, 1.2,
        'o2y={}'.format(offset_2_y[0]),
        transform=axs.transAxes
    )

    corr_text = axs.text(
        0.3, 1.1,
        'Pearson Corr: {}'.format(corr),
        transform=axs.transAxes
    )

    f1_text = axs.text(
        -0.1, -0.2,
        'F1: {}'.format(filename1),
        transform=axs.transAxes
    )

    f1_text = axs.text(
        -0.1, -0.4,
        'F2: {}'.format(filename2),
        transform=axs.transAxes
    )

    text_image = axs.text(
        0.1, 0.9,
        'F1',
        transform=axs.transAxes
    )

    slider_ax_time = plt.axes([0.1, 0.33, 0.8, 0.03])
    slider_ax_offset1x = plt.axes([0.1, 0.28, 0.8, 0.03])
    slider_ax_offset1y = plt.axes([0.1, 0.23, 0.8, 0.03])
    slider_ax_offset2x = plt.axes([0.1, 0.18, 0.8, 0.03])
    slider_ax_offset2y = plt.axes([0.1, 0.13, 0.8, 0.03])
    slider_ax_wave2 = plt.axes([0.1, 0.03, 0.8, 0.03])
    slider_ax_wave1 = plt.axes([0.1, 0.08, 0.8, 0.03])
    button_ax = plt.axes([0.6, 0.45, 0.2, 0.04])
    save_button = Button(button_ax, 'Save Data')

    t_max = 0

    if len(data1.shape) == 5:
        t_max = data1.shape[0] - 1

    time_slider = Slider(
        ax=slider_ax_time,
        label='time',
        valmin=0,
        valmax=t_max,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    offset1x_slider = Slider(
        ax=slider_ax_offset1x,
        label='offset1x',
        valmin=0,
        valmax=10 * subpixel_accuracy,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    offset1y_slider = Slider(
        ax=slider_ax_offset1y,
        label='offset1y',
        valmin=0,
        valmax=10 * subpixel_accuracy,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    offset2x_slider = Slider(
        ax=slider_ax_offset2x,
        label='offset2x',
        valmin=0,
        valmax=10 * subpixel_accuracy,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    offset2y_slider = Slider(
        ax=slider_ax_offset2y,
        label='offset2y',
        valmin=0,
        valmax=10 * subpixel_accuracy,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    if len(data1.shape) == 5:
        w1max = data1.shape[4] - 1
        w2max = data2.shape[4] - 1
    else:
        w1max = data1.shape[3] - 1
        w2max = data2.shape[3] - 1
    
    wave1_slider = Slider(
        ax=slider_ax_wave1,
        label='wave1',
        valmin=0,
        valmax=w1max,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    wave2_slider = Slider(
        ax=slider_ax_wave2,
        label='wave2',
        valmin=0,
        valmax=w2max,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

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

    # Timer for flickering
    timer = fig.canvas.new_timer(interval=500)  # milliseconds
    timer.add_callback(flicker_callback)
    timer.start()

    def update_timer():
        a, b = get_final_images(
            data1, data2, time[0],
            offset_1_x[0], offset_1_y[0],
            offset_2_x[0], offset_2_y[0],
            wave1[0], wave2[0],
            subpixel_accuracy
        )
        final_image_1[0] = a

        final_image_2[0] = b

        corr = get_correlation_value(
            data1, data2,
            time[0], wave1[0], wave2[0],
            offset_1_x[0], offset_1_y[0],
            offset_2_x[0], offset_2_y[0],
            final_image_1[0], final_image_2[0],
            subpixel_accuracy
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
        w1_text.set_text('w1={}'.format(wave1[0]))
        w2_text.set_text('w2={}'.format(wave2[0]))
        o1x_text.set_text('o1x={}'.format(offset_1_x[0]))
        o1y_text.set_text('o1y={}'.format(offset_1_y[0]))
        o2x_text.set_text('o2x={}'.format(offset_2_x[0]))
        o2y_text.set_text('o2y={}'.format(offset_2_y[0]))
        corr_text.set_text('Pearson Corr: {}'.format(corr))

    def do_save(event):
        align_and_merge_ca_he_data_streaming(
            base_path, filename1, filename2, subpixel_accuracy,
            offset_1_y[0], offset_1_x[0], offset_2_y[0], offset_2_x[0]
        )
    
    def update_wave1(val):
        wave1[0] = val
        update_timer()

    def update_wave2(val):
        wave2[0] = val
        update_timer()

    def update_offset_1x(val):
        offset_1_x[0] = val
        update_timer()

    def update_offset_1y(val):
        offset_1_y[0] = val
        update_timer()

    def update_offset_2x(val):
        offset_2_x[0] = val
        update_timer()

    def update_offset_2y(val):
        offset_2_y[0] = val
        update_timer()

    def update_time(val):
        time[0] = val
        update_timer()

    wave1_slider.on_changed(update_wave1)

    wave2_slider.on_changed(update_wave2)

    time_slider.on_changed(update_time)
    offset1x_slider.on_changed(update_offset_1x)
    offset1y_slider.on_changed(update_offset_1y)
    offset2x_slider.on_changed(update_offset_2x)
    offset2y_slider.on_changed(update_offset_2y)
    # align_button.on_clicked(do_subpixel_alignment)
    save_button.on_clicked(do_save)

    plt.subplots_adjust(left=0.2, right=0.99, bottom=0.4, top=0.99, wspace=0, hspace=0)

    plt.show()


if __name__ == '__main__':

    base_path = Path('/mnt/f/GRIS')

    filename1 = '25Apr25ARM1-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    filename2 = '25Apr25ARM2-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    subpixel_accuracy = 1

    animate(base_path, filename1, filename2, subpixel_accuracy)
