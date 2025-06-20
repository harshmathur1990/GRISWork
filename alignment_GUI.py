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


plt.switch_backend('QtAgg')



def refine_alignment_subpixel(img1, img2, final_image_2):
    """
    Computes subpixel offset required to align img2 to img1.
    Returns the subpixel shifts and the shifted image2.
    """
    # Compute shift between the two images
    shift_result, error, diffphase = phase_cross_correlation(img1, img2, space='real', overlap_ratio=0.9, upsample_factor=100)

    # Apply the computed shift to image 2
    img2_aligned = shift(final_image_2, shift=shift_result, mode='constant', cval=np.median(final_image_2), order=5)

    return shift_result, img2_aligned


def compute_fft_shift(img1, img2, upsample_factor=100):
    """
    Estimate subpixel shift between img1 and img2 using FFT cross-correlation.
    Returns (dy, dx) in pixel units.
    """
    # Ensure same shape
    assert img1.shape == img2.shape, "Images must have the same shape"

    # Cross-correlation via FFT
    corr = fftconvolve(img1, img2[::-1, ::-1], mode='same')

    # Find integer pixel peak
    max_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_y, peak_x = max_idx

    # Shift relative to center
    center_y, center_x = np.array(corr.shape) // 2
    dy_int = peak_y - center_y
    dx_int = peak_x - center_x

    # Extract 3×3 neighborhood around peak for interpolation
    y0, x0 = peak_y, peak_x
    if 1 < y0 < corr.shape[0]-2 and 1 < x0 < corr.shape[1]-2:
        window = corr[y0-1:y0+2, x0-1:x0+2]
        # Fit quadratic surface: z = a*x^2 + b*y^2 + c*x + d*y + e
        # We use Taylor expansion around the peak
        dx_subpix = (window[1,2] - window[1,0]) / (2*(2*window[1,1] - window[1,0] - window[1,2]))
        dy_subpix = (window[2,1] - window[0,1]) / (2*(2*window[1,1] - window[0,1] - window[2,1]))
    else:
        dx_subpix = 0
        dy_subpix = 0

    dx = dx_int + dx_subpix
    dy = dy_int + dy_subpix

    return (dy, dx)


def get_orig_images(data1, data2, time, wave1, wave2):

    if len(data1.shape) == 5:
        image1 = data1[time, 0, :, :, wave1]

        image2 = data2[time, 0, :, :, wave2]
    else:
        image1 = data1[0, :, :, wave1]

        image2 = data2[0, :, :, wave2]

    return image1, image2


def get_final_images(
    data1, data2, time,
    offset_1_x, offset_1_y,
    offset_2_x, offset_2_y,
    wave1, wave2
):

    image1, image2 = get_orig_images(data1, data2, time, wave1, wave2)

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
    final_image_1, final_image_2
):
    image1, image2 = get_orig_images(data1, data2, time, wave1, wave2)

    overlap_y0, overlap_x0, overlap_y1, overlap_x1 = get_overlapping_coordinates(
        image1, image2, offset_1_x, offset_1_y, offset_2_x, offset_2_y
    )

    region1 = final_image_1[overlap_y0:overlap_y1, overlap_x0:overlap_x1]
    region2 = final_image_2[overlap_y0:overlap_y1, overlap_x0:overlap_x1]

    mask = ~np.isnan(region1) & ~np.isnan(region2)
    corr = np.round(pearsonr(region1[mask].flatten(), region2[mask].flatten()).statistic, 4)

    return corr


def align_and_merge_ca_he_data(
        base_path, filename1, filename2,
        offset_1_y=0, offset_1_x=0, offset_2_y=0, offset_2_x=0
):

    data1, _ = sunpy.io.read_file(base_path / filename1)[0]

    data2, _ = sunpy.io.read_file(base_path / filename2)[0]

    print ("read data...")

    if len(data1.shape) == 5:
        index_y = 2
        index_x = 3
    else:
        index_y = 1
        index_x = 2

    ny = max(
        data1.shape[index_y] + offset_1_y,
        data2.shape[index_y] + offset_2_y
    )

    nx = max(
        data1.shape[index_x] + offset_1_x,
        data2.shape[index_x] + offset_2_x
    )

    print ("Starting to initialize arrays...")

    if len(data1.shape) == 5:
        aligned_data_1 = np.zeros((data1.shape[0], data1.shape[1], ny, nx, data1.shape[4]), dtype=np.float64)
        aligned_data_2 = np.zeros((data2.shape[0], data2.shape[1], ny, nx, data2.shape[4]), dtype=np.float64)
        d1 = np.transpose(data1, axes=(2, 3, 0, 1, 4))
        d2 = np.transpose(data2, axes=(2, 3, 0, 1, 4))
    else:
        aligned_data_1 = np.zeros((data1.shape[0], ny, nx, data1.shape[3]), dtype=np.float64)
        aligned_data_2 = np.zeros((data2.shape[0], ny, nx, data2.shape[3]), dtype=np.float64)
        d1 = np.transpose(data1, axes=(1, 2, 0, 3))
        d2 = np.transpose(data2, axes=(1, 2, 0, 3))

    print ("Arrays initialized, now calculating fill value...")

    fill_value_1 = np.mean(d1, axis=(0, 1))

    fill_value_2 = np.mean(d2, axis=(0, 1))

    print ("calculated dummy fill value, now filing them...")

    if len(data1.shape) == 5:
        aligned_data_1[:] = fill_value_1[:, :, np.newaxis, np.newaxis]

        aligned_data_2[:] = fill_value_2[:, :, np.newaxis, np.newaxis]
    else:
        aligned_data_1[:] = fill_value_1[:, np.newaxis, np.newaxis]

        aligned_data_2[:] = fill_value_2[:, np.newaxis, np.newaxis]

    print ("filled dummy data...now filling real values...")

    if len(data1.shape) == 5:
        aligned_data_1[:, :, offset_1_y:data1.shape[index_y] + offset_1_y, offset_1_x:data1.shape[index_x] + offset_1_x] = data1

        aligned_data_2[:, :, offset_2_y:data2.shape[index_y] + offset_2_y, offset_2_x:data2.shape[index_x] + offset_2_x] = data2
    else:
        aligned_data_1[:, offset_1_y:data1.shape[index_y] + offset_1_y, offset_1_x:data1.shape[index_x] + offset_1_x] = data1

        aligned_data_2[:, offset_2_y:data2.shape[index_y] + offset_2_y, offset_2_x:data2.shape[index_x] + offset_2_x] = data2
   
    print ("filled real data...saving...")

    sunpy.io.write_file(
        base_path / '{}_aligned.fits'.format(filename1),
        aligned_data_1,
        dict(),
        overwrite=True
    )

    sunpy.io.write_file(
        base_path / '{}_aligned.fits'.format(filename2),
        aligned_data_2,
        dict(),
        overwrite=True
    )

    print ("saved...")


def animate(base_path, filename1, filename2):

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
        wave1[0], wave2[0]
    )

    final_image_1 = [a]

    final_image_2 = [b]

    corr = get_correlation_value(
        data1, data2,
        time[0], wave1[0], wave2[0],
        offset_1_x[0], offset_1_y[0],
        offset_2_x[0], offset_2_y[0],
        final_image_1[0], final_image_2[0]
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
        valmax=10,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    offset1y_slider = Slider(
        ax=slider_ax_offset1y,
        label='offset1y',
        valmin=0,
        valmax=10,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    offset2x_slider = Slider(
        ax=slider_ax_offset2x,
        label='offset2x',
        valmin=0,
        valmax=10,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    offset2y_slider = Slider(
        ax=slider_ax_offset2y,
        label='offset2y',
        valmin=0,
        valmax=10,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    wave1_slider = Slider(
        ax=slider_ax_wave1,
        label='wave1',
        valmin=0,
        valmax=data1.shape[4] - 1,
        valinit=0,
        valstep=1,
        orientation='horizontal'
    )

    wave2_slider = Slider(
        ax=slider_ax_wave2,
        label='wave2',
        valmin=0,
        valmax=data2.shape[4] - 1,
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
            wave1[0], wave2[0]
        )
        final_image_1[0] = a

        final_image_2[0] = b

        corr = get_correlation_value(
            data1, data2,
            time[0], wave1[0], wave2[0],
            offset_1_x[0], offset_1_y[0],
            offset_2_x[0], offset_2_y[0],
            final_image_1[0], final_image_2[0]
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

    # def do_subpixel_alignment(event):

    #     image1, image2 = get_orig_images(data1, data2, time[0], wave1[0], wave2[0])

    #     overlap_y0, overlap_x0, overlap_y1, overlap_x1 = get_overlapping_coordinates(
    #         image1, image2, offset_1_x[0], offset_1_y[0], offset_2_x[0], offset_2_y[0]
    #     )

    #     region1 = final_image_1[0][overlap_y0:overlap_y1, overlap_x0:overlap_x1]
    #     region2 = final_image_2[0][overlap_y0:overlap_y1, overlap_x0:overlap_x1]

    #     # Subpixel alignment
    #     # shift_result, region2_aligned = refine_alignment_subpixel(region1, region2, final_image_2[0])

    #     dy, dx = compute_fft_shift(region1, region2)

    #     print ('{} - {}'.format(dy, dx))

    #     region2_aligned = shift(final_image_2[0], shift=(dy, dx), order=0, mode='constant', cval=np.median(image2))

    #     # Replace region2 in final_image_2
    #     final_image_2[0] = region2_aligned

    #     region2_aligned = final_image_2[0][overlap_y0:overlap_y1, overlap_x0:overlap_x1]
    #     # Update flicker logic with new aligned image2
    #     corr = np.round(pearsonr(region1.flatten(), region2_aligned.flatten()).statistic, 4)
        
    #     new_flicker_callback = prepare_flicker_callback(
    #         frame_toggle, im, fig, final_image_1[0], final_image_2[0], text_image
    #     )
    #     timer.stop()
    #     timer.callbacks = []
    #     timer.add_callback(flicker_callback)
    #     timer.start()
    #     t_text.set_text('t={}'.format(time[0]))
    #     w1_text.set_text('w1={}'.format(wave1[0]))
    #     w2_text.set_text('w2={}'.format(wave2[0]))
    #     o1x_text.set_text('o1x={}'.format(offset_1_x[0]))
    #     o1y_text.set_text('o1y={}'.format(offset_1_y[0]))
    #     o2x_text.set_text('o2x={}'.format(offset_2_x[0]))
    #     o2y_text.set_text('o2y={}'.format(offset_2_y[0]))
    #     corr_text.set_text('Pearson Corr: {}'.format(corr))

    def do_save(event):
        align_and_merge_ca_he_data(
            base_path, filename1, filename2,
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

    filename1 = '25Apr25ARM1-003.fits_squarred_pixels.fits'

    filename2 = '25Apr25ARM2-003.fits_squarred_pixels.fits'

    # filename1 = '25Apr25ARM1-003.fits_squarred_pixels.fits_aligned.fits'

    # filename2 = '25Apr25ARM2-003.fits_squarred_pixels.fits_aligned.fits'

    animate(base_path, filename1, filename2)
