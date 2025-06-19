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


plt.switch_backend('QtAgg')


def get_final_images(
    data1, data2, time,
    offset_1_x, offset_1_y,
    offset_2_x, offset_2_y,
    wave1, wave2
):

    image1 = data1[time, 0, :, :, wave1]

    image2 = data2[time, 0, :, :, wave2]

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

    corr = 0

    if overlap_y1 > overlap_y0 and overlap_x1 > overlap_x0:
        region1 = final_image_1[overlap_y0:overlap_y1, overlap_x0:overlap_x1]
        region2 = final_image_2[overlap_y0:overlap_y1, overlap_x0:overlap_x1]

        mask = ~np.isnan(region1) & ~np.isnan(region2)
        corr = pearsonr(region1[mask].flatten(), region2[mask].flatten()).statistic

    return final_image_1, final_image_2, np.round(corr, 4)


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

    final_image_1, final_image_2, corr = get_final_images(
        data1, data2, time[0],
        offset_1_x[0], offset_1_y[0],
        offset_2_x[0], offset_2_y[0],
        wave1[0], wave2[0]
    )

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(1, 1, figsize=(7, 9))

    im = axs.imshow(final_image_1, cmap='gray', origin='lower')

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

    slider_ax_time = plt.axes([0.1, 0.33, 0.8, 0.03])
    slider_ax_offset1x = plt.axes([0.1, 0.28, 0.8, 0.03])
    slider_ax_offset1y = plt.axes([0.1, 0.23, 0.8, 0.03])
    slider_ax_offset2x = plt.axes([0.1, 0.18, 0.8, 0.03])
    slider_ax_offset2y = plt.axes([0.1, 0.13, 0.8, 0.03])
    slider_ax_wave2 = plt.axes([0.1, 0.03, 0.8, 0.03])
    slider_ax_wave1 = plt.axes([0.1, 0.08, 0.8, 0.03])

    time_slider = Slider(
        ax=slider_ax_time,
        label='time',
        valmin=0,
        valmax=30,
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


    def prepare_flicker_callback(frame_toggle, im, fig, final_image_1, final_image_2):
        def flicker_callback(*args):

            if frame_toggle[0] == 0:
                frame = final_image_1
            else:
                frame = final_image_2
    
            im.set_array(frame)
            mn, mx = np.nanmin(frame) * 0.9, np.nanmax(frame) * 1.1
            im.set_clim(mn, mx)
            fig.canvas.draw_idle()
            frame_toggle[0] = 1 - frame_toggle[0]
        return flicker_callback

    flicker_callback = prepare_flicker_callback(frame_toggle, im, fig, final_image_1, final_image_2)

    # Timer for flickering
    timer = fig.canvas.new_timer(interval=500)  # milliseconds
    timer.add_callback(flicker_callback)
    timer.start()

    def update_timer():
        final_image_1, final_image_2, corr = get_final_images(
            data1, data2, time[0],
            offset_1_x[0], offset_1_y[0],
            offset_2_x[0], offset_2_y[0],
            wave1[0], wave2[0]
        )
        flicker_callback = prepare_flicker_callback(
            frame_toggle, im, fig,
            final_image_1, final_image_2
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

    plt.subplots_adjust(left=0.2, right=0.99, bottom=0.4, top=0.99, wspace=0, hspace=0)

    plt.show()


if __name__ == '__main__':

    base_path = Path('/mnt/f/GRIS')

    filename1 = '25Apr25ARM1-003.fits_squarred_pixels.fits'

    filename2 = '25Apr25ARM2-003.fits_squarred_pixels.fits'

    animate(base_path, filename1, filename2)
