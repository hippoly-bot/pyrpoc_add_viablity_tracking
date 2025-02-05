import numpy as np
import math
import tkinter as tk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import gui

def create_axes(gui, n_channels):
    # unchanged ...
    ...

def display_data(gui, data_list):
    if len(data_list) == 0:
        return

    n_channels = len(data_list)
    if not gui.channel_axes or (len(gui.channel_axes) != n_channels):
        create_axes(gui, n_channels)

    gui.data = data_list
    for i, orig_data in enumerate(data_list):
        data = np.squeeze(orig_data) if orig_data.ndim > 2 else orig_data

        ch_ax = gui.channel_axes[i]
        ax_main = ch_ax["main"]
        ny, nx = data.shape
        x_extent = np.linspace(
            gui.config['offset_x'] - gui.config['amp_x'],
            gui.config['offset_x'] + gui.config['amp_x'],
            nx
        )
        y_extent = np.linspace(
            gui.config['offset_y'] + gui.config['amp_y'],
            gui.config['offset_y'] - gui.config['amp_y'],
            ny
        )

        # set title from channel names
        if 'channel_names' in gui.config and len(gui.config['channel_names']) > i:
            channel_name = gui.config['channel_names'][i]
        else:
            channel_name = gui.config['ai_chans'][i] if i < len(gui.config['ai_chans']) else f"chan{i}"
        ax_main.set_title(channel_name, fontsize=10, color='white')

        if ch_ax["img_handle"] is None:
            im = ax_main.imshow(
                data,
                extent=[x_extent[0], x_extent[-1], y_extent[-1], y_extent[0]],  # flip Y
                origin='upper',  # or 'lower' if you want the first row at bottom
                aspect='equal',
                cmap='magma'
            )
            ch_ax["img_handle"] = im
            gui.slice_x[i] = nx // 2
            gui.slice_y[i] = ny // 2
            ch_ax["vline"] = ax_main.axvline(x=[x_extent[gui.slice_x[i]]], color='red', linestyle='--', lw=2)
            ch_ax["hline"] = ax_main.axhline(y=[y_extent[gui.slice_y[i]]], color='blue', linestyle='--', lw=2)

            cax = ax_main.inset_axes([1.05, 0, 0.05, 1])
            cb = gui.fig.colorbar(im, cax=cax, orientation='vertical')
            cb.set_label('Intensity', color='white')
            cb.ax.yaxis.set_tick_params(color='white', labelsize=8)
            cb.outline.set_edgecolor('white')
            for label in cb.ax.yaxis.get_ticklabels():
                label.set_color('white')
            ch_ax["colorbar"] = cb
        else:
            im = ch_ax["img_handle"]
            im.set_data(data)

            # colorbar logic
            channel_auto_scale_var = gui.auto_colorbar_vars.get(channel_name, tk.BooleanVar(value=True))
            auto_scale = channel_auto_scale_var.get()
            if auto_scale:
                im.set_clim(vmin=data.min(), vmax=data.max())
            else:
                try:
                    fixed_max_str = gui.fixed_colorbar_vars.get(channel_name, tk.StringVar(value="")).get()
                    fixed_max = float(fixed_max_str)
                    if fixed_max < data.min():
                        # if user typed a silly value, clamp
                        fixed_max = data.max()
                except ValueError:
                    # fallback if user typed something invalid
                    fixed_max = data.max()

                im.set_clim(vmin=data.min(), vmax=fixed_max)

            im.set_extent([x_extent[0], x_extent[-1], y_extent[-1], y_extent[0]])

        sx = gui.slice_x[i] if gui.slice_x[i] is not None and gui.slice_x[i] < nx else nx // 2
        sy = gui.slice_y[i] if gui.slice_y[i] is not None and gui.slice_y[i] < ny else ny // 2
        if ch_ax["vline"]:
            ch_ax["vline"].set_xdata([x_extent[sx]])
        if ch_ax["hline"]:
            ch_ax["hline"].set_ydata([y_extent[sy]])

        ax_hslice = ch_ax["hslice"]
        ax_hslice.clear()
        ax_hslice.plot(x_extent, data[sy, :], color='blue', linewidth=1)
        ax_hslice.yaxis.tick_right()
        ax_hslice.tick_params(axis='both', labelsize=8)
        ax_hslice.set_xlim(x_extent[0], x_extent[-1])

        ax_vslice = ch_ax["vslice"]
        ax_vslice.clear()
        ax_vslice.plot(data[:, sx], y_extent, color='red', linewidth=1)
        ax_vslice.tick_params(axis='both', labelsize=8)
        ax_vslice.set_ylim(y_extent[-1], y_extent[0])  # if we want the first row at top

    gui.canvas.draw_idle()
    
def on_image_click(gui, event):
    if str(gui.toolbar.mode) in ["zoom rect", "pan/zoom"]:
        return
    if not gui.channel_axes:
        return
    for i, ch_ax in enumerate(gui.channel_axes):
        if event.inaxes == ch_ax["main"]:
            data = gui.data[i]
            if data.ndim > 2:
                data = np.squeeze(data)
            ny, nx = data.shape
            x_extent = np.linspace(-gui.config['amp_x'], gui.config['amp_x'], nx)
            y_extent = np.linspace(-gui.config['amp_y'], gui.config['amp_y'], ny)
            gui.slice_x[i] = int(np.abs(x_extent - event.xdata).argmin())
            gui.slice_y[i] = int(np.abs(y_extent - event.ydata).argmin())
            if ch_ax["vline"] is not None:
                ch_ax["vline"].set_xdata([x_extent[gui.slice_x[i]]])
            if ch_ax["hline"] is not None:
                ch_ax["hline"].set_ydata([y_extent[gui.slice_y[i]]])
            current_data = gui.data if gui.data else []
            display_data(gui, current_data)
            return
