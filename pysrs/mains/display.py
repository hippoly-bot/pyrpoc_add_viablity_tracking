import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import tkinter as tk

def create_axes(gui, n_channels):
    gui.fig.clf()
    gui.fig.patch.set_facecolor('#1E1E1E') 

    gui.channel_axes = []
    gui.slice_x = [None] * n_channels
    gui.slice_y = [None] * n_channels

    ncols = math.ceil(math.sqrt(n_channels))
    nrows = math.ceil(n_channels / ncols)

    for i in range(n_channels):
        ax_main = gui.fig.add_subplot(nrows, ncols, i+1)
        ax_main.set_facecolor('#1E1E1E')
        for spine in ax_main.spines.values():
            spine.set_color('white')
        ax_main.xaxis.label.set_color('white')
        ax_main.yaxis.label.set_color('white')
        ax_main.tick_params(axis='both', colors='white', labelsize=8)
        
        divider = make_axes_locatable(ax_main)
        ax_hslice = divider.append_axes("bottom", size="10%", pad=0.05, sharex=ax_main)
        ax_vslice = divider.append_axes("left", size="10%", pad=0.05, sharey=ax_main)

        for ax in [ax_hslice, ax_vslice]:
            ax.set_facecolor('#1E1E1E')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='both', colors='white', labelsize=8)

        ch_dict = {
            "main": ax_main,
            "hslice": ax_hslice,
            "vslice": ax_vslice,
            "img_handle": None,
            "colorbar": None,
            "vline": None,
            "hline": None,
        }
        gui.channel_axes.append(ch_dict)
    
    gui.canvas.draw()


def display_data(gui, data_list):
    if len(data_list) == 0:
        return

    n_channels = len(data_list)
    if (not gui.channel_axes) or (len(gui.channel_axes) != n_channels):
        create_axes(gui, n_channels)

    gui.data = data_list
    for i, orig_data in enumerate(data_list):
        data = np.squeeze(orig_data) if orig_data.ndim > 2 else orig_data

        ch_ax = gui.channel_axes[i]
        ax_main = ch_ax["main"]
        ny, nx = data.shape
        x_extent = np.linspace(-gui.config['amp_x'], gui.config['amp_x'], nx)
        y_extent = np.linspace(-gui.config['amp_y'], gui.config['amp_y'], ny)

        # **Ensure correct channel naming**
        if 'channel_names' in gui.config and len(gui.config['channel_names']) > i:
            channel_name = gui.config['channel_names'][i]
        else:
            channel_name = gui.config['ai_chans'][i] if i < len(gui.config['ai_chans']) else f"chan{i}"
        
        ax_main.set_title(channel_name, fontsize=10, color='white')

        if ch_ax["img_handle"] is None:
            im = ax_main.imshow(
                data,
                extent=[x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]],
                origin='lower',
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
                label.set_fontsize(8)
            ch_ax["colorbar"] = cb
        else:
            im = ch_ax["img_handle"]
            im.set_data(data)

            # **Use channel name for colorbar reference**
            auto_scale = gui.auto_colorbar_vars.get(channel_name, tk.BooleanVar(value=True)).get()
            
            if not auto_scale:
                try:
                    fixed_max = float(gui.fixed_colorbar_vars.get(channel_name, tk.StringVar(value="")).get())
                except Exception:
                    fixed_max = data.max()
                im.set_clim(vmin=data.min(), vmax=fixed_max)
            else:
                im.set_clim(vmin=data.min(), vmax=data.max())

            im.set_extent([x_extent[0], x_extent[-1], y_extent[0], y_extent[-1]])

        sx = gui.slice_x[i] if gui.slice_x[i] is not None and gui.slice_x[i] < nx else nx // 2
        sy = gui.slice_y[i] if gui.slice_y[i] is not None and gui.slice_y[i] < ny else ny // 2
        if ch_ax["vline"] is not None:
            ch_ax["vline"].set_xdata([x_extent[sx]])
        if ch_ax["hline"] is not None:
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
        ax_vslice.set_ylim(y_extent[0], y_extent[-1])

    gui.canvas.draw_idle()



def on_image_click(gui, event):
    if str(gui.toolbar.mode) in ["zoom rect", "pan/zoom"]:
        return
    if not gui.channel_axes:
        return

    for i, ch_ax in enumerate(gui.channel_axes):
        if event.inaxes == ch_ax["main"]:
            data = gui.data[i]
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]
            elif data.ndim > 2:
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
            
            current_data = gui.data if gui.data else [ax["img_handle"].get_array() for ax in gui.channel_axes]
            display_data(gui, current_data)
            return
