import threading
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image
from pysrs.helpers.utils import generate_data, convert
from pysrs.mains.display import display_data
from pysrs.helpers.galvo_funcs import Galvo
from pysrs.helpers.run_image_2d import run_scan
import pysrs.helpers.prior_stage.functions as prior

def reset_gui(gui):
    gui.running = False
    gui.acquiring = False
    gui.continuous_button['state'] = 'normal'
    gui.single_button['state'] = 'normal'
    gui.stop_button['state'] = 'disabled'
    gui.progress_label.config(text='(0/0)')

def acquire(gui, continuous=False, startup=False, auxilary=False):
    if (gui.running or gui.acquiring) and not (startup or auxilary):
        return

    gui.running = continuous
    gui.acquiring = True
    gui.stop_button['state'] = 'normal'
    gui.continuous_button['state'] = 'disabled'
    gui.single_button['state'] = 'disabled'

    try:
        while gui.running if continuous else True:
            gui.update_config()

            hyperspectral = gui.hyperspectral_enabled.get()
            zscan = gui.zscan_enabled.get()

            if hyperspectral and zscan:
                messagebox.showerror("Acquisition Conflict", "Cannot run both Hyperspectral and Z-Scan. Please uncheck one.")
                break

            save = gui.save_acquisitions.get()
            filename = gui.save_file_entry.get().strip() if save else None
            channels = [f"{gui.config['device']}/{ch}" for ch in gui.config['ai_chans']]

            if hyperspectral:
                num_steps_entry = gui.entry_numshifts
            elif zscan:
                num_steps_entry = gui.entry_z_steps
            else:
                num_steps_entry = gui.save_num_entry

            try:
                num_steps = int(num_steps_entry.get().strip())
                if num_steps < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror('Error', 'Invalid number of steps.')
                break

            if save and not filename:
                messagebox.showerror('Error', 'Please specify a valid TIFF filename.')
                break

            positions = [0] * num_steps  # dummy default

            if hyperspectral:
                try:
                    gui.zaber_stage.connect()
                    start = float(gui.entry_start_um.get().strip())
                    stop = float(gui.entry_stop_um.get().strip())
                    positions = [start + i * (stop - start) / (num_steps - 1) for i in range(num_steps)] if num_steps > 1 else [start]
                except Exception as e:
                    messagebox.showerror("Zaber Error", str(e))
                    break

            elif zscan:
                try:
                    port = int(gui.prior_port_entry.get().strip())
                    prior.connect_prior(port)
                    start = int(10*float(gui.entry_z_start.get().strip())) # convert to 100s of nms, prior stage native units
                    stop = int(10*float(gui.entry_z_stop.get().strip()))
                    positions = [start + i * (stop - start) / (num_steps - 1) for i in range(num_steps)] if num_steps > 1 else [start]
                except Exception as e:
                    messagebox.showerror("Prior Z-Stage Error", str(e))
                    break

            images = []
            for i in range(num_steps):
                if not gui.acquiring:
                    break

                if hyperspectral:
                    gui.zaber_stage.move_absolute_um(positions[i])
                elif zscan:
                    prior.move_z(port, int(positions[i]))

                galvo = Galvo(gui.config)
                data = acquire_single(gui, channels, galvo)
                if data is None:
                    break

                images.append(data)
                gui.progress_label.config(text=f'({i + 1}/{num_steps})')
                gui.root.update_idletasks()

            if save and images:
                save_images(gui, images, filename)

            if not continuous:
                break

    except Exception as e:
        reset_gui(gui)
        messagebox.showerror('Acquisition Error', f'Unexpected error:\n{e}')
    finally:
        if not auxilary:
            reset_gui(gui)


def acquire_single(gui, channels, galvo, move_z=None):
    if move_z is not None:
        try:
            gui.zaber_stage.move_absolute_um(move_z)
        except Exception as e:
            reset_gui(gui)
            messagebox.showerror('Stage Move Error', f'Error moving stage: {e}')
            return None

    try:
        if gui.simulation_mode.get():
            data_list = generate_data(len(channels), config=gui.config)
        else:
            mode = 'variable' if gui.var_dwell_var.get() else 'standard'

            mod_do_chans = []
            mod_masks = []

            if hasattr(gui, 'mod_enabled_vars') and hasattr(gui, 'mod_masks'):
                for i, enabled_var in enumerate(gui.mod_enabled_vars):
                    if enabled_var.get() and i in gui.mod_masks:
                        ttl_var = gui.mod_ttl_channel_vars[i].get()
                        mask = gui.mod_masks[i]
                        mod_do_chans.append(ttl_var)
                        mod_masks.append(mask)

            data_list = run_scan(
                ai_channels=channels,
                galvo=galvo,
                mode=mode,
                dwell_multiplier=gui.dwell_mult_var.get(),
                modulate=bool(mod_do_chans),  # only True if any channels enabled
                mod_do_chans=mod_do_chans,
                mod_masks=mod_masks,
            )

        gui.root.after(0, display_data, gui, data_list)
        return [convert(d) for d in data_list]

    except Exception as e:
        reset_gui(gui)
        messagebox.showerror('Acquisition Error', f'Error acquiring frame: {e}')
        return None



def save_images(gui, images, filename):
    if not images:
        return
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    base, ext = os.path.splitext(filename)
    num_channels = len(images[0])
    saved_fnames = []

    for ch_idx in range(num_channels):
        channel_frames = [frame[ch_idx] for frame in images]
        counter = 1

        if 'channel_names' in gui.config and ch_idx < len(gui.config['channel_names']):
            channel_suffix = gui.config['channel_names'][ch_idx]
        elif ch_idx < len(gui.config['ai_chans']):
            channel_suffix = gui.config['ai_chans'][ch_idx]
        else:
            channel_suffix = f"chan{ch_idx}"

        new_filename = f"{base}_{channel_suffix}{ext}"
        while os.path.exists(new_filename):
            new_filename = f"{base}_{channel_suffix}_{counter}{ext}"
            counter += 1

        if len(channel_frames) > 1:
            channel_frames[0].save(
                new_filename,
                save_all=True,
                append_images=channel_frames[1:],
                format='TIFF'
            )
        else:
            channel_frames[0].save(new_filename, format='TIFF')

        saved_fnames.append(new_filename)

    msg = "Saved frames:\n" + "\n".join(saved_fnames)
    messagebox.showinfo('Done', msg)
    gui.progress_label.config(text=f'(0/{len(images)})')
