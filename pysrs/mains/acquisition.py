import threading
import os
from tkinter import messagebox
import numpy as np
from PIL import Image
from pysrs.helpers.utils import generate_data, convert
from pysrs.mains.display import display_data
from pysrs.helpers.galvo_funcs import Galvo
from pysrs.helpers.run_image_2d import raster_scan, raster_scan_rpoc, variable_scan_rpoc

def reset_gui(gui):
    gui.running = False
    gui.acquiring = False
    gui.continuous_button['state'] = 'normal'
    gui.single_button['state'] = 'normal'
    gui.stop_button['state'] = 'disabled'
    gui.progress_label.config(text='(0/0)')

def acquire(gui, continuous=False, startup=False, auxilary=False):
    if (gui.running or gui.acquiring) and not (startup or auxilary):
        return  # Prevent acquisition if already running

    gui.running = continuous
    gui.acquiring = True
    gui.stop_button['state'] = 'normal'
    gui.continuous_button['state'] = 'disabled'
    gui.single_button['state'] = 'disabled'

    try:
        while gui.running if continuous else True:
            gui.update_config()
            hyperspectral = gui.hyperspectral_enabled.get()
            save = gui.save_acquisitions.get()
            numshifts_str = (gui.entry_numshifts.get().strip() if hyperspectral 
                             else gui.save_num_entry.get().strip())
            filename = gui.save_file_entry.get().strip() if save else None

            if save and not filename:
                messagebox.showerror('Error', 'Please specify a valid TIFF filename.')
                break

            try:
                numshifts = int(numshifts_str)
                if numshifts < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror('Error', 'Invalid number of steps.')
                break

            try:
                if hyperspectral:
                    images = acquire_hyperspectral(gui, numshifts)
                else:
                    images = acquire_multiple(gui, numshifts)
                if save and images:
                    save_images(gui, images, filename)
            except Exception as e:
                if continuous:
                    gui.running = False 
                messagebox.showerror('Acquisition Error', f"An error occurred during acquisition:\n{e}")
                break  
            if not continuous: 
                break
    except Exception as e:
        reset_gui(gui)
        messagebox.showerror('Acquisition Error', f"Error acquiring in acquire():\n{e}")
        return None
    finally:
        if not auxilary:
            reset_gui(gui)

def acquire_multiple(gui, numshifts):
    images = []
    gui.progress_label.config(text=f'(0/{numshifts})')
    gui.root.update_idletasks()

    channels = [f"{gui.config['device']}/{ch}" for ch in gui.config['ai_chans']]
    galvo = Galvo(gui.config)

    for i in range(numshifts):
        if not gui.acquiring:
            break
        try:
            if gui.simulation_mode.get():
                data_list = generate_data(len(channels), config=gui.config)
            else:
                # Check if RPOC is enabled
                if gui.rpoc_mask is not None and gui.rpoc_enabled.get():
                    if gui.modulate_lasers_var.get():
                        # Modulated lasers: gather TTL channel strings and masks.
                        mod_do_chans = ([var.get() for var in gui.mod_ttl_channel_vars]
                                        if hasattr(gui, 'mod_ttl_channel_vars') else None)
                        mod_masks = ([gui.mod_masks[i] for i in range(len(gui.mod_ttl_channel_vars))]
                                     if hasattr(gui, 'mod_masks') else None)
                        if gui.rpoc_mode_var.get() == 'standard': 
                            data_list = raster_scan_rpoc(channels, galvo, gui.rpoc_mask,
                                                          modulate=True,
                                                          mod_do_chans=mod_do_chans,
                                                          mod_masks=mod_masks)
                        elif gui.rpoc_mode_var.get() == 'variable':
                            data_list = variable_scan_rpoc(channels, galvo, gui.rpoc_mask,
                                                            dwell_multiplier=gui.dwell_mult_var.get(),
                                                            modulate=True,
                                                            mod_masks=mod_masks)
                    else:
                        # Legacy RPOC behavior: single TTL channel.
                        if gui.rpoc_mode_var.get() == 'standard': 
                            data_list = raster_scan_rpoc(channels, galvo, gui.rpoc_mask,
                                                          do_chan=gui.mask_ttl_channel_var.get())
                        elif gui.rpoc_mode_var.get() == 'variable':
                            data_list = variable_scan_rpoc(channels, galvo, gui.rpoc_mask,
                                                            dwell_multiplier=gui.dwell_mult_var.get())
                else:
                    data_list = raster_scan(channels, galvo)

            gui.root.after(0, display_data, gui, data_list)
            pil_images = [convert(d) for d in data_list]
            images.append(pil_images)

            gui.progress_label.config(text=f'({i + 1}/{numshifts})')
            gui.root.update_idletasks()
        except Exception as e:
            reset_gui(gui)
            messagebox.showerror('Acquisition Error', f'Error acquiring in acquire_multiple(): {e}')
            return None

    return images

def acquire_hyperspectral(gui, numshifts):
    start_val = float(gui.entry_start_um.get().strip())
    stop_val = float(gui.entry_stop_um.get().strip())
    positions = ([start_val] if numshifts == 1 else
                 [start_val + i*(stop_val - start_val)/(numshifts-1) for i in range(numshifts)])

    try:
        gui.zaber_stage.connect()
    except Exception as e:
        messagebox.showerror("Zaber Error", str(e))
        return None

    images = []
    gui.progress_label.config(text=f'(0/{numshifts})')
    gui.root.update_idletasks()

    channels = [f"{gui.config['device']}/{ch}" for ch in gui.config['ai_chans']]

    for i, pos in enumerate(positions):
        if not gui.acquiring:
            break
        try:
            gui.zaber_stage.move_absolute_um(pos)
        except Exception as e:
            reset_gui(gui)
            messagebox.showerror('Stage Move Error', f'Error acquiring in acquire_hyperspectral(): {e}')
            return None

        galvo = Galvo(gui.config)
        if gui.simulation_mode.get():
            data_list = generate_data(len(channels), config=gui.config)
        else:
            if gui.rpoc_mask is not None:
                if gui.modulate_lasers_var.get():
                    mod_do_chans = ([var.get() for var in gui.mod_ttl_channel_vars]
                                    if hasattr(gui, 'mod_ttl_channel_vars') else None)
                    mod_masks = ([gui.mod_masks[i] for i in range(len(gui.mod_ttl_channel_vars))]
                                 if hasattr(gui, 'mod_ttl_channel_vars') and hasattr(gui, 'mod_masks') else None)
                    if gui.rpoc_mode_var.get() == 'standard': 
                        data_list = raster_scan_rpoc(channels, galvo, gui.rpoc_mask,
                                                      modulate=True,
                                                      mod_do_chans=mod_do_chans,
                                                      mod_masks=mod_masks)
                    elif gui.rpoc_mode_var.get() == 'variable':
                        data_list = variable_scan_rpoc(channels, galvo, gui.rpoc_mask,
                                                        dwell_multiplier=gui.dwell_mult_var.get(),
                                                        modulate=True,
                                                        mod_masks=mod_masks)
                else:
                    if gui.rpoc_mode_var.get() == 'standard': 
                        data_list = raster_scan_rpoc(channels, galvo, gui.rpoc_mask,
                                                      do_chan=gui.mask_ttl_channel_var.get())
                    elif gui.rpoc_mode_var.get() == 'variable':
                        data_list = variable_scan_rpoc(channels, galvo, gui.rpoc_mask,
                                                        dwell_multiplier=gui.dwell_mult_var.get())
            else:
                data_list = raster_scan(channels, galvo)
        gui.root.after(0, display_data, gui, data_list)
        pil_images = [convert(d) for d in data_list]
        images.append(pil_images)

        gui.progress_label.config(text=f'({i + 1}/{numshifts})')
        gui.root.update_idletasks()

    return images

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
