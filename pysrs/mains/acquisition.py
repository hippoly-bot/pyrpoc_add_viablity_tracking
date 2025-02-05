import threading, time, os
from tkinter import messagebox
import numpy as np
from PIL import Image
from utils import *
from display import *
from pysrs.instruments.galvo_funcs import Galvo
from pysrs.runners.run_image_2d import lockin_scan

def start_scan(gui):
    if gui.running:
        messagebox.showwarning('Warning', 'Scan is already running.')
        return
        
    rpoc_mask = None
    ttl_channel = None

    if gui.rpoc_enabled.get() and gui.apply_mask_var.get():
        if hasattr(gui, 'rpoc_mask') and gui.rpoc_mask is not None:
            rpoc_mask = gui.rpoc_mask
            ttl_channel = gui.mask_ttl_channel_var.get().strip()
        else:
            messagebox.showerror("Mask Error", "No valid mask loaded. Please load or create a mask.")
            return

    gui.running = True
    gui.continuous_button['state'] = 'disabled'
    gui.stop_button['state'] = 'normal'

    threading.Thread(target=scan, args=(gui,), kwargs={'rpoc_mask': rpoc_mask, 'ttl_channel': ttl_channel}, daemon=True).start()

def stop_scan(gui):
    gui.running = False
    gui.acquiring = False
    gui.continuous_button['state'] = 'normal'
    gui.stop_button['state'] = 'disabled'
    gui.single_button['state'] = 'normal'

def scan(gui, rpoc_mask=None, ttl_channel=None):
    try:
        while gui.running:
            gui.update_config()
            channels = [f"{gui.config['device']}/{ch}" for ch in gui.config['ai_chans']]
            galvo = Galvo(gui.config, rpoc_mask=rpoc_mask, ttl_channel=ttl_channel)
            if gui.simulation_mode.get():
                data_list = generate_data(len(channels), config=gui.config)
            else:
                data_list = lockin_scan(channels, galvo)
            gui.root.after(0, display_data, gui, data_list)
    except Exception as e:
        messagebox.showerror('Error', f'Cannot display data: {e}')
    finally:
        gui.running = False
        gui.continuous_button['state'] = 'normal'
        gui.stop_button['state'] = 'disabled'


def acquire(gui, startup=False):
    if gui.running and not startup:
        messagebox.showwarning('Warning',
            'Stop continuous acquisition first before saving or single acquisitions.')
        return

    gui.acquiring = True
    gui.stop_button['state'] = 'normal'

    try:
        gui.update_config()
        if gui.hyperspectral_enabled.get() and gui.save_acquisitions.get():
            numshifts_str = gui.entry_numshifts.get().strip()
            filename = gui.save_file_entry.get().strip()
            if not filename:
                messagebox.showerror('Error', 'Please specify a valid TIFF filename.')
                return
        elif gui.hyperspectral_enabled.get() and not gui.save_acquisitions.get():
            numshifts_str = gui.entry_numshifts.get().strip()
            filename = None
        elif not gui.hyperspectral_enabled.get() and gui.save_acquisitions.get():
            numshifts_str = gui.save_num_entry.get().strip()
            filename = gui.save_file_entry.get().strip()
            if not filename:
                messagebox.showerror('Error', 'Please specify a valid TIFF filename.')
                return
        else:
            numshifts_str = gui.save_num_entry.get().strip()
            filename = None

        try:
            numshifts = int(numshifts_str)
            if numshifts < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror('Error', 'Invalid number of steps.')
            return

        if not gui.hyperspectral_enabled.get():
            images = acquire_multiple(gui, numshifts)
            if gui.save_acquisitions.get() and images:
                save_images(gui, images, filename)
        else:
            images = acquire_hyperspectral(gui, numshifts)
            if gui.save_acquisitions.get() and images:
                save_images(gui, images, filename)
    except Exception as e:
        messagebox.showerror('Error', f'Cannot collect/save data: {e}')
    finally:
        gui.acquiring = False
        gui.stop_button['state'] = 'disabled'

def acquire_multiple(gui, numshifts):
    numframes = numshifts
    images = []
    gui.progress_label.config(text=f'(0/{numframes})')
    gui.root.update_idletasks()
    channels = [f"{gui.config['device']}/{ch}" for ch in gui.config['ai_chans']]
    galvo = Galvo(gui.config)
    for i in range(numframes):
        if not gui.acquiring:
            break
        if gui.simulation_mode.get():
            data_list = generate_data(len(channels), config=gui.config)
        else:
            data_list = lockin_scan(channels, galvo)
        gui.root.after(0, display_data, gui, data_list)
        pil_images = [convert(d) for d in data_list]
        images.append(pil_images)
        gui.progress_label.config(text=f'({i + 1}/{numframes})')
        gui.root.update_idletasks()
    return images

def acquire_hyperspectral(gui, numshifts):
    start_val = float(gui.entry_start_um.get().strip())
    stop_val = float(gui.entry_stop_um.get().strip())
    positions = [start_val] if numshifts == 1 else [start_val + i * (stop_val - start_val) / (numshifts - 1) for i in range(numshifts)]
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
            messagebox.showerror("Stage Move Error", str(e))
            return None
        galvo = Galvo(gui.config)
        if gui.simulation_mode.get():
            data_list = generate_data(len(channels), config=gui.config)
        else:
            data_list = lockin_scan(channels, galvo)
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
        # fallback to "chan{ch_idx}"
        if 'channel_names' in gui.config and len(gui.config['channel_names']) > ch_idx:
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