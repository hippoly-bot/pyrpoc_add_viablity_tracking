import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping
from nidaqmx.errors import DaqWarning
import numpy as np
from pysrs.helpers.galvo_funcs import Galvo
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw, ImageOps
import warnings
warnings.filterwarnings("ignore", category=DaqWarning, message=".*200011.*")

def run_scan(ai_channels, galvo, mode="standard", mask=None, dwell_multiplier=2.0,
             modulate=False, mod_do_chans=None, mod_masks=None):
    if isinstance(ai_channels, str):
        ai_channels = [ai_channels]

    # needs to have modulation on in general, have channels, have masks, and have their lengths be the same
    has_mods = modulate and mod_do_chans and mod_masks and (len(mod_do_chans) == len(mod_masks))

    if mode =='variable':
        # use first modulation mask if available; otherwise use global mask
        gen_mask = mod_masks[0] if has_mods else mask
        if gen_mask is None:
            raise ValueError("Variable dwell mode requires a valid mask.")
        if isinstance(gen_mask, Image.Image):
            gen_mask = np.array(gen_mask)
        gen_mask = gen_mask > 128

        x_wave, y_wave, pixel_map = galvo.gen_variable_waveform(gen_mask, dwell_multiplier)
        composite_wave = np.vstack([x_wave, y_wave])
        total_samps = len(x_wave)
    else:
        composite_wave = galvo.waveform.copy()
        total_samps = galvo.total_samples

    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
        for chan in galvo.ao_chans:
            ao_task.ao_channels.add_ao_voltage_chan(f"{galvo.device}/{chan}")
        for ch in ai_channels:
            ai_task.ai_channels.add_ai_voltage_chan(ch)

        ao_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samps
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            source=f"/{galvo.device}/ao/SampleClock",
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=total_samps
        )

        if has_mods:
            ttl_signals = []
            for m in mod_masks:
                m_arr = np.array(m) if isinstance(m, Image.Image) else m
                m_arr = m_arr > 0.5
                padded = []
                for row in range(galvo.numsteps_y):
                    padded_row = np.concatenate((
                        np.zeros(galvo.extrasteps_left, dtype=bool),
                        m_arr[row, :],
                        np.zeros(galvo.extrasteps_right, dtype=bool)
                    ))
                    padded.append(padded_row)
                flat = np.repeat(np.array(padded).ravel(), galvo.pixel_samples).astype(bool)
                ttl_signals.append(flat)

            if len(mod_do_chans) == 1:
                line = mod_do_chans[0]
                do_task.do_channels.add_do_chan(f"{galvo.device}/{line}")
                do_task.timing.cfg_samp_clk_timing(
                    rate=galvo.rate,
                    source=f"/{galvo.device}/ao/SampleClock",
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samps
                )
                do_task.write(ttl_signals[0].tolist(), auto_start=False)

            else:
                line_string = ",".join([f"{galvo.device}/{chan}" for chan in mod_do_chans])
                do_task.do_channels.add_do_chan(
                    line_string,
                    line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
                )
                do_task.timing.cfg_samp_clk_timing(
                    rate=galvo.rate,
                    source=f"/{galvo.device}/ao/SampleClock",
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=total_samps
                )

                ttl_matrix = np.array(ttl_signals, dtype=bool)
                packed_ttl = np.zeros(total_samps, dtype=np.uint8)
                for bit, line in enumerate(ttl_matrix):
                    packed_ttl |= (line.astype(np.uint8) << bit)

                do_task.write(packed_ttl.tolist(), auto_start=False)

        ao_task.write(composite_wave, auto_start=False)
        ai_task.start()
        if has_mods:
            do_task.start()
        ao_task.start()

        ao_task.wait_until_done(timeout=total_samps / galvo.rate + 5)
        ai_task.wait_until_done(timeout=total_samps / galvo.rate + 5)
        if has_mods:
            do_task.wait_until_done(timeout=total_samps / galvo.rate + 5)

        acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samps))

    results = []
    for i in range(len(ai_channels)):
        channel_data = acq_data if len(ai_channels) == 1 else acq_data[i]
        if mode =='variable':
            pixel_values = partition_and_average(channel_data, gen_mask, pixel_map, galvo)
        else:
            reshaped = channel_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
            pixel_values = np.mean(reshaped, axis=2)
        cropped = pixel_values[:, galvo.extrasteps_left:galvo.extrasteps_left + galvo.numsteps_x]
        results.append(cropped)

    return results


@staticmethod
def partition_and_average(ai_data_1d, mask, pixel_map, galvo):
    num_y, total_x = pixel_map.shape
    pixel_values_2d = np.zeros((num_y, total_x), dtype=float)
    cursor = 0
    for row_idx in range(num_y):
        for col_idx in range(total_x):
            samps = pixel_map[row_idx, col_idx]
            pixel_block = ai_data_1d[cursor:cursor + samps]
            cursor += samps
            pixel_values_2d[row_idx, col_idx] = np.mean(pixel_block)
    return pixel_values_2d

import os
if __name__ == "__main__":
    config = {
        "numsteps_x": 512,
        "numsteps_y": 512,
        "extrasteps_left": 200,
        "extrasteps_right": 20,
        "offset_x": 0.5,
        "offset_y": 0.4,
        "dwell": 5e-5,
        "amp_x": 0.75,
        "amp_y": 0.75,
        "rate": 100000,
        "device": 'Dev1',
        "ao_chans": ['ao1', 'ao0'],
    }

    galvo = Galvo(config)

    mask_path = r"C:\Users\Lab Admin\Documents\Python Scripts\new_pysrs\lefton_rightoff.png"
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Test mask not found at {mask_path}")
    
    mask_img = Image.open(mask_path).convert('L')
    mask = (np.array(mask_img.resize((galvo.numsteps_x, galvo.numsteps_y), Image.NEAREST))).astype(bool)

    plt.imshow(mask)
    plt.show()

    padded_mask = []
    for row in range(galvo.numsteps_y):
        padded_row = np.concatenate((
            np.zeros(galvo.extrasteps_left, dtype=bool),
            mask[row, :],
            np.zeros(galvo.extrasteps_right, dtype=bool)
        ))
        padded_mask.append(padded_row)
    
    padded_mask = np.array(padded_mask)
    flattened = np.repeat(padded_mask.ravel(), galvo.pixel_samples)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].imshow(padded_mask, cmap='gray', interpolation='nearest')
    axs[0].set_title("Padded Mask (2D view)")

    axs[1].plot(flattened, drawstyle='steps-pre') 
    axs[1].set_title("Flattened TTL Waveform (first 300 samples)")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_xlabel("Sample Index")

    plt.tight_layout()
    plt.show()


'''old functions, helpful to have for reference though'''
# def raster_scan(ai_channels, galvo):
#     if isinstance(ai_channels, str):
#         ai_channels = [ai_channels]

#     with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
#         ao_channels = list(galvo.ao_chans)
#         composite_wave = galvo.waveform.copy()

#         for chan in ao_channels:
#             ao_task.ao_channels.add_ao_voltage_chan(f'{galvo.device}/{chan}')
#         ao_task.timing.cfg_samp_clk_timing(
#             rate=galvo.rate,
#             sample_mode=AcquisitionType.FINITE,
#             samps_per_chan=composite_wave.shape[1]
#         )

#         for ch in ai_channels:
#             ai_task.ai_channels.add_ai_voltage_chan(ch)

#         ao_task.timing.cfg_samp_clk_timing(
#             rate=galvo.rate,
#             sample_mode=AcquisitionType.FINITE,
#             samps_per_chan=galvo.total_samples
#         )
#         ai_task.timing.cfg_samp_clk_timing(
#             rate=galvo.rate,
#             source=f'/{galvo.device}/ao/SampleClock',
#             sample_mode=AcquisitionType.FINITE,
#             samps_per_chan=galvo.total_samples
#         )

#         ao_task.write(composite_wave, auto_start=False)
#         ai_task.start()
#         ao_task.start()

#         ao_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
#         ai_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)

#         acq_data = np.array(ai_task.read(number_of_samples_per_channel=galvo.total_samples))

#     n_ch = len(ai_channels)
#     results = []

#     if n_ch == 1:
#         # shape is (total_y, total_x, pixel_samples), dont forget that u dummy
#         acq_data = acq_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
#         data2d = np.mean(acq_data, axis=2)

#         # crop out extrasteps_left and extrasteps_right
#         x1 = galvo.extrasteps_left
#         x2 = galvo.extrasteps_left + galvo.numsteps_x
#         cropped = data2d[:, x1:x2]
#         return [cropped]
#     else:
#         for i in range(n_ch):
#             chan_data = acq_data[i].reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
#             data2d = np.mean(chan_data, axis=2)

#             x1 = galvo.extrasteps_left
#             x2 = galvo.extrasteps_left + galvo.numsteps_x
#             cropped = data2d[:, x1:x2]
#             results.append(cropped)
#         return results

# def raster_scan_rpoc(ai_channels, galvo, mask, do_chan="port0/line5", modulate=False, mod_do_chans=None, mod_masks=None):
#     if isinstance(ai_channels, str):
#         ai_channels = [ai_channels]

#     with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
#         ao_channels = list(galvo.ao_chans)
#         composite_wave = galvo.waveform.copy()  
#         for chan in ao_channels:
#             ao_task.ao_channels.add_ao_voltage_chan(f'{galvo.device}/{chan}')
#         ao_task.timing.cfg_samp_clk_timing(
#             rate=galvo.rate,
#             sample_mode=AcquisitionType.FINITE,
#             samps_per_chan=composite_wave.shape[1]
#         )

#         for ch in ai_channels:
#             ai_task.ai_channels.add_ai_voltage_chan(ch)
#         ai_task.timing.cfg_samp_clk_timing(
#             rate=galvo.rate,
#             source=f'/{galvo.device}/ao/SampleClock',
#             sample_mode=AcquisitionType.FINITE,
#             samps_per_chan=galvo.total_samples
#         )

#         if modulate and mod_do_chans and mod_masks:
#             if len(mod_do_chans) != len(mod_masks):
#                 raise ValueError("The number of modulation DO channels and masks must match.")
#             ttl_signals = []
#             for m in mod_masks:
#                 if isinstance(m, Image.Image):
#                     m_arr = np.array(m)
#                 else:
#                     m_arr = m
#                 if m_arr.shape[0] != galvo.numsteps_y or m_arr.shape[1] != galvo.numsteps_x:
#                     m_arr = np.array(Image.fromarray(m_arr.astype(np.uint8)*255).resize((galvo.numsteps_x, galvo.numsteps_y), Image.NEAREST)) > 128
#                 padded_mask = []
#                 for row_idx in range(galvo.numsteps_y):
#                     row_data = m_arr[row_idx, :]
#                     padded_row = np.concatenate((
#                         np.zeros(galvo.extrasteps_left, dtype=bool),
#                         row_data,
#                         np.zeros(galvo.extrasteps_right, dtype=bool)
#                     ))
#                     padded_mask.append(padded_row)
#                 padded_mask = np.array(padded_mask, dtype=bool)
#                 flattened = padded_mask.ravel()
#                 ttl_signal = np.repeat(flattened, galvo.pixel_samples).astype(bool)
#                 ttl_signals.append(ttl_signal)
#             for chan in mod_do_chans:
#                 do_task.do_channels.add_do_chan(f"{galvo.device}/{chan}")
#             do_task.timing.cfg_samp_clk_timing(
#                 rate=galvo.rate,
#                 source=f'/{galvo.device}/ao/SampleClock',
#                 sample_mode=AcquisitionType.FINITE,
#                 samps_per_chan=galvo.total_samples
#             )
#             do_task.write(ttl_signals, auto_start=False)
#         else:
#             if isinstance(mask, Image.Image):
#                 mask = np.array(mask)
#             if not isinstance(mask, np.ndarray):
#                 raise TypeError('Mask must be a numpy array or PIL Image.')
#             padded_mask = []
#             for row_idx in range(galvo.numsteps_y):
#                 row_data = mask[row_idx, :]
#                 padded_row = np.concatenate((
#                     np.zeros(galvo.extrasteps_left, dtype=bool),
#                     row_data,
#                     np.zeros(galvo.extrasteps_right, dtype=bool)
#                 ))
#                 padded_mask.append(padded_row)
#             padded_mask = np.array(padded_mask, dtype=bool)
#             flattened = padded_mask.ravel()
#             ttl_signal = np.repeat(flattened, galvo.pixel_samples).astype(bool)
#             do_task.do_channels.add_do_chan(f"{galvo.device}/{do_chan}")
#             do_task.timing.cfg_samp_clk_timing(
#                 rate=galvo.rate,
#                 source=f'/{galvo.device}/ao/SampleClock',
#                 sample_mode=AcquisitionType.FINITE,
#                 samps_per_chan=galvo.total_samples
#             )
#             do_task.write(ttl_signal.tolist(), auto_start=False)

#         ao_task.write(composite_wave, auto_start=False)
#         ai_task.start()
#         do_task.start()
#         ao_task.start()

#         ao_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
#         do_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
#         ai_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)

#         acq_data = np.array(ai_task.read(number_of_samples_per_channel=galvo.total_samples))

#     n_ch = len(ai_channels)
#     results = []
#     if n_ch == 1:
#         acq_data = acq_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
#         data2d = np.mean(acq_data, axis=2)
#         x1 = galvo.extrasteps_left
#         x2 = galvo.extrasteps_left + galvo.numsteps_x
#         cropped = data2d[:, x1:x2]
#         results = [cropped]
#     else:
#         for i in range(n_ch):
#             chan_data = acq_data[i].reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
#             data2d = np.mean(chan_data, axis=2)
#             x1 = galvo.extrasteps_left
#             x2 = galvo.extrasteps_left + galvo.numsteps_x
#             cropped = data2d[:, x1:x2]
#             results.append(cropped)
#     return results


# def variable_scan_rpoc(ai_channels, galvo, mask, dwell_multiplier=2.0, modulate=False, mod_masks=None):
#     if isinstance(ai_channels, str):
#         ai_channels = [ai_channels]

#     if modulate and mod_masks:
#         gen_mask = mod_masks[0]
#     else:
#         gen_mask = mask

#     if isinstance(gen_mask, Image.Image):
#         gen_mask = np.array(gen_mask)
#     if not isinstance(gen_mask, np.ndarray):
#         raise TypeError("Mask must be a NumPy array or PIL Image.")
#     gen_mask = gen_mask > 128

#     x_wave, y_wave, pixel_map = galvo.gen_variable_waveform(gen_mask, dwell_multiplier)
    
#     if modulate and mod_masks:
#         ttl_signals = []
#         for m in mod_masks:
#             if isinstance(m, Image.Image):
#                 m_arr = np.array(m)
#             else:
#                 m_arr = m
#             m_arr = m_arr > 128  
#             padded_mask = []
#             for row_idx in range(galvo.numsteps_y):
#                 row_data = m_arr[row_idx, :]
#                 padded_row = np.concatenate((
#                     np.zeros(galvo.extrasteps_left, dtype=bool),
#                     row_data,
#                     np.zeros(galvo.extrasteps_right, dtype=bool)
#                 ))
#                 padded_mask.append(padded_row)
#             padded_mask = np.array(padded_mask, dtype=bool)
#             flattened = padded_mask.ravel()
#             ttl_signal = np.repeat(flattened, galvo.pixel_samples).astype(bool)
#             ttl_signals.append(ttl_signal)
#     else:
#         if isinstance(mask, Image.Image):
#             mask = np.array(mask)
#         mask = mask > 128
#         padded_mask = []
#         for row_idx in range(galvo.numsteps_y):
#             row_data = mask[row_idx, :]
#             padded_row = np.concatenate((
#                 np.zeros(galvo.extrasteps_left, dtype=bool),
#                 row_data,
#                 np.zeros(galvo.extrasteps_right, dtype=bool)
#             ))
#             padded_mask.append(padded_row)
#         padded_mask = np.array(padded_mask, dtype=bool)
#         flattened = padded_mask.ravel()
#         ttl_signal = np.repeat(flattened, galvo.pixel_samples).astype(bool)
    
#     with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
#         for chan in galvo.ao_chans:
#             ao_task.ao_channels.add_ao_voltage_chan(f"{galvo.device}/{chan}")
#         for ch in ai_channels:
#             ai_task.ai_channels.add_ai_voltage_chan(ch)

#         total_samps = len(x_wave)
#         ao_task.timing.cfg_samp_clk_timing(
#             rate=galvo.rate,
#             sample_mode=AcquisitionType.FINITE,
#             samps_per_chan=total_samps
#         )
#         ai_task.timing.cfg_samp_clk_timing(
#             rate=galvo.rate,
#             source=f"/{galvo.device}/ao/SampleClock",
#             sample_mode=AcquisitionType.FINITE,
#             samps_per_chan=total_samps
#         )

#         if modulate and mod_masks:
#             num_mod = len(ttl_signals)
#             for i in range(num_mod):
#                 do_task.do_channels.add_do_chan(f"{galvo.device}/port0/line{5+i}")
#             do_task.timing.cfg_samp_clk_timing(
#                 rate=galvo.rate,
#                 source=f"/{galvo.device}/ao/SampleClock",
#                 sample_mode=AcquisitionType.FINITE,
#                 samps_per_chan=total_samps
#             )
#             do_task.write(ttl_signals, auto_start=False)
#         else:
#             do_task.do_channels.add_do_chan(f"{galvo.device}/port0/line5")
#             do_task.timing.cfg_samp_clk_timing(
#                 rate=galvo.rate,
#                 source=f"/{galvo.device}/ao/SampleClock",
#                 sample_mode=AcquisitionType.FINITE,
#                 samps_per_chan=total_samps
#             )
#             do_task.write(ttl_signal.tolist(), auto_start=False)

#         composite_wave = np.vstack([x_wave, y_wave])
#         ao_task.write(composite_wave, auto_start=False)

#         ai_task.start()
#         do_task.start()
#         ao_task.start()
#         ao_task.wait_until_done(timeout=total_samps / galvo.rate + 5)
#         ai_task.wait_until_done(timeout=total_samps / galvo.rate + 5)

#         acq_data = np.array(ai_task.read(number_of_samples_per_channel=total_samps))

#     n_channels = len(ai_channels)
#     results = []
#     if n_channels == 1:
#         pixel_values_2d = partition_and_average(acq_data, gen_mask, pixel_map, galvo)
#         x1 = galvo.extrasteps_left
#         x2 = x1 + galvo.numsteps_x
#         cropped = pixel_values_2d[:, x1:x2]
#         results = [cropped]
#     else:
#         for ch_idx in range(n_channels):
#             ch_data = acq_data[ch_idx]
#             pixel_values_2d = partition_and_average(ch_data, gen_mask, pixel_map, galvo)
#             x1 = galvo.extrasteps_left
#             x2 = x1 + galvo.numsteps_x
#             cropped = pixel_values_2d[:, x1:x2]
#             results.append(cropped)
#     return results