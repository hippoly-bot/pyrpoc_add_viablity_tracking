import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from pysrs.mains.galvo_funcs import Galvo
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw, ImageOps

def raster_scan(ai_channels, galvo):
    if isinstance(ai_channels, str):
        ai_channels = [ai_channels]

    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
        ao_channels = list(galvo.ao_chans)
        composite_wave = galvo.waveform.copy()

        for chan in ao_channels:
            ao_task.ao_channels.add_ao_voltage_chan(f'{galvo.device}/{chan}')
        ao_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=composite_wave.shape[1]
        )

        for ch in ai_channels:
            ai_task.ai_channels.add_ai_voltage_chan(ch)

        ao_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=galvo.total_samples
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            source=f'/{galvo.device}/ao/SampleClock',
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=galvo.total_samples
        )

        ao_task.write(composite_wave, auto_start=False)
        ai_task.start()
        ao_task.start()

        ao_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        ai_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)

        acq_data = np.array(ai_task.read(number_of_samples_per_channel=galvo.total_samples))

    n_ch = len(ai_channels)
    results = []

    if n_ch == 1:
        # shape is (total_y, total_x, pixel_samples), dont forget that u dummy
        acq_data = acq_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
        data2d = np.mean(acq_data, axis=2)

        # crop out extrasteps_left and extrasteps_right
        x1 = galvo.extrasteps_left
        x2 = galvo.extrasteps_left + galvo.numsteps_x
        cropped = data2d[:, x1:x2]
        return [cropped]
    else:
        for i in range(n_ch):
            chan_data = acq_data[i].reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
            data2d = np.mean(chan_data, axis=2)

            x1 = galvo.extrasteps_left
            x2 = galvo.extrasteps_left + galvo.numsteps_x
            cropped = data2d[:, x1:x2]
            results.append(cropped)
        return results

def raster_scan_rpoc(ai_channels, galvo, mask, do_chan="port0/line5"):
    if isinstance(ai_channels, str):
        ai_channels = [ai_channels]

    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
        ao_channels = list(galvo.ao_chans)
        composite_wave = galvo.waveform.copy()  
        for chan in ao_channels:
            ao_task.ao_channels.add_ao_voltage_chan(f'{galvo.device}/{chan}')
        ao_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=composite_wave.shape[1]
        )


        for ch in ai_channels:
            ai_task.ai_channels.add_ai_voltage_chan(ch)
        ai_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            source=f'/{galvo.device}/ao/SampleClock',
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=galvo.total_samples
        )

        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if not isinstance(mask, np.ndarray):
            raise TypeError('Mask must be a numpy array.')
        padded_mask = []

        for row_idx in range(galvo.numsteps_y):
            row_data = mask[row_idx, :] 
            padded_row = np.concatenate((
                np.zeros(galvo.extrasteps_left, dtype=bool),
                row_data,
                np.zeros(galvo.extrasteps_right, dtype=bool)
            ))
            padded_mask.append(padded_row)
        padded_mask = np.array(padded_mask, dtype=bool)  

        flattened = padded_mask.ravel() 
        ttl_signal = np.repeat(flattened, galvo.pixel_samples).astype(bool)


        do_task.do_channels.add_do_chan(f"{galvo.device}/{do_chan}")
        do_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            source=f'/{galvo.device}/ao/SampleClock',
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=galvo.total_samples
        )

        ao_task.write(composite_wave, auto_start=False)
        do_task.write(ttl_signal.tolist(), auto_start=False)

        ai_task.start()
        do_task.start()
        ao_task.start()

        ao_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        do_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        ai_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)

        acq_data = np.array(ai_task.read(number_of_samples_per_channel=galvo.total_samples))

    n_ch = len(ai_channels)
    results = []

    if n_ch == 1:
        acq_data = acq_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
        data2d = np.mean(acq_data, axis=2)
        x1 = galvo.extrasteps_left
        x2 = galvo.extrasteps_left + galvo.numsteps_x
        cropped = data2d[:, x1:x2]
        results = [cropped]

    else:
        for i in range(n_ch):
            chan_data = acq_data[i].reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
            data2d = np.mean(chan_data, axis=2)
            x1 = galvo.extrasteps_left
            x2 = galvo.extrasteps_left + galvo.numsteps_x
            cropped = data2d[:, x1:x2]
            results.append(cropped)

    return results

@staticmethod
def build_rpoc_wave(mask_image, pixel_samples, total_x, total_y, high_voltage=5.0):
    mask_arr = np.array(mask_image)
    binary_mask = (mask_arr > 128).astype(np.uint8)
    print(f'mask image shape {binary_mask.shape}')

    if binary_mask.shape != (total_y, total_x):
        mask_pil = Image.fromarray(binary_mask * 255)
        mask_resized = mask_pil.resize((total_x, total_y), Image.NEAREST)
        binary_mask = (np.array(mask_resized) > 128).astype(np.uint8)

    ttl_rows = [
        np.repeat(binary_mask[row, :], pixel_samples)
        for row in range(total_y)
    ]
    ttl_wave = np.concatenate(ttl_rows)
    ttl_wave = ttl_wave * high_voltage
    ttl_wave = ttl_wave.astype(bool)
    return ttl_wave

if __name__ == "__main__":
    config = {
        "numsteps_x": 200,
        "numsteps_y": 200,
        "extrasteps_left": 0,
        "extrasteps_right": 0,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "dwell": 5e-5,
        "amp_x": 0.5,
        "amp_y": 0.5,
        "rate": 100000,
        "device": 'Dev1',
        "ao_chans": ['ao1', 'ao0']
    }

    galvo = Galvo(config)
    acquired_data = raster_scan_rpoc(['Dev1/ai0'], galvo)

    print("Scan complete. Data shape:", acquired_data[0].shape)

    # system = nidaqmx.system.System.local()
    # do_channels = []

    # for dev in system.devices:
    #     if dev.name == 'Dev1':
    #         do_channels = dev.do_lines
    #         break

    # if do_channels:
    #     for ch in do_channels:
    #         print(ch)
    # else:
    #     print('epic fail')
    
