import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from .galvo_funcs import Galvo

def raster_scan(ai_channels, galvo):
    """
    Performs a 2D raster acquisition from the provided analog input channels
    using the waveform(s) in galvo.waveform.
    
    Args:
        ai_channels (list[str]): List of analog input channels, e.g. ['Dev1/ai0','Dev1/ai1'].
        galvo (Galvo): A Galvo-like object that generates the X/Y (and optional TTL) waveforms.
    
    Returns:
        list of 2D NumPy arrays, one per AI channel, each containing the scanned data.
    """
    if isinstance(ai_channels, str):
        ai_channels = [ai_channels]
    
    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
        # Prepare the AO (analog output) channels from galvo config
        ao_channels = list(galvo.ao_chans)
        composite_wave = galvo.waveform.copy()

        # Add AO channels
        for chan in ao_channels:
            ao_task.ao_channels.add_ao_voltage_chan(f'{galvo.device}/{chan}')
        ao_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=composite_wave.shape[1]
        )

        # Add AI channels
        for ch in ai_channels:
            ai_task.ai_channels.add_ai_voltage_chan(ch)

        # Both tasks use the same clock rate, sample mode, sample count
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

        # Write the composite waveform (X, Y, optional TTL) to AO, then start AI
        ao_task.write(composite_wave, auto_start=False)
        ai_task.start()
        ao_task.start()

        # Wait for completion
        ao_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        ai_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)

        # Read the acquired data
        acq_data = np.array(
            ai_task.read(number_of_samples_per_channel=galvo.total_samples)
        )

    n_ch = len(ai_channels)
    results = []

    if n_ch == 1:
        # Single-channel case => shape = (total_samples,)
        acq_data = acq_data.reshape(
            galvo.total_y, galvo.total_x, galvo.pixel_samples
        )
        data2d = np.mean(acq_data, axis=2)
        cropped = data2d[
            galvo.numsteps_extra:-galvo.numsteps_extra,
            galvo.numsteps_extra:-galvo.numsteps_extra
        ]
        return [cropped]
    else:
        # Multiple AI channels => shape = (n_ch, total_samples)
        for i in range(n_ch):
            chan_data = acq_data[i].reshape(
                galvo.total_y, galvo.total_x, galvo.pixel_samples
            )
            data2d = np.mean(chan_data, axis=2)
            cropped = data2d[
                galvo.numsteps_extra:-galvo.numsteps_extra,
                galvo.numsteps_extra:-galvo.numsteps_extra
            ]
            results.append(cropped)
        return results
