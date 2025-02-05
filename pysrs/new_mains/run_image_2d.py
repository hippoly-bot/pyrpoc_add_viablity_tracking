import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from pysrs.new_mains.galvo_funcs import Galvo

def digital_test(ai_channels, galvo, ttl_do_chan="port0/line0"):
    """
    Runs a raster scan with an additional TTL signal.
    The TTL signal will be high for the first half of the scan and low for the second half.
    """
    if isinstance(ai_channels, str):
        ai_channels = [ai_channels]

    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
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

        ai_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            source=f'/{galvo.device}/ao/SampleClock',  # Sync to AO
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=galvo.total_samples
        )

        # TTL Signal: High for first half, Low for second half
        ttl_signal = np.zeros(galvo.total_samples, dtype=np.uint8)
        ttl_signal[:galvo.total_samples // 2] = 1  # First half high, second half low

        # Add Digital Output Channel
        do_task.do_channels.add_do_chan(f"{galvo.device}/{ttl_do_chan}")

        # Configure DO to use same clock as AO (ensures synchronization)
        do_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            source=f'/{galvo.device}/ao/SampleClock',
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=galvo.total_samples
        )

        # Write the AO waveform (Galvo movement)
        ao_task.write(composite_wave, auto_start=False)

        # Write the TTL waveform (0s and 1s for the digital line)
        do_task.write(ttl_signal, auto_start=False)

        # Start tasks in sequence
        ai_task.start()
        ao_task.start()
        do_task.start()

        ao_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        ai_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        do_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)

        # Read acquired AI data
        acq_data = np.array(ai_task.read(number_of_samples_per_channel=galvo.total_samples))

    # Reshape and process acquired data
    n_ch = len(ai_channels)
    results = []

    if n_ch == 1:
        acq_data = acq_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
        data2d = np.mean(acq_data, axis=2)

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

if __name__ == "__main__":
    config = {
        "numsteps_x": 400,
        "numsteps_y": 400,
        "extrasteps_left": 50,
        "extrasteps_right": 50,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "dwell": 10e-6,
        "amp_x": 0.5,
        "amp_y": 0.5,
        "rate": 10000,
        "device": 'Dev1',
        "ao_chans": ['ao1', 'ao0']
    }

    galvo = Galvo(config)

    acquired_data = digital_test(['Dev1/ai0'], galvo, ttl_do_chan="port0/line0")

    print("Scan complete. Data shape:", acquired_data[0].shape)
