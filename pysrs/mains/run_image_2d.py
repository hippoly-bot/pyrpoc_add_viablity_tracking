import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from pysrs.mains.galvo_funcs import Galvo
import matplotlib.pyplot as plt

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

def digital_test(ai_channels, galvo, ttl_do_chan_user1="port0/line5", ttl_do_chan_user2="port0/line7"):
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

        ttl_signal_user1 = np.zeros(galvo.total_samples, dtype=bool)
        ttl_signal_user1[:galvo.total_samples // 2] = True

        ttl_signal_user2 = np.zeros(galvo.total_samples, dtype=bool)
        ttl_signal_user2[:galvo.total_samples // 2] = True

        ttl_signals = np.array([ttl_signal_user1, ttl_signal_user2])

        plt.figure(figsize=(10, 4))
        plt.plot(ttl_signal_user1, label="TTL Signal USER1", linestyle="--")
        plt.plot(ttl_signal_user2, label="TTL Signal USER2", linestyle=":")
        plt.xlabel("Sample Index")
        plt.ylabel("TTL Signal (0 or 1)")
        plt.title("TTL Signals for USER1 and USER2")
        plt.legend()
        plt.show()

        do_task.do_channels.add_do_chan(f"{galvo.device}/{ttl_do_chan_user1}")
        do_task.do_channels.add_do_chan(f"{galvo.device}/{ttl_do_chan_user2}")

        do_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            source=f'/{galvo.device}/ao/SampleClock',
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=galvo.total_samples
        )

        ao_task.write(composite_wave, auto_start=False)
        do_task.write(ttl_signals.tolist(), auto_start=False)

        ai_task.start()
        do_task.start()
        ao_task.start()

        print("Tasks Started Successfully.")

        do_task.wait_until_done(timeout=(2 * galvo.total_samples / galvo.rate + 10))
        ao_task.wait_until_done(timeout=(2 * galvo.total_samples / galvo.rate + 10))
        ai_task.wait_until_done(timeout=(2 * galvo.total_samples / galvo.rate + 10))

        acq_data = np.array(ai_task.read(number_of_samples_per_channel=galvo.total_samples))

    n_ch = len(ai_channels)
    results = []

    if n_ch == 1:
        acq_data = acq_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
        data2d = np.mean(acq_data, axis=2)
        x1 = galvo.extrasteps_left
        x2 = galvo.extrasteps_left + galvo.numsteps_x
        cropped = data2d[:, x1:x2]
        results.append(cropped)
    else:
        for i in range(n_ch):
            chan_data = acq_data[i].reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
            data2d = np.mean(chan_data, axis=2)
            x1 = galvo.extrasteps_left
            x2 = galvo.extrasteps_left + galvo.numsteps_x
            cropped = data2d[:, x1:x2]
            results.append(cropped)

    if len(results) > 0:
        plt.figure(figsize=(8, 6))
        plt.imshow(results[0], cmap='gray', aspect='auto')
        plt.colorbar()
        plt.title("Acquired Scan Data")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()

    return results

if __name__ == "__main__":
    config = {
        "numsteps_x": 200,
        "numsteps_y": 200,
        "extrasteps_left": 50,
        "extrasteps_right": 50,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "dwell": 10e-6,
        "amp_x": 0.5,
        "amp_y": 0.5,
        "rate": 100000,
        "device": 'Dev1',
        "ao_chans": ['ao1', 'ao0']
    }

    galvo = Galvo(config)
    acquired_data = digital_test(['Dev1/ai0'], galvo)

    print("Scan complete. Data shape:", acquired_data[0].shape)
