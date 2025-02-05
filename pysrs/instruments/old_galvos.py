import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import time
import matplotlib.pyplot as plt

def setup(device, ao_chans):
    with nidaqmx.Task() as task:
        for chan in ao_chans:
            task.ao_channels.add_ao_voltage_chan(f"{device}/{chan}")
        print(f"setup complete on chans: {ao_chans}")

def gen_wave(waveform, amplitude, frequency, duration, rate):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    if waveform == "sine":
        return t, amplitude * np.sin(2 * np.pi * frequency * t)
    elif waveform == "triangle":
        return t, amplitude * (2 * np.abs(2 * (t * frequency % 1) - 1) - 1)
    elif waveform == "square":
        return t, amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    else:
        raise ValueError("not a valid waveform, use string input of 'sine', 'triangle', or 'square'.")

def control(device, ao_chans, waveform_type, amplitude, frequency, duration, rate):
    t, waveform = gen_wave(waveform_type, amplitude, frequency, duration, rate)
    waveform = np.tile(waveform, (len(ao_chans), 1))

    with nidaqmx.Task() as task:
        for chan in ao_chans:
            task.ao_channels.add_ao_voltage_chan(f"{device}/{chan}")

        task.timing.cfg_samp_clk_timing(
            rate=rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=waveform.shape[1]
        )

        print(f"scanning galvos via chans {ao_chans}")
        task.write(waveform, auto_start=True)
        task.wait_until_done()
        print("scanning done")

if __name__ == "__main__":
    device_name = "Dev1"
    ao_chans = ["ao0", "ao1"]
    waveform = "triangle"
    amp = 0.5 # v, not pp
    freq = 100 # hz
    duration = 1 # s
    sampling_rate = 1000 # hz

    setup(device_name, ao_chans)
    control(
        device=device_name,
        ao_chans=ao_chans,
        waveform_type=waveform,
        amplitude=amp,
        frequency=freq,
        duration=duration,
        rate=sampling_rate
    )