import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import matplotlib.pyplot as plt
import time
from pysrs.instruments.galvo_funcs import Galvo

@staticmethod
def lockin_scan(lockin_chan, galvo):
    if isinstance(lockin_chan, str):
        lockin_chan = [lockin_chan]  # wrap single in list

    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
        chans = list(galvo.ao_chans)
        composite = galvo.waveform.copy()

        for chan in chans:
            ao_task.ao_channels.add_ao_voltage_chan(f'{galvo.device}/{chan}')
        ao_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=composite.shape[1]
        )
        
        for ch in lockin_chan:
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

        ao_task.write(galvo.waveform, auto_start=False)
        ai_task.start()
        ao_task.start()

        ao_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        ai_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)

        lockin_data = np.array(
            ai_task.read(
                number_of_samples_per_channel=galvo.total_samples
            )
        )

    nChan = len(lockin_chan)
    out_list = []

    if nChan == 1:
        lockin_data = lockin_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
        data = np.mean(lockin_data, axis=2)
        cropped = data[galvo.numsteps_extra:-galvo.numsteps_extra,
                       galvo.numsteps_extra:-galvo.numsteps_extra]
        return [cropped]  
    else:
        for i in range(nChan):
            chan_data = lockin_data[i]  # shape=(total_samples,)
            chan_data = chan_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
            data2d = np.mean(chan_data, axis=2)
            cropped = data2d[galvo.numsteps_extra:-galvo.numsteps_extra,
                             galvo.numsteps_extra:-galvo.numsteps_extra]
            out_list.append(cropped)
        return out_list

@staticmethod
def lockin_scan_rpoc(lockin_chan, galvo):
    if isinstance(lockin_chan, str):
        lockin_chan = [lockin_chan]  # wrap single in list

    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task, nidaqmx.Task() as do_task:
        chans = list(galvo.ao_chans)
        composite = galvo.waveform.copy()

        for chan in chans:
            ao_task.ao_channels.add_ao_voltage_chan(f'{galvo.device}/{chan}')
        ao_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=composite.shape[1]
        )

        digital_waveform = np.array(galvo.waveform[2] > 0, dtype=int)
        do_task.do_channels.add_do_chan(f'{galvo.device}/port0/line0') # THIS IS THE KEY CHANGE!!
        do_task.timing.cfg_samp_clk_timing(
            rate=galvo.rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=galvo.total_samples
        )
        
        for ch in lockin_chan:
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

        ao_task.write(galvo.waveform, auto_start=False)
        do_task.write(digital_waveform, auto_start=False)

        ai_task.start()
        ao_task.start()
        do_task.start()

        ao_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        ai_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)
        do_task.wait_until_done(timeout=galvo.total_samples / galvo.rate + 5)

        lockin_data = np.array(
            ai_task.read(
                number_of_samples_per_channel=galvo.total_samples
            )
        )

    nChan = len(lockin_chan)
    out_list = []

    if nChan == 1:
        lockin_data = lockin_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
        data = np.mean(lockin_data, axis=2)
        cropped = data[galvo.numsteps_extra:-galvo.numsteps_extra,
                       galvo.numsteps_extra:-galvo.numsteps_extra]
        return [cropped]  
    else:
        for i in range(nChan):
            chan_data = lockin_data[i]  # shape=(total_samples,)
            chan_data = chan_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
            data2d = np.mean(chan_data, axis=2)
            cropped = data2d[galvo.numsteps_extra:-galvo.numsteps_extra,
                             galvo.numsteps_extra:-galvo.numsteps_extra]
            out_list.append(cropped)
        return out_list    

def plot_image(data: np.ndarray, galvo: Galvo, savedat=True) -> None:
    if savedat:
        np.savez('scanned_data.npz', data=data, **galvo.__dict__)

    plt.imshow(data, 
               extent=[-galvo.amp_x, galvo.amp_x, -galvo.amp_y, galvo.amp_y], 
               origin='lower', 
               aspect='auto', 
               cmap='gray')
    plt.colorbar(label="Lock-in amplitude (V)")
    plt.title("Raster Scanned Image with Lock-in Data")
    plt.xlabel('Galvo X Voltage')
    plt.ylabel('Galvo Y Voltage')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    config = {
        "device": 'Dev1',
        "ao_chans": ['ao1', 'ao0'],  
        "amp_x": 0.5,
        "amp_y": 0.5,
        "rate": 1e5,  # hz
        "numsteps_x": 400,
        "numsteps_y": 400,
        "dwell": 10,  # us
        "numsteps_extra": 100
    }
    galvo = Galvo(config)

    data = lockin_scan('Dev1/ai0', galvo)
    plot_image(data, galvo)