import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from .galvo_funcs import Galvo

def lockin_scan(lockin_chan, galvo):
    if isinstance(lockin_chan, str):
        lockin_chan = [lockin_chan]
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
        lockin_data = np.array(ai_task.read(number_of_samples_per_channel=galvo.total_samples))
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
            chan_data = lockin_data[i]
            chan_data = chan_data.reshape(galvo.total_y, galvo.total_x, galvo.pixel_samples)
            data2d = np.mean(chan_data, axis=2)
            cropped = data2d[galvo.numsteps_extra:-galvo.numsteps_extra,
                             galvo.numsteps_extra:-galvo.numsteps_extra]
            out_list.append(cropped)
        return out_list
