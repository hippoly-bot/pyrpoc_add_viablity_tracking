import nidaqmx
from nidaqmx.constants import (AcquisitionType, CountDirection, Edge,
    READ_ALL_AVAILABLE, TaskMode, TriggerType)
from nidaqmx.stream_readers import CounterReader
import numpy as np 
import time
import matplotlib.pyplot as plt

'''
old functions from initial testing
now largely integrated into utils/insstruments/lockin.py
'''

def setup(anout=False, anin=False, dig=False, cin=False, cout=False):
    system = nidaqmx.system.System.local()

    devices = [device.name for device in system.devices]
    print(f'available devices: {devices}\n\n')
    for device in system.devices:
        print(f'device: {device.name}')

        if anin: 
            print(f'analog inputs:')
            for ai_channel in device.ai_physical_chans: 
                print(f'    {ai_channel.name}')

        if anout: 
            print(f'analog outputs:')
            for ao_channel in device.ao_physical_chans: 
                print(f'    {ao_channel.name}')

        if dig: 
            print('digital lines:')
            for di_channel in device.di_lines: 
                print(f'    {di_channel.name}')
            for do_channel in device.do_lines: 
                print(f'    {do_channel.name}')

        if cin: 
            print('counter inputs:')
            for ci_channel in device.ci_physical_chans: 
                print(f'    {ci_channel.name}')

        if cout: 
            print('counter outputs:')
            for co_channel in device.co_physical_chans: 
                print(f'    {co_channel.name}')
        print('\n\n')

    
    return None

def test(device_name):
    with nidaqmx.Task() as task:
        for ai_channel in nidaqmx.system.System.local().devices[device_name].ai_physical_chans:
            print(f'ai channel: {ai_channel.name}')
            
            task.ai_channels.add_ai_voltage_chan(ai_channel.name)

        task.start()
        data = task.read(number_of_samples_per_channel=1)
        print('signal data from channels:')
        for i, ai_channel in enumerate(task.ai_channels):
            print(f'  {ai_channel.physical_channel.name}: {data[i]}')

def test_connection(device_name):
    print(f'testing ai input channels for {device_name}')
    
    connected_chans = []
    
    with nidaqmx.Task() as task:
        for ai_chan in nidaqmx.system.System.local().devices[device_name].ai_physical_chans:
            task.ai_channels.add_ai_voltage_chan(ai_chan.name)

        task.start()
        data = task.read(number_of_samples_per_channel=1)
        task.stop()
        
        print('signal data from channels:')
        for i, ai_chan in enumerate(task.ai_channels):
            chan = ai_chan.physical_channel.name
            voltage = float(data[i][0])
            print(f'  {chan}: {voltage:.3f} V')
            
            if abs(voltage) > 0.01: 
                connected_chans.append(chan)
    
    print('\nconnected channels:')
    if connected_chans:
        for chan in connected_chans:
            print(f'  {chan}')
    else:
        print('  no connected channels detected')
    
def time_series(device_name, channel_name, duration, sampling_rate):
    num_samples = int(duration * sampling_rate)
    timestamps = np.linspace(0, duration, num_samples)
    data = np.zeros(num_samples)
    
    with nidaqmx.Task() as task:
        full_channel_name = f'{device_name}/{channel_name}'
        task.ai_channels.add_ai_voltage_chan(full_channel_name)
        
        task.timing.cfg_samp_clk_timing(
            rate=sampling_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=num_samples
        )
        
        print(f'taking {num_samples} samples from {full_channel_name} at {sampling_rate} Hz')
        
        task.start()
        time.sleep(duration + 0.1)  
        data = task.read(number_of_samples_per_channel=num_samples)
        task.stop()
    
    print('data acquisition complete')
    return timestamps, np.array(data)

def live_series(device_name, channel_name, duration, sampling_rate):
    num_samples = int(duration * sampling_rate)
    interval = 1 / sampling_rate  # Interval between each sample
    total_samples = 0
    
    timestamps = []
    data = []

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label=f'{device_name}/{channel_name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'Real-Time Data from {device_name}/{channel_name}')
    ax.grid(True)
    ax.legend()

    with nidaqmx.Task() as task:
        full_channel_name = f'{device_name}/{channel_name}'
        task.ai_channels.add_ai_voltage_chan(full_channel_name)
        
        task.timing.cfg_samp_clk_timing(
            rate=sampling_rate,
            sample_mode=AcquisitionType.CONTINUOUS
        )

        print(f'Starting real-time data acquisition from {full_channel_name}...')
        
        task.start()
        start_time = time.time()

        while total_samples < num_samples:
            elapsed_time = time.time() - start_time
            chunk_size = int(min(sampling_rate * elapsed_time, num_samples - total_samples))
            if chunk_size > 0:
                new_data = task.read(number_of_samples_per_channel=chunk_size)
                total_samples += chunk_size

                current_time = np.linspace(total_samples / sampling_rate, 
                                           (total_samples + chunk_size) / sampling_rate, 
                                           chunk_size, endpoint=False)
                timestamps.extend(current_time)
                data.extend(new_data)

                line.set_xdata(timestamps)
                line.set_ydata(data)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)  
        
        task.stop()
    plt.ioff()
    print('Data acquisition complete.')
    plt.show()

if __name__ == '__main__':
    # setup(anin=True, anout=True)
    test_connection('Dev1')

    device_name = 'Dev1'
    channel_name = 'ai6'
    duration = 50  # seconds
    sampling_rate = 200  # Hz

    timestamps, data = time_series(device_name, channel_name, duration, sampling_rate)
    
    plt.plot(timestamps, data) 
    plt.tight_layout()
    plt.show()