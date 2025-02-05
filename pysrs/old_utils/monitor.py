import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import time
import matplotlib.pyplot as plt

def monitor(device, channels, duration, rate, interval):
    num_samples = int(duration * rate)
    chunk_size = int(interval * rate)
    total_chunks = int(duration / interval)

    timestamps = {ch: [] for ch in channels}
    values = {ch: [] for ch in channels}

    plt.ion()
    fig, ax = plt.subplots()
    plots = {}
    for ch in channels:
        plots[ch], = ax.plot([], [], label=f"{device}/{ch}")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Real-Time Monitoring")
    ax.legend()
    ax.grid(True)

    with nidaqmx.Task() as task:
        for channel in channels:
            task.ai_channels.add_ai_voltage_chan(f"{device}/{channel}")

        task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.CONTINUOUS)

        task.start()
        start_time = time.time()

        for chunk_idx in range(total_chunks):
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            data = task.read(number_of_samples_per_channel=chunk_size)
            if len(channels) == 1:
                data = [data]

            time_chunk = np.linspace(
                chunk_idx * interval,
                (chunk_idx + 1) * interval,
                chunk_size,
                endpoint=False
            )

            for idx, channel in enumerate(channels):
                timestamps[channel].extend(time_chunk)
                values[channel].extend(data[idx])
                plots[channel].set_xdata(timestamps[channel])
                plots[channel].set_ydata(values[channel])
            
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        task.stop()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    monitor(device="Dev1", channels=["ai6"], duration=200, rate=100, interval=0.5)