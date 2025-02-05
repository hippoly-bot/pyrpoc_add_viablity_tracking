import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import matplotlib.pyplot as plt
import time

class LockIn:
    def __init__(self, device, ai_chan, **kwargs):
        defaults = {
            "device": device,
            "ai_chan": ai_chan, 
            "sampling_rate": 100,
            "duration": 50
        }

        defaults.update(kwargs)
        for key, val in defaults.items():
            setattr(self, key, val)

        self.name = self.device + '/' + self.ai_chan

    def live_series(self):
        num_samples = int(self.duration * self.sampling_rate)
        counter = 0

        times = []
        data = []

        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], label=self.name)
        ax.set_xlabel('Time, s')
        ax.set_ylabel('Voltage, V')
        ax.set_title(f'Real Time Data from {self.name}')
        ax.grid(True)
        ax.legend()

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.name)
            task.timing.cfg_samp_clk_timing(
                rate=self.sampling_rate,
                sample_mode=AcquisitionType.CONTINUOUS
            )

            print(f'acquiring real time data from {self.name}')
            task.start()
            tic = time.time()

            while counter < num_samples:
                toc = time.time() - tic
                chunk = int(min(self.sampling_rate * toc, num_samples - counter))
                if chunk > 0:
                    dt = task.read(number_of_samples_per_channel=chunk)
                    counter += chunk

                    current = np.linspace(counter / self.sampling_rate, 
                                          (counter + chunk) / self.sampling_rate, 
                                          chunk, endpoint = False)
                    times.extend(current)
                    data.extend(dt)

                    line.set_xdata(times)
                    line.set_ydata(data)
                    ax.relim()
                    ax.autoscale_view()
                    plt.pause(0.01)
                
            task.stop()

        plt.ioff()
        toc = time.time() - tic
        print(f'acquisition done in {toc-tic} s')
        plt.show()

    def collect(self, avg=True):
        num_samples = int(self.duration * self.sampling_rate)
        times = np.linspace(0, self.duration, num_samples)
        data = []

        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(self.name)
            task.timing.cfg_samp_clk_timing(
                rate=self.sampling_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=num_samples
            )

            print(f"taking {num_samples} samples from {self.name} at {self.sampling_rate} hz")
            task.start()
            time.sleep(self.duration + 0.1) # buffer to make sure the task is actually done
            data = task.read(number_of_samples_per_channel=num_samples, timeout=1)
            task.stop()

        print("Data collection complete")
        return times, np.array(data)


if __name__ == '__main__':
    lockin = LockIn('Dev1', 'ai1', sampling_rate=200, duration=1)
    times, data = lockin.collect()
    other = LockIn('Dev1', 'ai0', sampling_rate=200, duration=1)
    times2, data2 = other.collect()

    plt.plot(times, data, label='ai1', color='black')
    plt.plot(times2, data2, label='ai0 (nothing useful)', color='red')
    plt.xlabel('Time, s')
    plt.ylabel('Voltage, V')
    plt.title('long-time comparison of lockin BG with unoccupied channel')
    plt.grid(True)
    plt.legend()
    plt.show()