import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, TransferFunction, bode, step
from scipy.optimize import curve_fit



config = {
    "ao_channel": "Dev1/ao0",    
    "ai_channel": "Dev1/ai0",   
    "rate": 10000,                
    "duration": 2.0,            
    "test_type": "step",
    "step_amplitude": 0.5,
    "ramp_amplitude": 0.5,   
    "sine_amplitude": 0.3,   
    "sine_frequency": 10,          
    "save_plots": True,           
}


def generate_waveform(test_type, duration, rate):
    t = np.linspace(0, duration, int(duration * rate), endpoint=False)
    if test_type == "step":
        waveform = np.ones_like(t) * config["step_amplitude"]
        waveform[:len(t)//2] = 0
    elif test_type == "ramp":
        waveform = np.linspace(0, config["ramp_amplitude"], len(t))
    elif test_type == "sinusoidal":
        waveform = config["sine_amplitude"] * np.sin(2 * np.pi * config["sine_frequency"] * t)
    else:
        raise ValueError("Invalid test_type. Choose 'step', 'ramp', or 'sinusoidal'.")
    return t, waveform

def acquire_response(ao_channel, ai_channel, waveform, rate):
    total_samples = len(waveform)
    with nidaqmx.Task() as ao_task, nidaqmx.Task() as ai_task:
        ao_task.ao_channels.add_ao_voltage_chan(ao_channel)
        ai_task.ai_channels.add_ai_voltage_chan(ai_channel)

        ao_task.timing.cfg_samp_clk_timing(
            rate=rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples
        )
        ai_task.timing.cfg_samp_clk_timing(
            rate=rate, source=f"/{config['ao_channel'].split('/')[0]}/ao/SampleClock",
            sample_mode=AcquisitionType.FINITE, samps_per_chan=total_samples
        )

        ao_task.write(waveform, auto_start=False)
        ai_task.start()
        ao_task.start()
        ao_task.wait_until_done(timeout=10)
        ai_task.wait_until_done(timeout=10)
        response = ai_task.read(number_of_samples_per_channel=total_samples)
    return np.array(response)


def fit_second_order_step(t, response, step_amplitude):
    def model(t, K, wn, zeta):
        return K * (1 - np.exp(-zeta * wn * t) * (np.cos(wn * np.sqrt(1 - zeta**2) * t) + 
                                                  (zeta / np.sqrt(1 - zeta**2)) * np.sin(wn * np.sqrt(1 - zeta**2) * t)))

    response_normalized = response / step_amplitude
    popt, _ = curve_fit(model, t, response_normalized, bounds=(0, [10, 1000, 1]))
    return popt 


def plot_results(t, command, response, transfer_params):
    K, wn, zeta = transfer_params

    tf = TransferFunction([K * wn ** 2], [1, 2 * zeta * wn, wn ** 2])
    t_step, step_response = step(tf)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, command, 'k--', label="Command Signal")
    plt.plot(t, response, 'b', label="Measured Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Galvo Response")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t_step, step_response * config["step_amplitude"], 'r', label="Fitted Model")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Fitted Transfer Function: K={K:.3f}, wn={wn:.2f} rad/s, zeta={zeta:.3f}")
    plt.legend()
    plt.tight_layout()

    if config["save_plots"]:
        plt.savefig("galvo_calibration_results.png")
    plt.show()

def main():
    t, command_wave = generate_waveform(config["test_type"], config["duration"], config["rate"])
    response_wave = acquire_response(config["ao_channel"], config["ai_channel"], command_wave, config["rate"])

    transfer_params = fit_second_order_step(t, response_wave, config["step_amplitude"])
    plot_results(t, command_wave, response_wave, transfer_params)

    print("\nCalibration Complete:")
    print(f"Gain (K): {transfer_params[0]:.3f}")
    print(f"Natural Frequency (wn): {transfer_params[1]:.2f} rad/s")
    print(f"Damping Ratio (zeta): {transfer_params[2]:.3f}")

if __name__ == "__main__":
    main()
