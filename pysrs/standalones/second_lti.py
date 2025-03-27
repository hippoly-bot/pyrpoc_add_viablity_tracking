import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import time
import pyvisa
import re
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ao_channel = 'Dev1/ao1'
step_amplitude = 0.2  
duration = 0.01      
rate = 1000000            
osc_resource = 'USB0::0x0699::0x03C7::C010691::INSTR'
vertical_scale = 100E-3  
vertical_offset = 0.0

num_samples = int(duration * rate)
t_wave = np.linspace(0, duration, num_samples, endpoint=False)
waveform = np.ones(num_samples) * step_amplitude

plt.plot(waveform)
plt.show()

rm = pyvisa.ResourceManager()
scope = rm.open_resource(osc_resource)
scope.write('*CLS')
for ch in range(1, 5):
    scope.write(f"CH{ch}:SCALE {vertical_scale}")
    scope.write(f"CH{ch}:OFFSET {vertical_offset}")
    print(scope.query(f"CH{ch}:OFFSET?"))
scope.write('HORIZONTAL:SCALE 2e-6')
scope.write('ACQuire:STOPAfter SEQ')
scope.write('TRIGger:A:EDGE:SOURCE CH1')
scope.write('TRIGger:A:EDGE:SLOPe RIS')
scope.write('ACQuire:STATE ON')

with nidaqmx.Task() as ao_task:
    ao_task.ao_channels.add_ao_voltage_chan(ao_channel)
    ao_task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=num_samples)
    ao_task.write(waveform, auto_start=False)
    ao_task.start()
    ao_task.wait_until_done()

time.sleep(0.1)

def parse_waveform(raw_str):
    num_strings = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", raw_str)
    return np.array([float(val) for val in num_strings])

waveforms = {}
time_axes = {}
for ch in range(1, 5):
    scope.write(f"DATa:SOUrce CH{ch}")
    scope.write("DATa:ENCdg ASCII")
    scope.write("DATa:WIDth 1")
    scope.write("DATa:STARt 1")
    scope.write("DATa:STOP 10000")
    raw_wave = scope.query("CURVe?")
    x_incr = float(scope.query("WFMOutpre:XINcr?"))
    x_zero = float(scope.query("WFMOutpre:XZEro?"))
    num_pts = int(scope.query("WFMOutpre:NR_Pt?"))
    wave_data = parse_waveform(raw_wave)
    if len(wave_data) > num_pts:
        wave_data = wave_data[:num_pts]
    time_axis = x_zero + x_incr * np.arange(len(wave_data))
    waveforms[f"CH{ch}"] = wave_data
    time_axes[f"CH{ch}"] = time_axis

plt.figure(figsize=(12, 8))
for ch in range(1, 5):
    plt.subplot(2, 2, ch)
    plt.plot(time_axes[f"CH{ch}"], waveforms[f"CH{ch}"], label=f'Channel {ch}')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage')
    plt.legend()
plt.tight_layout()
plt.show()

def step_response(t, K, wn, zeta):
    wd = wn * np.sqrt(1 - zeta**2)
    return K * (1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta/np.sqrt(1 - zeta**2)) * np.sin(wd * t)))

ch1_data = waveforms["CH1"]
ch1_time = time_axes["CH1"]
theta_inf = np.mean(ch1_data[-int(0.05 * len(ch1_data)):])
K_guess = theta_inf / step_amplitude
t_peak = ch1_time[np.argmax(ch1_data)]
wn_guess = np.pi / (t_peak) if t_peak > 0 else 10.0
zeta_guess = 0.1
p0 = [K_guess, wn_guess, zeta_guess]

fit_mask = ch1_time > 0
try:
    popt, _ = curve_fit(step_response, ch1_time[fit_mask], ch1_data[fit_mask], p0=p0, maxfev=10000)
    K_est, wn_est, zeta_est = popt
    print('Estimated parameters from Channel 1:')
    print('K =', K_est)
    print('wn =', wn_est)
    print('zeta =', zeta_est)
    plt.figure(figsize=(10, 6))
    plt.plot(ch1_time, ch1_data, 'b', label='Measured (CH1)')
    plt.plot(ch1_time, step_response(ch1_time, *popt), 'r--', label='Fitted Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Output (V or angle unit)')
    plt.legend()
    plt.show()
except Exception as e:
    print("Curve fitting failed with error:", e)
