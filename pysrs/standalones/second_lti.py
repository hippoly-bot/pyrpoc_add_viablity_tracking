import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import time
import pyvisa
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ao_channel = 'Dev1/ao0'
step_amplitude = 1.0
duration = 2
rate = 100000  
osc_resource = 'USB0::0x0699::0x03C7::C010691::INSTR'

num_samples = int(duration * rate)
t_wave = np.linspace(0, duration, num_samples, endpoint=False)
waveform = np.ones(num_samples) * step_amplitude

with nidaqmx.Task() as ao_task:
    ao_task.ao_channels.add_ao_voltage_chan(ao_channel)
    ao_task.timing.cfg_samp_clk_timing(rate=rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=num_samples)
    ao_task.write(waveform, auto_start=False)
    ao_task.start()
    ao_task.wait_until_done()
    
time.sleep(0.5)

rm = pyvisa.ResourceManager()
scope = rm.open_resource(osc_resource)
scope.write('*CLS')
scope.write('ACQuire:STATE RUN')

time.sleep(0.2)

scope.write('SELect:CH1')
raw_wave = scope.query('WAVFrm?')
x_incr = float(scope.query('WFMOutpre:XINcr?'))
x_zero = float(scope.query('WFMOutpre:XZEro?'))
num_pts = int(scope.query('WFMOutpre:NR_Pt?'))

print(raw_wave)
wave_data = np.array([float(val) for val in raw_wave.strip().split(';')])
time_axis = x_zero + x_incr * np.arange(len(wave_data))

def step_response(t, K, wn, zeta):
    wd = wn * np.sqrt(1 - zeta**2)
    return K * (1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta/np.sqrt(1-zeta**2)) * np.sin(wd * t)))

theta_inf = np.mean(wave_data[-int(0.05*len(wave_data)):])
K_guess = theta_inf / step_amplitude
t_peak = time_axis[np.argmax(wave_data)]
wn_guess = np.pi / (t_peak)
zeta_guess = 0.1
p0 = [K_guess, wn_guess, zeta_guess]

fit_mask = time_axis > 0
popt, _ = curve_fit(step_response, time_axis[fit_mask], wave_data[fit_mask], p0=p0, maxfev=10000)
K_est, wn_est, zeta_est = popt

print('Estimated parameters:')
print('K =', K_est)
print('wn =', wn_est)
print('zeta =', zeta_est)

plt.figure(figsize=(10, 6))
plt.plot(time_axis, wave_data, 'b', label='Measured')
plt.plot(time_axis, step_response(time_axis, *popt), 'r--', label='Fitted Model')
plt.xlabel('Time (s)')
plt.ylabel('Output (V or angle unit)')
plt.legend()
plt.show()
