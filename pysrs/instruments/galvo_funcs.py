import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

class Galvo:
    def __init__(self, config, rpoc_mask=None, ttl_channel=None, **kwargs):
        # Default parameters
        defaults = {
            "numsteps_x": 400,  
            "numsteps_y": 400,  
            "numsteps_extra": 100,  # Extra padding for stability
            "offset_x": -1.2,  
            "offset_y": 1.5, 
            "dwell": 10e-6,  # in seconds (adjust if needed)
            "amp_x": 0.5, 
            "amp_y": 0.5,  
            "rate": 10000,  # Sampling rate in Hz
            "device": 'Dev1',  # NI-DAQ device name
            "ao_chans": ['ao1', 'ao0']  # Analog output channels for galvos
        }
        if config:
            defaults.update(config)
        defaults.update(kwargs)
        
        # Set all parameters as attributes
        for key, val in defaults.items():
            setattr(self, key, val)
        
        # Save the optional RPOC mask and TTL channel (if provided)
        self.rpoc_mask = rpoc_mask
        self.ttl_channel = ttl_channel

        # Determine the number of samples per pixel.
        self.pixel_samples = max(1, int(self.dwell * self.rate))
        self.total_x = self.numsteps_x + 2 * self.numsteps_extra
        self.total_y = self.numsteps_y + 2 * self.numsteps_extra
        self.total_samples = self.total_x * self.total_y * self.pixel_samples
        
        # Generate the basic x and y waveform.
        self.waveform = self.gen_raster()

    def gen_raster(self):
        total_rowsamples = self.pixel_samples * self.total_x

        x_row = np.linspace(-self.amp_x, self.amp_x, self.total_x, endpoint=False)
        x_waveform = np.tile(np.repeat(x_row, self.pixel_samples), self.total_y)
        y_steps = np.linspace(self.amp_y, -self.amp_y, self.total_y)
        y_waveform = np.repeat(y_steps, total_rowsamples)
        composite = np.vstack([x_waveform, y_waveform])

        if self.rpoc_mask is not None and self.ttl_channel is not None:
            channels = list(self.ao_chans)
            ttl_wave = Galvo.generate_ttl_waveform(self.rpoc_mask, self.pixel_samples, self.total_x, self.total_y, high_voltage=5.0)
            if ttl_wave.size != y_waveform.shape[1]:
                raise ValueError("TTL waveform length does not match scan waveform length!")

            channels.append(self.ttl_channel)
            composite = np.vstack([composite, ttl_wave])

        if len(x_waveform) < self.total_samples:
            x_waveform = np.pad(x_waveform, (0, self.total_samples - len(x_waveform)), constant_values=x_waveform[-1])
        else:
            x_waveform = x_waveform[:self.total_samples]
        composite[0] = x_waveform
        return composite

    def do_raster(self): # self.waveform is composite, 3rd entry is the RPOC mask
        channels = list(self.ao_chans)
        composite = self.waveform.copy()
        
        with nidaqmx.Task() as task:
            for chan in channels:
                task.ao_channels.add_ao_voltage_chan(f"{self.device}/{chan}")
            task.timing.cfg_samp_clk_timing(
                rate=self.rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=composite.shape[1]
            )
            print(f"Raster scanning with channels: {channels}")
            task.write(composite, auto_start=True)
            task.wait_until_done()
            print("Raster complete")

    @staticmethod
    def generate_ttl_waveform(mask_image, pixel_samples, total_x, total_y, high_voltage=5.0):
        """
        Convert a mask (PIL Image in grayscale) into a TTL waveform.
        
        - Pixels with a value > 128 are considered "active" (logic 1).
        - Each pixel is repeated 'pixel_samples' times.
        - The binary values are scaled by 'high_voltage' (e.g., 5 V for active, 0 V for inactive).
        """
        mask_arr = np.array(mask_image)
        binary_mask = (mask_arr > 128).astype(np.uint8)
        
        # If the mask dimensions do not match the scan dimensions, resize it.
        if binary_mask.shape != (total_y, total_x):
            mask_pil = Image.fromarray(binary_mask * 255)
            mask_resized = mask_pil.resize((total_x, total_y), Image.NEAREST)
            binary_mask = (np.array(mask_resized) > 128).astype(np.uint8)
        
        # For each row, repeat each pixel value 'pixel_samples' times.
        ttl_rows = [np.repeat(binary_mask[row, :], pixel_samples) for row in range(total_y)]
        ttl_wave = np.concatenate(ttl_rows)
        ttl_wave = ttl_wave * high_voltage
        return ttl_wave

# --- For testing ---
if __name__ == '__main__':
    # Example configuration
    config = {
        "device": 'Dev1',
        "ao_chans": ['ao1', 'ao0'],
        "amp_x": 0.5,
        "amp_y": 0.5,
        "rate": 1e5,  # Hz
        "numsteps_x": 100,
        "numsteps_y": 100,
        "dwell": 50e-6,
    }
    # Create a dummy mask (for example purposes, a white rectangle on a black background)
    dummy_mask = Image.new('L', (config["numsteps_x"] + 2*100, config["numsteps_y"] + 2*100), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(dummy_mask)
    draw.rectangle([100, 100, 200, 200], fill=255)

    # Create a Galvo instance with RPOC (TTL) mode enabled.
    galvo = Galvo(config, rpoc_mask=dummy_mask, ttl_channel="ao2")
    galvo.waveform = galvo.gen_raster()

    # (Optional) Plot the original x and y waveforms.
    times = np.arange(galvo.waveform.shape[1]) / config['rate']
    plt.figure(figsize=(10, 6))
    plt.plot(times, galvo.waveform[0], label='x (fast axis)', color='black')
    plt.plot(times, galvo.waveform[1], label='y (slow axis)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Raster Scan Waveforms (without TTL channel)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Run the raster scan (this will output the TTL signal along with the galvo signals)
    galvo.do_raster()
