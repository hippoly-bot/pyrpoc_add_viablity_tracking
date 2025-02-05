import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from PIL import Image

class Galvo:
    def __init__(self, config, rpoc_mask=None, ttl_channel=None, **kwargs):
        defaults = {
            "numsteps_x": 400,
            "numsteps_y": 400,
            "numsteps_extra": 100,
            "offset_x": -1.2,
            "offset_y": 1.5,
            "dwell": 10e-6,
            "amp_x": 0.5,
            "amp_y": 0.5,
            "rate": 10000,
            "device": 'Dev1',
            "ao_chans": ['ao1', 'ao0']
        }
        if config:
            defaults.update(config)
        defaults.update(kwargs)
        for key, val in defaults.items():
            setattr(self, key, val)
        self.rpoc_mask = rpoc_mask
        self.ttl_channel = ttl_channel
        self.pixel_samples = max(1, int(self.dwell * self.rate))
        self.total_x = self.numsteps_x + 2 * self.numsteps_extra
        self.total_y = self.numsteps_y + 2 * self.numsteps_extra
        self.total_samples = self.total_x * self.total_y * self.pixel_samples
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
            ttl_wave = Galvo.generate_ttl_waveform(
                self.rpoc_mask, self.pixel_samples,
                self.total_x, self.total_y, high_voltage=5.0
            )
            if ttl_wave.size != y_waveform.size:
                raise ValueError("TTL waveform length does not match scan waveform length!")
            channels.append(self.ttl_channel)
            composite = np.vstack([composite, ttl_wave])
        if len(x_waveform) < self.total_samples:
            x_waveform = np.pad(x_waveform, (0, self.total_samples - len(x_waveform)),
                                constant_values=x_waveform[-1])
        else:
            x_waveform = x_waveform[:self.total_samples]
        composite[0] = x_waveform
        return composite

    @staticmethod
    def generate_ttl_waveform(mask_image, pixel_samples, total_x, total_y, high_voltage=5.0):
        mask_arr = np.array(mask_image)
        binary_mask = (mask_arr > 128).astype(np.uint8)
        if binary_mask.shape != (total_y, total_x):
            mask_pil = Image.fromarray(binary_mask * 255)
            mask_resized = mask_pil.resize((total_x, total_y), Image.NEAREST)
            binary_mask = (np.array(mask_resized) > 128).astype(np.uint8)
        ttl_rows = [np.repeat(binary_mask[row, :], pixel_samples) for row in range(total_y)]
        ttl_wave = np.concatenate(ttl_rows)
        ttl_wave = ttl_wave * high_voltage
        return ttl_wave
