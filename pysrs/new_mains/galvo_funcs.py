import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from PIL import Image

from .rpoc2 import build_rpoc_wave

class Galvo:
    """
    Builds waveforms for scanning in X (fast) and Y (slow),
    plus optionally an RPOC mask (TTL) row if rpoc_mask & rpoc_do_chan are given.
    """

    def __init__(self, config, rpoc_mask=None, rpoc_do_chan=None, **kwargs):
        defaults = {
            "numsteps_x": 400,
            "numsteps_y": 400,

            # Replacing old 'numsteps_extra' with extrasteps_left/right
            "extrasteps_left": 50,
            "extrasteps_right": 50,

            "offset_x": 0.0,
            "offset_y": 0.0,

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
        self.rpoc_do_chan = rpoc_do_chan

        # Number of samples per pixel
        self.pixel_samples = max(1, int(self.dwell * self.rate))

        # X dimension includes extra steps
        self.total_x = self.numsteps_x + self.extrasteps_left + self.extrasteps_right
        # Y dimension has no extra padding now
        self.total_y = self.numsteps_y

        self.total_samples = self.total_x * self.total_y * self.pixel_samples

        # Build wave
        self.waveform = self.gen_raster()

    def gen_raster(self):
        total_rowsamples = self.pixel_samples * self.total_x

        # X from -amp_x + offset_x up to +amp_x + offset_x
        # We use endpoint=False to avoid double counts at the boundary
        x_row = np.linspace(self.offset_x - self.amp_x,
                            self.offset_x + self.amp_x,
                            self.total_x, endpoint=False)
        x_waveform = np.tile(np.repeat(x_row, self.pixel_samples), self.total_y)

        # Y from offset_y + amp_y down to offset_y - amp_y
        y_steps = np.linspace(self.offset_y + self.amp_y,
                              self.offset_y - self.amp_y,
                              self.total_y)
        y_waveform = np.repeat(y_steps, total_rowsamples)

        composite = np.vstack([x_waveform, y_waveform])

        if self.rpoc_mask is not None and self.rpoc_do_chan is not None:
            rpoc_wave = build_rpoc_wave(
                self.rpoc_mask,
                self.pixel_samples,
                self.total_x,
                self.total_y,
                high_voltage=5.0
            )
            if rpoc_wave.size != y_waveform.size:
                raise ValueError("RPOC wave length does not match total scan length!")
            composite = np.vstack([composite, rpoc_wave])

        # Ensure final shape matches total_samples (pad or truncate X)
        if len(x_waveform) < self.total_samples:
            x_waveform = np.pad(x_waveform,
                                (0, self.total_samples - len(x_waveform)),
                                constant_values=x_waveform[-1])
        else:
            x_waveform = x_waveform[:self.total_samples]
        composite[0] = x_waveform

        return composite
