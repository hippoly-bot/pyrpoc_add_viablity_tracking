import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from PIL import Image

from .rpoc2 import build_rpoc_wave

class Galvo:
    def __init__(self, config, rpoc_mask=None, rpoc_do_chan=None, rpoc_mode=None, dwell_multiplier=2.0, **kwargs):
        defaults = {
            "numsteps_x": 400,
            "numsteps_y": 400,
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
        self.rpoc_mode = rpoc_mode  # "standard" or "variable" so far
        self.dwell_multiplier = dwell_multiplier

        self.pixel_samples = max(1, int(self.dwell * self.rate))
        self.total_x = self.numsteps_x + self.extrasteps_left + self.extrasteps_right
        self.total_y = self.numsteps_y
        self.total_samples = self.total_x * self.total_y * self.pixel_samples

        if self.rpoc_mode == "variable":
            self.waveform = None  # or maybe generate it here...
        else:
            self.waveform = self.gen_raster()

    def gen_raster(self):
        total_rowsamples = self.pixel_samples * self.total_x

        # dont double count the boundaries, so endpoint=False
        x_row = np.linspace(self.offset_x - self.amp_x,
                            self.offset_x + self.amp_x,
                            self.total_x, endpoint=False)
        x_waveform = np.tile(np.repeat(x_row, self.pixel_samples), self.total_y)

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

        # had some runs where x was the wrong shape, could not figure out why so just pad it 
        if len(x_waveform) < self.total_samples:
            x_waveform = np.pad(x_waveform,
                                (0, self.total_samples - len(x_waveform)),
                                constant_values=x_waveform[-1])
        else:
            x_waveform = x_waveform[:self.total_samples]
        composite[0] = x_waveform

        return composite

    def gen_variable_waveform(self, mask, dwell_multiplier):
        dwell = self.dwell
        rate = self.rate
        num_y = self.numsteps_y
        num_x = self.numsteps_x + self.extrasteps_left + self.extrasteps_right

        # We'll define dwell_on = dwell * dwell_multiplier
        # If mask pixel = True, we use dwell_on. Otherwise dwell_off = dwell.
        dwell_on = dwell * dwell_multiplier
        dwell_off = dwell

        # For the "flying" x steps, we want to linearly span offset_x - amp_x to +amp_x, etc.
        # But we do it pixel by pixel:
        x_min = self.offset_x - self.amp_x
        x_max = self.offset_x + self.amp_x
        # We'll form an array of x positions: e.g. np.linspace(..., num_x, endpoint=False)
        # so that pixel i is at x_positions[i]
        x_positions = np.linspace(x_min, x_max, num_x, endpoint=False)
        # Similarly for y_positions (top row to bottom row):
        y_positions = np.linspace(self.offset_y + self.amp_y,
                                self.offset_y - self.amp_y,
                                num_y)

        # We build lists of samples row by row
        x_wave_list = []
        y_wave_list = []
        pixel_map = np.zeros((num_y, num_x), dtype=int)

        for row_idx in range(num_y):
            row_y = y_positions[row_idx]
            for col_idx in range(num_x):
                # Decide dwell for this pixel
                if mask[row_idx, col_idx]:
                    this_dwell = dwell_on
                else:
                    this_dwell = dwell_off

                # Number of samples for this pixel
                pixel_samps = max(1, int(this_dwell * rate))

                # The galvo X remains constant for these pixel_samps
                x_val = x_positions[col_idx]
                x_wave_list.append(np.full(pixel_samps, x_val))

                # The galvo Y remains constant for these pixel_samps
                y_wave_list.append(np.full(pixel_samps, row_y))

                # Store in pixel_map
                pixel_map[row_idx, col_idx] = pixel_samps

        # Concatenate
        x_wave = np.concatenate(x_wave_list)
        y_wave = np.concatenate(y_wave_list)
        return x_wave, y_wave, pixel_map
