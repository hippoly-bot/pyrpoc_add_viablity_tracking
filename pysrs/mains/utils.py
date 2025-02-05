# utils.py
import numpy as np
from PIL import Image
import tkinter as tk

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.show_tooltip)
        widget.bind('<Leave>', self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tip_window:
            return
        x, y, _, _ = self.widget.bbox('insert')
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.geometry(f'+{x}+{y}')
        label = tk.Label(
            tw, text=self.text, justify='left',
            background='#ffffe0', relief='solid', borderwidth=1, padx=10, pady=5,
            font=('Calibri', 14)
        )
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

@staticmethod
def generate_data(num_channels=1, config=None):
    nx = config.get('numsteps_x', 200) if config else 200
    ny = config.get('numsteps_y', 200) if config else 200
    data_list = []
    for ch in range(num_channels):
        arr = np.random.uniform(0, 0.1, size=(ny, nx))
        offset_x = np.random.randint(-nx // 8, nx // 8)
        offset_y = np.random.randint(-ny // 8, ny // 8)
        center_x, center_y = nx // 2 + offset_x, ny // 2 + offset_y
        offset = 0  # 40 * ch maybe
        radius = min(nx, ny) // 4
        eye_offset = radius // 2
        eye_radius = radius // 8
        mouth_radius = radius // 2
        mouth_thickness = 2
        for x in range(nx):
            for y in range(ny):
                if ((x - (center_x - eye_offset - offset))**2 + (y - (center_y + eye_offset))**2) < eye_radius**2:
                    arr[y, x] = 1.0
                if ((x - (center_x + eye_offset - offset))**2 + (y - (center_y + eye_offset))**2) < eye_radius**2:
                    arr[y, x] = 1.0
                dist = ((x - center_x + offset)**2 + (y - (center_y + eye_offset // 2))**2)**0.5
                if mouth_radius - mouth_thickness < dist < mouth_radius + mouth_thickness and y < center_y:
                    arr[y, x] = 1.0
        data_list.append(arr)
    return data_list

@staticmethod
def convert(data, type=np.uint8):
    data_flipped = np.flipud(data)
    arr_norm = (data_flipped - data_flipped.min()) / (data_flipped.max() - data_flipped.min() + 1e-9)
    arr_typed = (arr_norm * 255).astype(type)
    return Image.fromarray(arr_typed)
