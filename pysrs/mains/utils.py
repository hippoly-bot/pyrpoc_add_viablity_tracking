import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image

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

def generate_data(num_channels=1, config=None):
    nx = config.get('numsteps_x', 200) if config else 200
    ny = config.get('numsteps_y', 200) if config else 200
    data_list = []
    for ch in range(num_channels):
        arr = np.random.uniform(0, 0.1, size=(ny, nx))
        offset_x = np.random.randint(-nx // 8, nx // 8)
        offset_y = np.random.randint(-ny // 8, ny // 8)
        center_x, center_y = nx // 2 + offset_x, ny // 2 + offset_y
        radius = min(nx, ny) // 4
        eye_offset = radius // 2
        eye_radius = radius // 8
        mouth_radius = radius // 2
        mouth_thickness = 2
        for x in range(nx):
            for y in range(ny):
                if ((x - (center_x - eye_offset))**2 + (y - (center_y + eye_offset))**2) < eye_radius**2:
                    arr[y, x] = 1.0
                if ((x - (center_x + eye_offset))**2 + (y - (center_y + eye_offset))**2) < eye_radius**2:
                    arr[y, x] = 1.0
                dist = ((x - center_x)**2 + (y - (center_y + eye_offset // 2))**2)**0.5
                if mouth_radius - mouth_thickness < dist < mouth_radius + mouth_thickness and y < center_y:
                    arr[y, x] = 1.0
        data_list.append(arr)
    return data_list

import tkinter as tk
from tkinter import ttk

class FeatureSelection(tk.Toplevel): 
    def __init__(self, master, feature_config): # lucky me, dictionaries are mutable by reference apparently
        super().__init__(master)
        self.title("Select Features")
        self.transient(master) 
        self.grab_set()        

        self.feature_config = feature_config # pass around the gui feature_config without passing the whole gui
        self.result = None     

        ttk.Label(self, text="Select which components you want to enable:").grid(
            row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w"
        )

        # TODO: add more, dont forget
        self.zaber_var = tk.BooleanVar(value=True)
        self.prior_var = tk.BooleanVar(value=True)
        self.rpoc_var  = tk.BooleanVar(value=True)

        row_index = 1
        ttk.Checkbutton(self, text="Zaber Delay Stage", variable=self.zaber_var).grid(
            row=row_index, column=0, sticky="w", padx=20
        )
        row_index += 1
        ttk.Checkbutton(self, text="Prior Stage", variable=self.prior_var).grid(
            row=row_index, column=0, sticky="w", padx=20
        )
        row_index += 1
        ttk.Checkbutton(self, text="RPOC Masking", variable=self.rpoc_var).grid(
            row=row_index, column=0, sticky="w", padx=20
        )

        row_index += 1
        ok_button = ttk.Button(self, text="OK", command=self.ok)
        ok_button.grid(row=row_index, column=0, columnspan=2, pady=10)

        self.protocol("WM_DELETE_WINDOW", self.ok) # make sure closing the window does the same thing as pressing

    def ok(self):
        self.feature_config["zaber"] = self.zaber_var.get()
        self.feature_config["prior"] = self.prior_var.get()
        self.feature_config['rpoc'] = self.rpoc_var.get()

        self.destroy()


def convert(data, type_=np.uint8):
    data_flipped = np.flipud(data)
    arr_norm = (data_flipped - data_flipped.min()) / (data_flipped.max() - data_flipped.min() + 1e-9)
    arr_typed = (arr_norm * 255).astype(type_)
    return Image.fromarray(arr_typed)
