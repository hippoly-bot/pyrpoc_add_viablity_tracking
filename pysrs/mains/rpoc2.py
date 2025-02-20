import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import numpy as np
import random

@staticmethod
def generate_synthetic_image(width, height, num_cells=10):
    image = np.full((height, width), 50, dtype=np.float32)
    Y, X = np.ogrid[:height, :width]
    for _ in range(num_cells):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        radius = random.randint(20, 50)
        mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
        mask_indices = np.where(mask)
        distances = np.sqrt((mask_indices[1] - cx)**2 + (mask_indices[0] - cy)**2)
        cell_intensity = np.clip(255 - 2 * distances, 0, 255)
        image[mask_indices] = np.maximum(image[mask_indices], cell_intensity)
    noise = np.random.normal(0, 10, (height, width))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    image_rgb = np.stack([image, image, image], axis=-1)
    return image_rgb


class RPOC:
    def __init__(self, parent, image=None):
        self.root = parent  # parent must be root or toplevel, this is how it is called from the gui
        self.root.title("RPOC Mask Editor")
 
        # TODO: move this into a config with the 
        self.bg_color = "#2E2E2E"
        self.fg_color = "#D0D0D0"
        self.highlight_color = "#4A90E2"
        self.button_bg = "#444"
        self.root.configure(bg=self.bg_color)

        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure("Dark.TFrame", background=self.bg_color)
        style.configure("Dark.TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("Dark.TButton", background=self.button_bg, foreground=self.fg_color, padding=6)
        style.map("Dark.TButton",
                  background=[('active', self.highlight_color)],
                  foreground=[('active', self.fg_color)])
        style.configure("Dark.TCheckbutton", background=self.bg_color, foreground=self.fg_color)

        if image is None: # this doesnt work sadly
            img_arr = generate_synthetic_image(800, 600, num_cells=10)
            self.original_image = Image.fromarray(img_arr, mode="RGB")
        else:
            if isinstance(image, np.ndarray):
                image = (255.0 * image / np.max(image)).astype(np.uint8)
                self.original_image = Image.fromarray(image).convert("RGB")
            else:
                self.original_image = image.convert("RGB") # assume already a PIL image

        self.img_width, self.img_height = self.original_image.size

        # original thresholding gray
        self.gray_image = self.original_image.convert("L")
        self.gray_np = np.array(self.gray_image)

        self.lower_threshold = 80
        self.upper_threshold = 180

        self.brush_size = 5
        self.eraser_mode = False
        self.fill_loop_mode = False

        self.invert_mode = False  # FIXME: invert works but it resets the whole image??

        self.mask_np = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        self.create_ui()

        self.root.update_idletasks()
        self.root.minsize(1000, 700)
        self.root.after(100, self.force_resize)

        self.update_all_displays()

    def force_resize(self):
        self.root.update_idletasks()

    def get_current_gray(self):
        if self.invert_mode:
            return 255 - self.gray_np
        else:
            return self.gray_np

    def create_ui(self):
        self.main_frame = ttk.Frame(self.root, style="Dark.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(self.main_frame, padding=(5, 5), style="Dark.TFrame")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Lower Threshold", style="Dark.TLabel").grid(row=0, column=0, padx=5, pady=5)

        self.lower_slider = ColorSlider(
            control_frame, min_val=0, max_val=255, init_val=self.lower_threshold,
            width=200, height=20, fill_side='left', accent_color='#FF0000',
            bg_color=self.bg_color,
            command=lambda val: self.on_threshold_change()
        )
        self.lower_slider.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Upper Threshold", style="Dark.TLabel").grid(row=0, column=2, padx=5, pady=5)
        self.upper_slider = ColorSlider(
            control_frame, min_val=0, max_val=255, init_val=self.upper_threshold,
            width=200, height=20, fill_side='right', accent_color='#0000FF',
            bg_color=self.bg_color,
            command=lambda val: self.on_threshold_change()
        )
        self.upper_slider.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(control_frame, text="Brush Size", style="Dark.TLabel").grid(row=1, column=0, padx=5, pady=5)
        self.brush_slider = tk.Scale(
            control_frame, from_=1, to=20, orient=tk.HORIZONTAL,
            command=self.on_brush_size_change,
            bg=self.bg_color, fg=self.fg_color, highlightbackground=self.bg_color,
            troughcolor="#505050", activebackground="#4A4A4A", bd=0
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.grid(row=1, column=1, padx=5, pady=5)

        self.eraser_var = tk.BooleanVar(value=self.eraser_mode)
        self.fill_loop_var = tk.BooleanVar(value=self.fill_loop_mode)

        self.eraser_cb = ttk.Checkbutton(control_frame, text="Eraser",
                                    style="Dark.TCheckbutton",
                                    variable=self.eraser_var,
                                    command=self.on_mode_change)
        self.eraser_cb.grid(row=1, column=2, padx=5, pady=5)

        self.fillloop_cb = ttk.Checkbutton(control_frame, text="Fill Loop",
                                      style="Dark.TCheckbutton",
                                      variable=self.fill_loop_var,
                                      command=self.on_mode_change)
        self.fillloop_cb.grid(row=1, column=3, padx=5, pady=5)

        self.invert_var = tk.BooleanVar(value=False)
        self.invert_cb = ttk.Checkbutton(control_frame, text="Invert",
                                    style="Dark.TCheckbutton",
                                    variable=self.invert_var,
                                    command=self.on_invert_toggled)
        self.invert_cb.grid(row=2, column=0, padx=5, pady=5)

        self.select_all_btn = ttk.Button(control_frame, text="Select All Active Pixels",
                                    style="Dark.TButton",
                                    command=self.select_all_active)
        self.select_all_btn.grid(row=2, column=1, padx=5, pady=5)

        self.save_mask_btn = ttk.Button(control_frame, text="Save Mask",
                                   style="Dark.TButton",
                                   command=self.save_mask)
        self.save_mask_btn.grid(row=2, column=2, padx=5, pady=5, columnspan=2)

        canvas_frame = ttk.Frame(self.main_frame, style="Dark.TFrame")
        canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.main_canvas = tk.Canvas(canvas_frame, bg=self.bg_color, highlightthickness=0)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.main_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.main_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        self.preview_canvas = tk.Canvas(canvas_frame, bg=self.bg_color, highlightthickness=0)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def on_threshold_change(self):
        self.lower_threshold = self.lower_slider.get()
        self.upper_threshold = self.upper_slider.get()
        self.update_all_displays()

    def on_invert_toggled(self):
        self.invert_mode = self.invert_var.get()
        self.update_all_displays()

    def on_brush_size_change(self, _):
        self.brush_size = int(self.brush_slider.get())

    def on_mode_change(self):
        print(f'fill: {self.fill_loop_var.get()}, eraser: {self.eraser_var.get()}')
        if self.fill_loop_var.get() and self.eraser_var.get():
            self.fill_loop_var.set(False)
        self.eraser_mode = self.eraser_var.get()
        self.fill_loop_mode = self.fill_loop_var.get()

        # if self.eraser_mode:
        #     self.fill_loop_var.set(False)
        #     self.fill_loop_mode = False
        #     print('fill loop turned off because eraser was on')
        # if self.fill_loop_mode:
        #     self.eraser_var.set(False)
        #     self.eraser_mode = False
        #     print('eraser turned off because fill loop was on \n')
        

    def select_all_active(self):
        gray_to_use = self.get_current_gray()
        active = (gray_to_use >= self.lower_threshold) & (gray_to_use <= self.upper_threshold)
        self.mask_np[active] = 255
        self.update_all_displays()

    def save_mask(self):
        path = filedialog.asksaveasfilename(defaultextension='.png',
                                            filetypes=[('PNG Files', '*.png')],
                                            title='Save Mask As')
        if path:
            Image.fromarray(self.mask_np).save(path)

    def update_all_displays(self):
        self.update_main_display()
        self.update_preview_display()

    def update_main_display(self):
        gray_to_use = self.get_current_gray()
        below = gray_to_use < self.lower_threshold
        above = gray_to_use > self.upper_threshold

        img_arr = np.array(self.original_image).copy()
        img_arr[below] = [255, 0, 0]
        img_arr[above] = [0, 0, 255]
        mask_pixels = (self.mask_np == 255)
        img_arr[mask_pixels] = [0, 255, 0]

        display_img = Image.fromarray(img_arr)
        cw = self.main_canvas.winfo_width() or self.img_width
        ch = self.main_canvas.winfo_height() or self.img_height
        display_img = display_img.resize((cw, ch), Image.Resampling.LANCZOS)
        self.tk_main_img = ImageTk.PhotoImage(display_img)
        self.main_canvas.delete("all")
        self.main_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_main_img)

    def update_preview_display(self):
        cw = self.preview_canvas.winfo_width() or self.img_width
        ch = self.preview_canvas.winfo_height() or self.img_height
        mask_img = Image.fromarray(np.where(self.mask_np == 255, 255, 0).astype(np.uint8))
        mask_img = mask_img.resize((cw, ch), Image.Resampling.LANCZOS)
        self.tk_preview_img = ImageTk.PhotoImage(mask_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_preview_img)

    def _canvas_to_image_coords(self, canvas, x, y):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        img_x = int(x / (cw if cw > 0 else 1) * self.img_width)
        img_y = int(y / (ch if ch > 0 else 1) * self.img_height)
        return img_x, img_y

    def on_canvas_press(self, event):
        self.drawing = True
        self.current_points = []
        pt = self._canvas_to_image_coords(event.widget, event.x, event.y)
        self.current_points.append(pt)

    def on_canvas_drag(self, event):
        if not self.drawing:
            return

        pt = self._canvas_to_image_coords(event.widget, event.x, event.y)
        self.current_points.append(pt)

        radius = self.brush_size
        event.widget.create_oval(event.x - radius, event.y - radius,
                                 event.x + radius, event.y + radius,
                                 fill=self.highlight_color, outline="")

    def on_canvas_release(self, event):
        if not self.drawing:
            return
        self.drawing = False

        if self.fill_loop_mode and len(self.current_points) >= 3:
            self.apply_polygon_to_mask(self.current_points, 0 if self.eraser_mode else 255)
        else:
            self.apply_line_to_mask(self.current_points, 0 if self.eraser_mode else 255)

        self.current_points = []
        self.update_all_displays()

    def apply_line_to_mask(self, points, value):
        for (x, y) in points:
            y_idx, x_idx = np.ogrid[:self.img_height, :self.img_width]
            dist = 2*np.sqrt((x_idx - x)**2 + (y_idx - y)**2)
            brush_mask = dist <= self.brush_size
            if value == 0:  
                self.mask_np[brush_mask] = 0
            else:
                gray_to_use = self.get_current_gray()
                valid = (gray_to_use >= self.lower_threshold) & (gray_to_use <= self.upper_threshold)
                update_area = brush_mask & valid
                self.mask_np[update_area] = 255

    def apply_polygon_to_mask(self, points, value):
        poly_img = Image.new("L", (self.img_width, self.img_height), 0)
        draw = ImageDraw.Draw(poly_img)
        draw.polygon(points, outline=value, fill=value)
        poly_np = np.array(poly_img)

        if value == 0:
            self.mask_np[poly_np == 255] = 0
        else:
            gray_to_use = self.get_current_gray()
            valid = (gray_to_use >= self.lower_threshold) & (gray_to_use <= self.upper_threshold)
            self.mask_np[np.logical_and(poly_np == 255, valid)] = 255


class ColorSlider(tk.Canvas):
    def __init__(self, master, min_val=0, max_val=255, init_val=0, width=200, height=20,
                 fill_side='left', accent_color='#4A90E2', bg_color='#505050', command=None, **kwargs):
        super().__init__(master, width=width, height=height, bg=bg_color, highlightthickness=0, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.value = init_val
        self.slider_width = width
        self.slider_height = height
        self.fill_side = fill_side
        self.accent_color = accent_color
        self.track_color = '#808080'
        self.command = command
        self.knob_radius = height // 2
        self.margin = self.knob_radius
        self.bind("<Button-1>", self.click)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.release)
        self.draw_slider()

    def draw_slider(self):
        self.delete("all")
        self.create_line(self.margin, self.slider_height/2,
                         self.slider_width - self.margin, self.slider_height/2,
                         fill=self.track_color, width=4)
        pos = self.margin + (self.value - self.min_val) / (self.max_val - self.min_val) * (self.slider_width - 2*self.margin)
        if self.fill_side == 'left':
            self.create_line(self.margin, self.slider_height/2, pos, self.slider_height/2,
                             fill=self.accent_color, width=4)
        else:
            self.create_line(pos, self.slider_height/2, self.slider_width - self.margin, self.slider_height/2,
                             fill=self.accent_color, width=4)
        self.create_oval(pos - self.knob_radius, self.slider_height/2 - self.knob_radius,
                         pos + self.knob_radius, self.slider_height/2 + self.knob_radius,
                         fill='#D0D0D0', outline="")

    def click(self, event):
        self.set_value_from_event(event.x)

    def drag(self, event):
        self.set_value_from_event(event.x)

    def release(self, event):
        self.set_value_from_event(event.x)

    def set_value_from_event(self, x):
        x = max(self.margin, min(self.slider_width - self.margin, x))
        ratio = (x - self.margin) / (self.slider_width - 2*self.margin)
        new_val = int(self.min_val + ratio * (self.max_val - self.min_val))
        self.value = new_val
        self.draw_slider()
        if self.command:
            self.command(new_val)

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        self.draw_slider()
        if self.command:
            self.command(value)

@staticmethod
def build_rpoc_wave(mask_image, pixel_samples, total_x, total_y, high_voltage=5.0):
    mask_arr = np.array(mask_image)
    binary_mask = (mask_arr > 128).astype(np.uint8)

    if binary_mask.shape != (total_y, total_x):
        mask_pil = Image.fromarray(binary_mask * 255)
        mask_resized = mask_pil.resize((total_x, total_y), Image.NEAREST)
        binary_mask = (np.array(mask_resized) > 128).astype(np.uint8)

    ttl_rows = [
        np.repeat(binary_mask[row, :], pixel_samples)
        for row in range(total_y)
    ]
    ttl_wave = np.concatenate(ttl_rows)
    ttl_wave = ttl_wave * high_voltage
    return ttl_wave


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    editor = RPOC(root)  # no image => gradient
    root.mainloop()