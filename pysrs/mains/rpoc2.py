import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import numpy as np

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

class RPOC:
    def __init__(self, root, image=None):
        self.root = root

        style = ttk.Style()
        style.theme_use('clam')
        self.bg_color = '#3A3A3A'
        self.fg_color = '#D0D0D0'
        self.highlight_color = '#4A90E2'
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("TButton", background='#444', foreground=self.fg_color, padding=6)
        style.configure("TCheckbutton", background=self.bg_color, foreground=self.fg_color)

        self.root.title('RPOC - Dark Mode')
        self.root.configure(bg=self.bg_color)

        if image is not None:
            if isinstance(image, np.ndarray):
                image = 255 * image / np.max(image)
                image = Image.fromarray(image.astype(np.uint8))
            grayscale = image.convert('L')
        else:
            grayscale = Image.open(r'C:\Users\Lab Admin\Documents\PythonStuff\pysrs\pysrs\data\image.jpg').convert('L')
            grayscale = grayscale.copy().resize((800, 800))
        self.original_image = Image.merge("RGB", (grayscale, grayscale, grayscale))
        self.img_width, self.img_height = self.original_image.size

        self.image = self.original_image.copy()
        self.binary_mask = Image.new('L', (self.img_width, self.img_height), 0)

        self.lower_threshold = tk.IntVar(value=80)
        self.upper_threshold = tk.IntVar(value=180)
        self.invert_var = tk.BooleanVar(value=False)
        self.eraser_var = tk.BooleanVar(value=False)
        self.fill_loop_var = tk.BooleanVar(value=True)

        self.build_ui()
        self.update_images()

        self.root.geometry('1200x800')
        self.root.update_idletasks()

        self.root.bind("<Configure>", self.on_resize)

    def build_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True)

        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(padx=5, pady=5, fill='both', expand=True)

        self.mask_canvas = tk.Canvas(self.display_frame, bg=self.bg_color, highlightthickness=0)
        self.mask_canvas.pack(side=tk.LEFT, fill='both', expand=True)

        self.preview_canvas = tk.Canvas(self.display_frame, bg=self.bg_color, highlightthickness=0)
        self.preview_canvas.pack(side=tk.LEFT, fill='both', expand=True)

        self.mask_canvas.bind("<Configure>", self.on_resize)
        self.preview_canvas.bind("<Configure>", self.on_resize)

        self.mask_image_id = None
        self.preview_image_id = None

        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.pack(padx=5, pady=5, fill='x')

        self.lower_slider = ColorSlider(
            controls_frame, min_val=0, max_val=255, init_val=self.lower_threshold.get(),
            width=200, height=20, fill_side='left', accent_color=self.highlight_color, bg_color='#505050',
            command=lambda val: [self.lower_threshold.set(val), self.update_images()]
        )
        self.lower_slider.pack(side=tk.LEFT, padx=5, pady=5)

        self.upper_slider = ColorSlider(
            controls_frame, min_val=0, max_val=255, init_val=self.upper_threshold.get(),
            width=200, height=20, fill_side='right', accent_color='#FF0000', bg_color='#505050',
            command=lambda val: [self.upper_threshold.set(val), self.update_images()]
        )
        self.upper_slider.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Checkbutton(
            controls_frame, text='Invert', variable=self.invert_var,
            command=self.update_images
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Checkbutton(
            controls_frame, text='Fill Loop', variable=self.fill_loop_var,
            command=self.update_images
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Checkbutton(
            controls_frame, text='Eraser', variable=self.eraser_var,
            command=self.update_images
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(
            controls_frame, text='Save Mask', command=self.save_mask
        ).pack(side=tk.LEFT, padx=5, pady=5)

        self.mask_canvas.bind('<ButtonPress-1>', self.start_drawing)
        self.mask_canvas.bind('<B1-Motion>', self.draw_mask)
        self.mask_canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        self.preview_canvas.bind('<ButtonPress-1>', self.start_drawing)
        self.preview_canvas.bind('<B1-Motion>', self.draw_mask)
        self.preview_canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        self.root.after(100, self.update_images)


    def on_resize(self, event=None):
        self.update_images()

    def get_base_image(self):
        if self.invert_var.get():
            return ImageOps.invert(self.original_image)
        return self.original_image

    def update_mask_image(self):
        self.mask_canvas.delete("all")        

        base = self.get_base_image()
        gray = base.convert('L')
        lower, upper = self.lower_threshold.get(), self.upper_threshold.get()

        gray_np = np.array(gray)
        rgb_np = np.stack([gray_np, gray_np, gray_np], axis=-1)

        self.valid_pixels = (gray_np >= lower) & (gray_np <= upper)

        rgb_np[gray_np < lower] = [0, 0, 255]
        rgb_np[gray_np > upper] = [255, 0, 0]

        mask_np = np.array(self.binary_mask)
        drawn = (mask_np == 255)
        valid_drawn = drawn & self.valid_pixels
        rgb_np[valid_drawn] = [0, 255, 0]

        thresholded_full = Image.fromarray(rgb_np.astype('uint8'), 'RGB')

        mask_w = self.mask_canvas.winfo_width()
        mask_h = self.mask_canvas.winfo_height()
        if mask_w < 2 or mask_h < 2:
            mask_w, mask_h = self.img_width, self.img_height

        thresholded_resized = thresholded_full.resize((mask_w, mask_h), Image.Resampling.LANCZOS)
        self.mask_image = thresholded_resized
        self.tk_mask_image = ImageTk.PhotoImage(self.mask_image)

        self.mask_image_id = self.mask_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_mask_image)

    def get_mask_applied_image(self):
        base = self.get_base_image()
        black_bg = Image.new("RGB", base.size, (0, 0, 0))
        return Image.composite(base, black_bg, self.binary_mask)

    def update_preview(self):
        self.preview_canvas.delete("all")

        preview_full = self.get_mask_applied_image()
        pw = self.preview_canvas.winfo_width()
        ph = self.preview_canvas.winfo_height()

        if pw < 2 or ph < 2:
            pw, ph = self.img_width, self.img_height

        preview_resized = preview_full.resize((pw, ph), Image.Resampling.LANCZOS)
        self.tk_preview_image = ImageTk.PhotoImage(preview_resized)

        self.preview_image_id = self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_preview_image)

    def update_images(self, event=None):
        self.update_mask_image()
        self.update_preview()

    def start_drawing(self, event):
        self.drawing = True
        self.points = [self._canvas_to_image_coords(event.widget, event.x, event.y)]

    def draw_mask(self, event):
        if not self.drawing:
            return

        current_point = self._canvas_to_image_coords(event.widget, event.x, event.y)
        if not (0 <= current_point[0] < self.img_width and 0 <= current_point[1] < self.img_height):
            return

        if not self.fill_loop_var.get():
            fill_val = 0 if self.eraser_var.get() else 255
            if self.eraser_var.get() or self.valid_pixels[current_point[1], current_point[0]]:
                draw_full = ImageDraw.Draw(self.binary_mask)
                draw_full.line([self.points[-1], current_point], fill=fill_val, width=2)
            self.update_images()

        self.points.append(current_point)

    def stop_drawing(self, event):
        self.drawing = False
        if len(self.points) < 2:
            return

        fill_val = 0 if self.eraser_var.get() else 255

        if not self.fill_loop_var.get():
            self.update_images()
            return

        temp_mask = Image.new("L", (self.img_width, self.img_height), 0)
        temp_draw = ImageDraw.Draw(temp_mask)
        temp_draw.polygon(self.points, outline=fill_val, fill=fill_val)

        mask_np = np.array(self.binary_mask)
        temp_mask_np = np.array(temp_mask)

        if self.eraser_var.get():
            mask_np[temp_mask_np == 255] = 0
        else:
            mask_np = np.where(self.valid_pixels, np.maximum(mask_np, temp_mask_np), mask_np)

        self.binary_mask = Image.fromarray(mask_np.astype('uint8'))
        self.update_images()

    def _canvas_to_image_coords(self, canvas_widget, cx, cy):
        canvas_w = canvas_widget.winfo_width()
        canvas_h = canvas_widget.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            return (0, 0)
        scale_x = self.img_width / canvas_w
        scale_y = self.img_height / canvas_h
        return (int(round(cx * scale_x)), int(round(cy * scale_y)))

    def save_mask(self):
        path = filedialog.asksaveasfilename(defaultextension='.png',
                                            filetypes=[('PNG files', '*.png')])
        if path:
            self.binary_mask.save(path)

if __name__ == '__main__':
    root = tk.Tk()
    app = RPOC(root)
    root.mainloop()
