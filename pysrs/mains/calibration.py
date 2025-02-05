# calibration.py
import tkinter as tk
from tkinter import ttk, messagebox
import time, threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pysrs.instruments.galvo_funcs import Galvo
from pysrs.runners.run_image_2d import lockin_scan

def calibrate_stage(gui):
    cal_win = tk.Toplevel(gui.root)
    cal_win.title("Stage Calibration")
    cal_win.geometry("900x600")

    config_frame = ttk.Frame(cal_win, padding=10)
    config_frame.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(config_frame, text='Start Position (µm)').grid(row=0, column=0, sticky='w', padx=5, pady=3)
    start_entry = ttk.Entry(config_frame, width=12, font=('Calibri', 14))
    start_entry.insert(0, str(gui.hyper_config['start_um']))
    start_entry.grid(row=0, column=1, padx=5, pady=3)

    ttk.Label(config_frame, text='Stop Position (µm)').grid(row=1, column=0, sticky='w', padx=5, pady=3)
    stop_entry = ttk.Entry(config_frame, width=12, font=('Calibri', 14))
    stop_entry.insert(0, str(gui.hyper_config['stop_um']))
    stop_entry.grid(row=1, column=1, padx=5, pady=3)

    ttk.Label(config_frame, text='Number of Steps').grid(row=2, column=0, sticky='w', padx=5, pady=3)
    cal_steps_entry = ttk.Entry(config_frame, width=10, font=('Calibri', 14))
    cal_steps_entry.insert(0, '10')
    cal_steps_entry.grid(row=2, column=1, padx=5, pady=3)

    start_button = ttk.Button(config_frame, text='Start Calibration', style='TButton')
    start_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title('Calibration Data')
    ax.set_xlabel('Stage Position (µm)')
    ax.set_ylabel('Average Intensity')

    canvas = FigureCanvasTkAgg(fig, master=cal_win)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    cal_running = [False]

    def run_calibration():
        cal_running[0] = True
        try:
            start_val = float(start_entry.get().strip())
            stop_val = float(stop_entry.get().strip())
            n_steps = int(cal_steps_entry.get().strip())
            if n_steps < 1 or start_val >= stop_val:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid calibration settings.")
            cal_running[0] = False
            return

        try:
            gui.zaber_stage.connect()
        except Exception as e:
            messagebox.showerror("Zaber Error", str(e))
            cal_running[0] = False
            return

        positions_to_scan = [start_val] if n_steps == 1 else [start_val + i * (stop_val - start_val) / (n_steps - 1) for i in range(n_steps)]
        positions, intensities = [], []

        for pos in positions_to_scan:
            if not cal_running[0]:
                break
            try:
                gui.zaber_stage.move_absolute_um(pos)
            except Exception as e:
                messagebox.showerror("Zaber Error", str(e))
                cal_running[0] = False
                break

            if gui.simulation_mode.get():
                data_list = gui.generate_data(len(gui.config['ai_chans']))
                data = data_list[0]
            else:
                galvo = Galvo(gui.config)
                data_list = lockin_scan([f"{gui.config['device']}/{ch}" for ch in gui.config['ai_chans']], galvo)
                data = data_list[0]

            avg_val = data.mean()
            positions.append(pos)
            intensities.append(avg_val)

            ax.clear()
            ax.set_title('Calibration Data')
            ax.set_xlabel('Stage Position (µm)')
            ax.set_ylabel('Average Intensity')
            ax.plot(positions, intensities, '-o', color='blue')
            canvas.draw()
            canvas.flush_events()

            time.sleep(0.2)

        cal_running[0] = False

    def start_cal():
        if not cal_running[0]:
            threading.Thread(target=run_calibration, daemon=True).start()

    start_button.configure(command=start_cal)

    stop_button = ttk.Button(config_frame, text='Stop Calibration', style='TButton',
                             command=lambda: cal_running.__setitem__(0, False))
    stop_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
