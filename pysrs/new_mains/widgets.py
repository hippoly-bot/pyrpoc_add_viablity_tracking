import tkinter as tk
from tkinter import ttk

class CollapsiblePane:
    def __init__(self, master, text="", gui=None, *args, **kwargs):
        self.show = tk.BooleanVar(value=True)
        self.text = text
        self.gui = gui
        self.frame = ttk.Frame(master)
        self.header = ttk.LabelFrame(self.frame, text=text)
        self.container = ttk.Frame(self.frame)
        self.header.pack(fill="x", expand=True)
        self.container.pack(fill="x", expand=True)
        self.toggle_btn = ttk.Button(self.header, text='-', width=2, command=self.toggle)
        self.toggle_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        self.frame.pack(fill="x", expand=True)
        self.toggle()

    def toggle(self):
        if self.show.get():
            self.container.pack_forget()
            self.show.set(False)
            self.toggle_btn.configure(text='+')
        else:
            self.container.pack(fill="x", expand=True)
            self.show.set(True)
            self.toggle_btn.configure(text='-')

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
