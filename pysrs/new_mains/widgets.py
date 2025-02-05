import tkinter as tk
from tkinter import ttk

class CollapsiblePane:
    """
    A collapsible pane visually similar to your old code:
    - Has a LabelFrame header (with text)
    - Has a plus/minus toggle button in the header
    - A container frame that can be shown/hidden
    """
    def __init__(self, master, text="", gui=None, *args, **kwargs):
        self.show = tk.BooleanVar(value=True)
        self.text = text
        self.gui = gui

        # We do not inherit from ttk.Frame directly; instead we contain one
        self.frame = ttk.Frame(master, *args, **kwargs)

        # Our heading is a LabelFrame (old style) with text
        self.header = ttk.LabelFrame(self.frame, text=text)
        self.header.pack(fill="x", expand=True)

        # Our container is the collapsible portion
        self.container = ttk.Frame(self.frame)
        self.container.pack(fill="x", expand=True)

        # Toggle button: minus sign by default (pane shown)
        self.toggle_btn = ttk.Button(self.header, text='-', width=2, command=self.toggle)
        self.toggle_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        # By default, we also pack the entire frame so it can appear in the GUI
        # If you want to override this, remove these lines or call .pack(...) yourself
        self.frame.pack(fill="x", expand=True)

        # Start collapsed or expanded as determined by self.show
        self.toggle()

    def toggle(self):
        """Switch between showing/hiding the container frame."""
        if self.show.get():
            # If currently 'True', that means it's visible -> hide it
            self.container.pack_forget()
            self.show.set(False)
            self.toggle_btn.configure(text='+')
        else:
            # If currently 'False', that means it's hidden -> show it
            self.container.pack(fill="x", expand=True)
            self.show.set(True)
            self.toggle_btn.configure(text='-')

        # If the GUI has a method to re-adjust widths, call it
        if self.gui is not None and hasattr(self.gui, "update_sidebar_visibility"):
            self.gui.update_sidebar_visibility()

    # Expose pack/grid/place so external code can do something like
    # "my_collapsible_pane.pack(fill='x')" if desired
    def pack(self, *args, **kwargs):
        self.frame.pack(*args, **kwargs)

    def grid(self, *args, **kwargs):
        self.frame.grid(*args, **kwargs)

    def place(self, *args, **kwargs):
        self.frame.place(*args, **kwargs)

    @property
    def container_frame(self):
        """Access the container frame for adding widgets inside it."""
        return self.container


class ScrollableFrame(ttk.Frame):
    """
    A simple scrollable frame, same visuals as your old code:
    - A Canvas
    - A vertical Scrollbar
    - A child ttk.Frame inside the canvas.
    """
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # Create the Canvas + Scrollbar
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Frame inside the Canvas where content is placed
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Whenever the size of that frame changes, update the scroll region
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Create a window in the Canvas to hold 'scrollable_frame'
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Pack everything
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind a few events for resizing
        self.bind("<Configure>", self._on_parent_configure)

    def _on_parent_configure(self, event=None):
        """
        Keep the canvas' window width matched to the ScrollableFrame's width,
        so no horizontal scroll bar is needed.
        """
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())