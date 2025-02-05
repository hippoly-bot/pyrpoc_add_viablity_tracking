import tkinter as tk
from tkinter import ttk

class CollapsiblePane(ttk.Frame):
    """
    A collapsible pane with the old style aesthetic:
    - Uses a TCheckbutton (styled as a Toolbutton) as the toggle header
    - The "title" text is shown directly on that checkbutton
    - Clicking the checkbutton expands/collapses a container frame
    - Calls back to `gui.update_sidebar_visibility()` so your auto-resizing logic remains
    """
    def __init__(self, parent, text="", gui=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.gui = gui
        self.show = tk.BooleanVar(value=True)

        # Header area (like old code: a simple Frame with some padding)
        self.header = ttk.Frame(self, padding=(5, 2))
        self.header.pack(fill="x", expand=True)

        # The checkbutton that toggles expansion/collapse
        self.toggle_button = ttk.Checkbutton(
            self.header, text=text, variable=self.show,
            command=self.toggle, style="Toolbutton"
        )
        self.toggle_button.pack(side="left", fill="x", expand=True)

        # The container that holds actual child widgets (when expanded)
        self.container = ttk.Frame(self, padding=(5, 5))
        self.container.pack(fill="both", expand=True)

    def toggle(self):
        """Show/hide the container frame based on self.show."""
        if self.show.get():
            # If newly checked -> expand
            self.container.pack(fill="both", expand=True)
        else:
            # If unchecked -> collapse
            self.container.forget()

        # Let the main GUI readjust the sidebar width
        if self.gui is not None:
            self.gui.update_sidebar_visibility()


class ScrollableFrame(ttk.Frame):
    """
    A scrollable frame with the old code’s dark style and mouse-wheel handling.
    Ensures the background is consistent and we can scroll with the wheel.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Old "dark" aesthetic
        self.bg_color = "#3A3A3A"
        style = ttk.Style()
        style.configure("Dark.TFrame", background=self.bg_color)

        # Create a Canvas that will contain the scrollable frame
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=self.bg_color)
        # Use a themed vertical scrollbar
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview, style="Vertical.TScrollbar")

        # The actual frame in which we place widgets
        self.scrollable_frame = ttk.Frame(self.canvas, style="Dark.TFrame")
        self.scrollable_frame.bind("<Configure>", self.update_scroll_region)

        # Create a window within the canvas to hold that frame
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack canvas & scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Keep background updated and match widths
        self.bind("<Configure>", self.update_background)

        # Mouse wheel binding (Windows, Mac, and older X11 events)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def update_scroll_region(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_background(self, event=None):
        self.canvas.config(bg=self.bg_color)
        self.canvas.itemconfig(self.canvas_window, width=self.canvas.winfo_width())

    def _on_mousewheel(self, event):
        """
        Cross-platform scrolling on different OSes:
         - On Windows/Mac, <MouseWheel> event.delta is ±120 typically
         - On many Linux setups, Button-4 is scroll up; Button-5 is scroll down
        """
        if event.num == 4:      # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:    # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        else:
            # On Windows/Mac, event.delta is typically ±120
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(direction, "units")
