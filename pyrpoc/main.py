def main():
    import tkinter as tk
    from pyrpoc.mains.gui import GUI

    root = tk.Tk()
    app = GUI(root)
    root.mainloop()