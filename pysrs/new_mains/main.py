import tkinter as tk
from pysrs.new_mains.gui import GUI

if __name__ == '__main__':
    root = tk.Tk()
    app = GUI(root)
    root.mainloop() 