# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 11:09:04 2025

@author: cheritier
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
plt.close('all')
class OopaoGui:
    def __init__(self, root):
        self.root = root
        self.running = True

        # ---- Parameters (shared state) ----
        self.params = {"freq": 0.5,"cmap": "viridis"}

        # ---- Layout ----
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        plot_frame = ttk.Frame(main)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---- Matplotlib Figure ----
        # from matplotlib.figure import Figure

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        self.fig = Figure(figsize=(8, 4))
        self.axes = self.fig.subplots(1, 2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # ---- Controls ----
        ttk.Label(controls, text="Frequency").pack(pady=5)
        self.freq_slider = ttk.Scale(controls, from_=0., to=2.0,value=self.params["freq"],command=self.set_freq)
        self.freq_slider.pack(fill=tk.X)
        
                
        self.min_val = tk.DoubleVar(value=0.0)
        self.max_val = tk.DoubleVar(value=1.0)
        
        ttk.Label(controls, text="Min").pack()
        ttk.Scale(controls, from_=0.0, to=1.0,variable=self.min_val).pack(fill=tk.X)
        
        ttk.Label(controls, textvariable=self.min_val).pack()
        ttk.Label(controls, text="Max").pack()
        
        ttk.Scale(controls, from_=0.0, to=1.0,variable=self.max_val).pack(fill=tk.X)
        ttk.Label(controls, textvariable=self.max_val).pack()



        ttk.Button(controls, text="Toggle Colormap",command=self.toggle_cmap).pack(pady=5)
        ttk.Button(controls, text="Pause / Resume",command=self.toggle_running).pack(pady=5)

        # ---- Animation state ----
        self.t = 0
        self.update()

    # ---------- Control callbacks ----------
    def set_freq(self, val):
        self.params["freq"] = float(val)
        
    def set_freq(self, val):
            self.params["clim"] = float(val)

    def toggle_cmap(self):
        self.params["cmap"] = (
            "plasma" if self.params["cmap"] == "viridis" else "viridis"
        )

    def toggle_running(self):
        self.running = not self.running

    # ---------- Draw functions ----------
    def draw_plot(self, ax):
        x = np.linspace(0, self.t, 200)
        y = np.sin(self.params["freq"] * x )
        ax.plot(x, y)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("Sine Wave")

    def draw_image(self, ax):
        img = np.random.rand(50, 50)
        im = ax.imshow(img, cmap=self.params["cmap"])
        # plt.title("Random Image")
        im.set_clim([self.min_val.get(),self.max_val.get()])

    # ---------- Main update loop ----------
    def update(self):
        if self.running:
            self.axes[0].clear()
            self.axes[1].clear()

            self.draw_plot(self.axes[0])
            self.draw_image(self.axes[1])

            self.t += 0.1
            self.canvas.draw()

        self.root.after(100, self.update)

# ---- Run ----


root = tk.Tk()
root.title("OOPAO GUI")
OopaoGui(root)
root.mainloop()


#%%

