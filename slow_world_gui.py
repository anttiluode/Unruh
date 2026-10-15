"""Live GUI for Unruh V2. Run: python slow_world_gui.py"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from emergence import slow_config, constant_drive, calibrate, equilibrate_fast
from unruh import World


def main():
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    c = slow_config()
    F = constant_drive(c.n)
    fit = calibrate(c, F)
    J = fit["jacobian"]
    root = tk.Tk()
    root.title("Unruh V2 — a slow world generated through a fast field")
    site_var, count = tk.IntVar(value=7), tk.IntVar(value=15)
    paused, powered = tk.BooleanVar(value=False), tk.BooleanVar(value=True)
    status = tk.StringVar(value="Inject one slow edge; watch induced changes at other edges.")
    baseline = perturbed = None
    predicted_a = predicted_b = None
    impulse, impulse_site = .2, 7
    hist = np.zeros((300, c.n))

    fig = Figure(figsize=(11.2, 7.6), dpi=95, layout="constrained")
    ax_fast, ax_h, ax_trace, ax_j = fig.subplots(2, 2).ravel()
    idx = np.arange(c.n)
    line_x, = ax_fast.plot(idx, np.zeros(c.n), lw=2)
    ax_fast.set(title="Fast field x", xlabel="Node", ylabel="Displacement")
    line_h, = ax_h.plot(idx, np.zeros(c.n), lw=2, label="Observed Δh")
    line_model, = ax_h.plot(idx, np.zeros(c.n), "--", lw=1.5, label="Slow-only prediction")
    ax_h.set(title="Slow response: measured vs effective law", xlabel="Edge i", ylabel="Δh")
    ax_h.legend(loc="upper right", fontsize=9)
    trace_im = ax_trace.imshow(hist.T, aspect="auto", origin="lower",
                               cmap="RdBu_r", vmin=-.012, vmax=.012)
    ax_trace.set(title="Induced slow response (direct decay subtracted)",
                 xlabel="Recent frames", ylabel="Edge i")
    bound = float(np.max(np.abs(J)))
    J_im = ax_j.imshow(J, cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax_j.set(title="Measured slow-to-slow Jacobian", xlabel="Perturbed edge j",
             ylabel="Responding edge i")
    fig.colorbar(J_im, ax=ax_j, shrink=.7)

    row = ttk.Frame(root, padding=8)
    row.pack(fill="x")
    ttk.Label(row, text="Impulse edge").pack(side="left")
    ttk.Spinbox(row, from_=0, to=c.n-1, textvariable=site_var, width=4).pack(side="left", padx=5)

    def reset():
        nonlocal baseline, perturbed, predicted_a, predicted_b, hist, impulse_site
        try:
            impulse_site = max(0, min(c.n-1, int(site_var.get())))
        except (ValueError, tk.TclError):
            impulse_site = 7
        site_var.set(impulse_site)
        baseline, perturbed = World(c), World(c)
        baseline.h[:] = fit["reference"]
        perturbed.h[:] = baseline.h
        perturbed.h[impulse_site] += impulse
        baseline.x[:] = equilibrate_fast(baseline.h, c, F)
        perturbed.x[:] = baseline.x
        predicted_a, predicted_b = baseline.h.copy(), perturbed.h.copy()
        hist = np.zeros((300, c.n))
        status.set(f"Injected slow edge {impulse_site}, amplitude {impulse:.2f}.")

    ttk.Button(row, text="Inject / reset", command=reset).pack(side="left", padx=5)
    ttk.Checkbutton(row, text="Pause", variable=paused).pack(side="left", padx=5)
    ttk.Checkbutton(row, text="DC pump on", variable=powered).pack(side="left", padx=5)
    ttk.Label(row, text="Steps/frame").pack(side="left")
    ttk.Spinbox(row, from_=1, to=50, textvariable=count, width=4).pack(side="left", padx=4)
    ttk.Button(row, text="What is colored?",
        command=lambda: messagebox.showinfo("Induced response",
            "The space-time colors show the slow perturbation after subtracting "
            "the isolated edge's direct exponential decay. Remaining changes "
            "must pass through the fast field. They are not a spacetime wave.")).pack(side="left", padx=5)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    ttk.Label(root, textvariable=status, padding=7).pack(fill="x")

    def tick():
        nonlocal hist
        if not paused.get():
            try:
                steps = max(1, min(50, int(count.get())))
            except (ValueError, tk.TclError):
                steps = 15
            for _ in range(steps):
                pump = F if powered.get() else np.zeros(c.n)
                baseline.step(pump)
                perturbed.step(pump)
                for h in (predicted_a, predicted_b):
                    h += c.dt/c.tau*(fit["target"]+J@(h-fit["reference"])-h)
                delta = perturbed.h-baseline.h
                isolated = np.zeros(c.n)
                isolated[impulse_site] = impulse*(1-c.dt/c.tau)**perturbed.steps
                hist[:-1] = hist[1:]
                hist[-1] = delta-isolated
            delta = perturbed.h-baseline.h
            predicted = predicted_b-predicted_a
            line_x.set_ydata(perturbed.x)
            line_h.set_ydata(delta)
            line_model.set_ydata(predicted)
            amp = max(.5, 1.15*float(np.max(np.abs(perturbed.x))))
            ax_fast.set_ylim(-amp, amp)
            amp = max(.015, 1.2*float(np.max(np.abs(np.r_[delta,predicted]))))
            ax_h.set_ylim(-amp, amp)
            trace_im.set_data(hist.T)
            amp = max(.002, float(np.max(np.abs(hist))))
            trace_im.set_clim(-amp, amp)
            offsite = delta.copy()
            offsite[impulse_site] = 0
            error = float(np.sqrt(np.mean((delta-predicted)**2)))
            status.set(f"t={perturbed.steps*c.dt:.1f} • induced offsite ||Δh||="
                f"{np.linalg.norm(offsite):.5f} • law RMSE={error:.5f} • " +
                ("pump ON: valid calibration" if powered.get() else
                 "PUMP OFF: effective law outside calibration conditions"))
            canvas.draw_idle()
        root.after(75, tick)

    reset()
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
