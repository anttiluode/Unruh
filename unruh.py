"""Unruh: a falsifiable toy laboratory for an endogenous slow operator.

A 1-D ring of fast damped oscillators continually writes local correlations into
slow edge variables. Those edges can change subsequent wave propagation.
This is NOT quantum field theory, analog gravity, or a consciousness model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

MODES = ("coupled", "frozen", "instant", "observer")


@dataclass(frozen=True)
class Config:
    n: int = 24
    dt: float = 0.12
    stiffness: float = 2.5
    onsite: float = 0.35
    damping: float = 0.28
    cubic: float = 0.018
    tau: float = 60.0
    gain: float = 1.8
    feedback: float = 0.93

    def __post_init__(self):
        if self.n < 12 or self.dt <= 0 or self.stiffness <= 0 or self.tau <= self.dt:
            raise ValueError("Require n>=12, dt>0, stiffness>0, tau>dt")
        if not 0 <= self.feedback < 1 or self.damping < 0 or self.onsite < 0 or self.cubic < 0:
            raise ValueError("Require 0<=feedback<1 and nonnegative losses/potentials")


class World:
    """Ring with dynamic nearest-neighbor spring coefficients (not a QFT)."""

    def __init__(self, config: Config = Config(), mode: str = "coupled", seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.cfg, self.mode = config, mode
        self.x = np.zeros(config.n, dtype=float)
        self.v = np.zeros(config.n, dtype=float)
        self.h = np.zeros(config.n, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def edges(self) -> np.ndarray:
        """Effective edge weights; observer keeps h but cannot use it."""
        if self.mode in ("frozen", "observer"):
            return np.full(self.cfg.n, self.cfg.stiffness)
        return self.cfg.stiffness * (1.0 + self.cfg.feedback * self.h)

    def operator(self) -> np.ndarray:
        """A[h] in acceleration A[h]@x - damping*v - cubic*x**3."""
        n, w = self.cfg.n, self.edges()
        A = np.zeros((n, n), dtype=float)
        idx = np.arange(n)
        A[idx, idx] = -(w + np.roll(w, 1) + self.cfg.onsite)
        A[idx, (idx + 1) % n] = w
        A[idx, (idx - 1) % n] = np.roll(w, 1)
        return A

    def reset_fast(self):
        """Experimental intervention: erase x,v, while retaining true slow memory."""
        self.x.fill(0)
        self.v.fill(0)
        if self.mode == "instant":
            # An algebraically slaved operator may not carry stale information.
            self.h.fill(0)

    def reset_all(self):
        self.reset_fast()
        self.h.fill(0)
        self.steps = 0

    def step(self, force: np.ndarray | None = None, noise: float = 0.0):
        c, x, v = self.cfg, self.x, self.v
        if force is not None and np.shape(force) != (c.n,):
            raise ValueError(f"force must have shape ({c.n},)")
        if noise < 0:
            raise ValueError("noise must be nonnegative")
        # At every instant, algebraic h is determined by the current fast state.
        if self.mode == "instant":
            self.h[:] = np.tanh(c.gain * x * np.roll(x, -1))
        w = self.edges()
        flux = w * (np.roll(x, -1) - x)
        acceleration = flux - np.roll(flux, 1) - c.onsite*x - c.damping*v - c.cubic*x**3
        if force is not None:
            acceleration = acceleration + force
        if noise:
            acceleration = acceleration + self.rng.normal(0, noise, c.n)
        v += c.dt * acceleration
        x += c.dt * v
        target = np.tanh(c.gain * x * np.roll(x, -1))
        if self.mode == "instant":
            self.h[:] = target
        elif self.mode != "frozen":
            self.h += (c.dt / c.tau) * (target - self.h)
        self.steps += 1
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(self.h))):
            raise FloatingPointError("Integration diverged; reduce dt / drive")

    def field_energy(self) -> float:
        c = self.cfg
        strain = np.roll(self.x, -1) - self.x
        return float(0.5*np.sum(self.v**2 + self.edges()*strain**2 + c.onsite*self.x**2)
                     + 0.25*c.cubic*np.sum(self.x**4))


def write_then_probe(mode: str, label: int, nuisance_seed: int,
                     cfg: Config = Config(), write_steps: int = 180,
                     probe_steps: int = 155, sample_every: int = 5,
                     return_world: bool = False):
    """Write at one of two sites, ERASE x,v, probe remotely; paired nuisance.

    The test does not let a readout inspect h. Same nuisance_seed with either
    label makes write noise, amplitude, phase and probe noise matched exactly.
    """
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    w = World(cfg, mode, seed=nuisance_seed + 7919)
    rng = np.random.default_rng(nuisance_seed)
    amplitude = rng.uniform(0.8, 1.2)
    phase = rng.uniform(-np.pi, np.pi)
    write_site = cfg.n // 4 if label == 0 else (2 * cfg.n) // 3
    force = np.zeros(cfg.n)
    for t in range(write_steps):
        force.fill(0)
        force[write_site] = amplitude * np.sin(0.95 * cfg.dt * t + phase)
        w.step(force, noise=0.01)
    slow_norm = float(np.linalg.norm(w.h))
    w.reset_fast()  # Key intervention: no surviving fast traces.
    # Reset nuisance generator for the probe, independent of write trajectory.
    w.rng = np.random.default_rng(nuisance_seed + 104729)
    sensors = np.array([cfg.n//4, 3*cfg.n//8, cfg.n//2,
                        5*cfg.n//8, 3*cfg.n//4, 7*cfg.n//8])
    trace = []
    for t in range(probe_steps):
        force.fill(0)
        force[0] = 0.65 * np.sin(1.1 * cfg.dt * t)
        w.step(force, noise=0.008)
        if t % sample_every == 0:
            trace.extend(w.x[sensors])
    values = np.asarray(trace)
    return (values, slow_norm, w) if return_world else (values, slow_norm)


def evaluate(mode: str, seed: int, train_pairs: int, test_pairs: int,
             cfg: Config = Config()) -> dict:
    """Independent paired trials, held-out nuisance seeds, linear template readout."""
    train, train_y, test, test_y, norms = [], [], [], [], []
    for pair in range(train_pairs + test_pairs):
        nuisance = 1000003*seed + 97*pair + 17
        for label in (0, 1):
            features, norm = write_then_probe(mode, label, nuisance, cfg)
            (train if pair < train_pairs else test).append(features)
            (train_y if pair < train_pairs else test_y).append(label)
            norms.append(norm)
    X, Y = np.array(train), np.array(train_y)
    Z, T = np.array(test), np.array(test_y)
    means = np.array([X[Y == label].mean(axis=0) for label in (0, 1)])
    direction = means[1] - means[0]
    scores = (Z - means.mean(axis=0)) @ direction
    # If the templates are identical, consistently guess 0 => balanced 50%.
    predictions = (scores > 0).astype(int)
    return {"accuracy": float(np.mean(predictions == T)),
            "correct": int(np.sum(predictions == T)), "total": int(len(T)),
            "template_distance": float(np.linalg.norm(direction)),
            "mean_slow_norm_after_write": float(np.mean(norms))}


def benchmark(seeds: int = 3, train_pairs: int = 20, test_pairs: int = 20,
              cfg: Config = Config(), out: str | None = None) -> dict:
    if seeds < 1 or train_pairs < 2 or test_pairs < 2:
        raise ValueError("Require seeds>=1 and at least 2 pairs per split")
    records = []
    for mode in MODES:
        for seed in range(seeds):
            row = {"mode": mode, "seed": seed}
            row.update(evaluate(mode, seed, train_pairs, test_pairs, cfg))
            records.append(row)
            print(f"{mode:9s} seed={seed}: accuracy={row['accuracy']:.3f} "
                  f"template_distance={row['template_distance']:.6f}", flush=True)
    summary = {}
    for mode in MODES:
        rs = [r for r in records if r["mode"] == mode]
        summary[mode] = {"mean_accuracy": float(np.mean([r["accuracy"] for r in rs])),
                         "std_accuracy": float(np.std([r["accuracy"] for r in rs])),
                         "mean_template_distance": float(np.mean([r["template_distance"] for r in rs]))}
    result = {"protocol": "paired-nuisance fast-state-erasure remote-probe v1",
              "config": asdict(cfg), "seeds": seeds, "train_pairs": train_pairs,
              "test_pairs": test_pairs, "records": records, "summary": summary,
              "warning": "Toy classical adaptive spring network; no evidence of consciousness or analog gravity."}
    if out:
        dest = Path(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def gui():
    """Interactive desktop: watch fast state, slow memory and effective operator."""
    try:
        import tkinter as tk
        from tkinter import ttk
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except ImportError as e:
        raise SystemExit("GUI needs matplotlib and a Python build with tkinter") from e

    root = tk.Tk()
    root.title("Unruh — a medium that writes its own operator")
    mode = tk.StringVar(value="coupled")
    steps_per_frame = tk.IntVar(value=5)
    paused = tk.BooleanVar(value=False)
    delta_operator = tk.BooleanVar(value=True)
    world = World(Config(n=32))
    history = np.zeros((150, world.cfg.n))
    pending = {"kind": "", "elapsed": 0, "duration": 0, "site": 0}
    banner = tk.StringVar(value="Write left/right, erase fast state, then probe.")

    controls = ttk.Frame(root, padding=7)
    controls.pack(fill="x")
    ttk.Label(controls, text="Mode").pack(side="left")
    choices = ttk.Combobox(controls, textvariable=mode, values=MODES, width=10, state="readonly")
    choices.pack(side="left", padx=5)

    def remake(_=None):
        nonlocal world, history
        world = World(Config(n=32), mode.get())
        history = np.zeros((150, world.cfg.n))
        pending["duration"] = 0
        banner.set(f"Reset in {mode.get()} mode")
    choices.bind("<<ComboboxSelected>>", remake)

    def action(kind):
        if kind == "erase":
            world.reset_fast()
            banner.set("Fast x,v erased; slow h retained (except instant).")
            return
        if kind == "reset":
            remake()
            return
        pending.update(kind=kind, elapsed=0, duration=180 if kind.startswith("write") else 155,
                       site=world.cfg.n//4 if kind == "write L" else
                            2*world.cfg.n//3 if kind == "write R" else 0)
        banner.set(f"{kind}: injecting a pulse train")

    for label, action_key in (("Write left", "write L"), ("Write right", "write R"),
                              ("Erase fast", "erase"), ("Probe", "probe"), ("Reset all", "reset")):
        ttk.Button(controls, text=label, command=lambda a=action_key: action(a)).pack(side="left", padx=2)
    ttk.Checkbutton(controls, text="Pause", variable=paused).pack(side="left", padx=7)
    ttk.Label(controls, text="Steps/frame").pack(side="left")
    ttk.Spinbox(controls, from_=1, to=20, width=4, textvariable=steps_per_frame).pack(side="left", padx=4)
    ttk.Checkbutton(controls, text="Show Δoperator", variable=delta_operator).pack(side="left", padx=5)
    def open_slow_world():
        import subprocess
        import sys
        subprocess.Popen([sys.executable, str(Path(__file__).with_name("slow_world_gui.py"))])
    ttk.Button(controls, text="Slow world V2", command=open_slow_world).pack(side="left", padx=4)

    fig = Figure(figsize=(11, 7), dpi=95, layout="constrained")
    axs = fig.subplots(2, 2)
    ax_field, ax_slow, ax_operator, ax_history = axs.ravel()
    pos = np.arange(world.cfg.n)
    line, = ax_field.plot(pos, world.x, lw=2)
    ax_field.set(title="Fast field x — waves", xlabel="Ring node", ylabel="Displacement", ylim=(-1.4, 1.4))
    bars = ax_slow.bar(pos, world.h, width=0.85)
    ax_slow.set(title="Slow edge memory h", xlabel="Edge i → i+1", ylim=(-0.5, 0.5))
    op_im = ax_operator.imshow(np.zeros_like(world.operator()), vmin=-.1, vmax=.1, cmap="RdBu_r", interpolation="nearest")
    ax_operator.set(title="Δ operator A[h] − A[0]", xlabel="Source node", ylabel="Receiver node")
    hist_im = ax_history.imshow(history, aspect="auto", cmap="RdBu_r", origin="lower", vmin=-1, vmax=1)
    ax_history.set(title="Recent space-time history", xlabel="Node", ylabel="Old → new")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    ttk.Label(root, textvariable=banner, padding=8).pack(fill="x")

    def tick():
        if not paused.get():
            count = max(1, min(20, steps_per_frame.get()))
            for _ in range(count):
                force = np.zeros(world.cfg.n)
                if pending["elapsed"] < pending["duration"]:
                    t = pending["elapsed"]
                    kind = pending["kind"]
                    force[pending["site"]] = (0.95*np.sin(0.95*world.cfg.dt*t)
                                               if kind.startswith("write") else
                                               0.65*np.sin(1.1*world.cfg.dt*t))
                    pending["elapsed"] += 1
                    if pending["elapsed"] == pending["duration"]:
                        banner.set(f"{kind} complete. Slow h norm: {np.linalg.norm(world.h):.4f}")
                world.step(force)
                history[:-1] = history[1:]
                history[-1] = world.x
            line.set_ydata(world.x)
            ax_field.set_ylim(-max(0.5, 1.1*np.max(np.abs(world.x))), max(0.5, 1.1*np.max(np.abs(world.x))))
            for bar, height in zip(bars, world.h):
                bar.set_height(height)
            if delta_operator.get():
                base = World(world.cfg, world.mode).operator()
                current = world.operator() - base
                bound = max(.025, float(np.max(np.abs(current))))
                op_im.set_clim(-bound, bound)
                ax_operator.set_title("Δ operator A[h] − A[0]")
            else:
                current = world.operator()
                op_im.set_clim(-7, 3)
                ax_operator.set_title("Absolute acceleration operator A[h]")
            op_im.set_data(current)
            slow_limit = max(.025, 1.15 * float(np.max(np.abs(world.h))))
            ax_slow.set_ylim(-slow_limit, slow_limit)
            hist_im.set_data(history)
            fig.suptitle(f"{world.mode}  •  step {world.steps}  •  E={world.field_energy():.3f}  "
                         f"•  ||h||={np.linalg.norm(world.h):.4f}", fontsize=11)
            canvas.draw_idle()
        root.after(70, tick)
    tick()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    b = command.add_parser("bench", help="held-out causal remote-probe benchmark")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--train-pairs", type=int, default=20)
    b.add_argument("--test-pairs", type=int, default=20)
    b.add_argument("--feedback", type=float, default=0.93)
    b.add_argument("--tau", type=float, default=60)
    b.add_argument("--output", default="results/benchmark.json")
    command.add_parser("gui", help="interactive matplotlib/tkinter desktop GUI")
    args = parser.parse_args()
    if args.command == "bench":
        report = benchmark(args.seeds, args.train_pairs, args.test_pairs,
                           Config(feedback=args.feedback, tau=args.tau), args.output)
        print("SUMMARY:", json.dumps(report["summary"], indent=2))
        print("Saved", args.output)
    else:
        gui()


if __name__ == "__main__":
    main()
