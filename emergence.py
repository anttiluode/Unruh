"""Unruh V2: measure whether a self-written operator admits a closed slow law.

Calibrate a constitutive h-only response under constant forcing; evaluate on
unseen fully resolved trajectories with fresh microscopic initial conditions.
No spatial h->h coupling is written into the microscopic model. Any measured
off-diagonal coupling must travel through fast x,v dynamics.

This is a classical, driven toy model, not quantum physics or consciousness.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from unruh import Config, World


def slow_config(n: int = 24, tau: float = 120., feedback: float = .93) -> Config:
    return Config(n=n, dt=.12, stiffness=2.5, onsite=.2, damping=.9,
                  cubic=.018, tau=tau, gain=3., feedback=feedback)


def constant_drive(n: int, amplitude: float = 3.) -> np.ndarray:
    """Four alternating DC pumps: the effective world is explicitly driven."""
    F = np.zeros(n)
    for site, sign in ((0, 1), (n//4, -.8), (n//2, 1), (3*n//4, -.8)):
        F[site] += amplitude * sign
    return F


def equilibrate_fast(h: np.ndarray, c: Config, force: np.ndarray,
                     tol: float = 1e-11) -> np.ndarray:
    """Solve the clamped fast force balance, for calibration/trial initialization.

    Slow-only predictors NEVER use this function during rollout.
    """
    if h.shape != (c.n,) or force.shape != (c.n,):
        raise ValueError("h and force require shape (n,)")
    w = World(c)
    w.h[:] = h
    A = w.operator()
    x = np.linalg.solve(-A, force)
    for _ in range(30):
        residual = A @ x - c.cubic*x**3 + force
        dx = np.linalg.solve(A - np.diag(3*c.cubic*x*x), residual)
        x -= dx
        if np.max(np.abs(dx)) < tol:
            break
    if np.max(np.abs(A @ x - c.cubic*x**3 + force)) > 1e-7:
        raise RuntimeError("Fast equilibrium solver did not converge")
    return x


def correlation_target(h: np.ndarray, c: Config, force: np.ndarray) -> np.ndarray:
    x = equilibrate_fast(h, c, force)
    return np.tanh(c.gain*x*np.roll(x, -1))


def slow_fixed_point(c: Config, force: np.ndarray) -> np.ndarray:
    """Approximate driven fixed point h*=target(h*)."""
    h = np.zeros(c.n)
    for _ in range(2000):
        h_new = .85*h + .15*correlation_target(h, c, force)
        if np.max(np.abs(h_new-h)) < 1e-10:
            h = h_new
            break
        h = h_new
    if np.max(np.abs(correlation_target(h, c, force)-h)) > 1e-7:
        raise RuntimeError("No stable slow fixed point found")
    return h


def calibrate(c: Config, force: np.ndarray, delta: float = .04) -> dict:
    """Finite difference 2N+1 clamped perturbations in *slow* coordinates.

    J_ij = d(tanh(gain*x_i*(x_next_i)))/d(h_j), where x=x*(h).
    J off-diagonal is mediated by the fast substrate, not programmed h diffusion.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    h_star = slow_fixed_point(c, force)
    t_star = correlation_target(h_star, c, force)
    J = np.empty((c.n, c.n))
    for j in range(c.n):
        hp, hm = h_star.copy(), h_star.copy()
        hp[j] += delta
        hm[j] -= delta
        J[:, j] = (correlation_target(hp, c, force) -
                   correlation_target(hm, c, force)) / (2*delta)
    return {"reference": h_star, "target": t_star, "jacobian": J}


def forecast(h0: np.ndarray, fit: dict, c: Config, steps: int,
             model: str = "full", stride: int = 1) -> np.ndarray:
    """Open-loop h-only prediction, without fast variables or equilibrium calls."""
    if model not in ("full", "nearest", "diagonal", "decay", "persistence"):
        raise ValueError("Unknown slow model")
    J = fit["jacobian"]
    if model == "full":
        B = J
    elif model == "nearest":
        idx = np.arange(c.n)
        d = np.abs(idx[:, None] - idx[None, :])
        B = np.where(np.minimum(d, c.n-d) <= 1, J, 0.)
    elif model == "diagonal":
        B = np.diag(np.diag(J))
    else:
        B = None
    h = h0.copy()
    out = [h.copy()]
    for t in range(steps):
        if model != "persistence":
            target = fit["target"] if B is None else fit["target"] + B@(h-fit["reference"])
            h += (c.dt/c.tau)*(target-h)
        if (t+1) % stride == 0:
            out.append(h.copy())
    return np.array(out)


def micro_rollout(h0: np.ndarray, c: Config, force: np.ndarray, steps: int,
                  stride: int, seed: int, perturb: float = .15) -> np.ndarray:
    """Unseen fully resolved x,v,h evolution, fast initial conditions randomized."""
    w = World(c, mode="coupled", seed=seed)
    w.h[:] = h0
    rng = np.random.default_rng(seed)
    w.x[:] = equilibrate_fast(h0, c, force) + rng.normal(0, perturb, c.n)
    w.v[:] = rng.normal(0, perturb, c.n)
    hist = [w.h.copy()]
    for t in range(steps):
        w.step(force)
        if (t+1) % stride == 0:
            hist.append(w.h.copy())
    return np.array(hist)


def make_test_states(hstar: np.ndarray, seed: int, trials: int,
                     magnitude: float = .16) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(trials):
        h = hstar.copy()
        indices = rng.choice(len(h), size=3, replace=False)
        h[indices] += rng.uniform(-magnitude, magnitude, size=3)
        cases.append(np.clip(h, -.9, .9))
    return cases


def closure_gate(c: Config, force: np.ndarray, fit: dict, trials: int = 12,
                 steps: int = 1000, stride: int = 50, seed: int = 30) -> dict:
    """Hold out both h initial conditions and fast microstate nuisance seeds."""
    cases = make_test_states(fit["reference"], seed, trials)
    errors = {m: [] for m in ("full", "nearest", "diagonal", "decay", "persistence")}
    example = None
    for trial, h0 in enumerate(cases):
        actual = micro_rollout(h0, c, force, steps, stride, seed=1000*seed+trial)
        predictions = {m: forecast(h0, fit, c, steps, m, stride) for m in errors}
        start = max(1, (steps//stride)//5)
        for m, predicted in predictions.items():
            errors[m].append(float(np.sqrt(np.mean((actual[start:]-predicted[start:])**2))))
        if trial == 0:
            example = {"true": actual.tolist(),
                       **{m: v.tolist() for m,v in predictions.items()}}
    return {"rmse_by_model": {m: float(np.mean(v)) for m,v in errors.items()},
            "per_trial_rmse": errors, "trials": trials,
            "horizon_physical_time": steps*c.dt,
            "score_excludes_early_time": max(1, (steps//stride)//5)*stride*c.dt,
            "example_trajectory": example}


def perturbation_gate(c: Config, force: np.ndarray, fit: dict, site: int = 7,
                      magnitude: float = .2, steps: int = 1200,
                      stride: int = 20) -> dict:
    """Paired impulse with and without feedback. No h edge directly touches another."""
    if not 0 <= site < c.n:
        raise ValueError("site out of range")
    if not 0 < magnitude < .5:
        raise ValueError("0 < magnitude < .5 required")
    outputs = {}
    for feedback in (c.feedback, 0.):
        cfg = replace(c, feedback=feedback)
        base, bumped = World(cfg), World(cfg)
        hbase = fit["reference"] if feedback else slow_fixed_point(cfg, force)
        base.h[:] = hbase
        bumped.h[:] = hbase
        bumped.h[site] += magnitude
        base.x[:] = equilibrate_fast(hbase, cfg, force)
        bumped.x[:] = base.x
        rows = [bumped.h-base.h]
        for t in range(steps):
            base.step(force)
            bumped.step(force)
            if (t+1) % stride == 0:
                rows.append((bumped.h-base.h).copy())
        outputs["coupled" if feedback else "feedback_zero"] = np.array(rows)
    active = outputs["coupled"]
    outside = active.copy()
    outside[:, site] = 0.
    return {"site": site, "magnitude": magnitude,
            "max_offsite_amplitude": float(np.max(np.abs(outside))),
            "final_offsite_norm": float(np.linalg.norm(outside[-1])),
            "zero_feedback_max_offsite": float(np.max(np.abs(
                np.delete(outputs["feedback_zero"], site, axis=1)))),
            "traces": {k:v.tolist() for k,v in outputs.items()},
            "dt_between_samples": stride*c.dt}


def run_benchmark(trials: int = 12, steps: int = 1000,
                  output: str | None = "results/slow_world.json",
                  cfg: Config | None = None) -> dict:
    c = cfg or slow_config()
    force = constant_drive(c.n)
    fit = calibrate(c, force)
    J = fit["jacobian"]
    off = J.copy()
    np.fill_diagonal(off, 0)
    closure = closure_gate(c, force, fit, trials=trials, steps=steps, stride=50)
    perturb = perturbation_gate(c, force, fit)
    report = {
        "protocol": "Unruh V2: equilibrated effective law + held-out full dynamics",
        "description": "Classical driven spring ring with slow plastic edges, not emergent spacetime",
        "config": asdict(c), "constant_drive": force.tolist(),
        "calibration": {"h_star": fit["reference"].tolist(), "target_star": fit["target"].tolist(),
                        "jacobian": J.tolist(), "max_offdiagonal": float(np.max(np.abs(off))),
                        "offdiagonal_frobenius": float(np.linalg.norm(off)),
                        "diagonal_frobenius": float(np.linalg.norm(np.diag(np.diag(J))))},
        "closure": closure, "impulse": perturb,
        "interpretation": "Mediated slow coupling under a continuing DC drive, not a self-sustaining world."
    }
    for name, value in closure["rmse_by_model"].items():
        print(f"Held-out slow forecast {name:11s} RMSE={value:.7f}")
    print("Max effective cross-edge Jacobian:", report["calibration"]["max_offdiagonal"])
    print("Induced off-site perturbation:", perturb["max_offsite_amplitude"])
    print("Zero-feedback off-site perturbation:", perturb["zero_feedback_max_offsite"])
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print("Saved:", path)
    return report


def plot_report(report: dict, out: str | None = None, show: bool = False):
    """Plot second-order response, mediated slow perturbations and error baselines."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    J = np.array(report["calibration"]["jacobian"])
    bound = np.max(abs(J))
    im = axs[0].imshow(J, cmap="RdBu_r", vmin=-bound, vmax=bound)
    fig.colorbar(im, ax=axs[0], shrink=.8)
    axs[0].set(title="Measured slow-to-slow Jacobian",
               xlabel="Perturbed edge j", ylabel="Responding edge i")
    impulse = report["impulse"]
    delta = np.array(impulse["traces"]["coupled"]) - np.array(impulse["traces"]["feedback_zero"])
    bound = max(1e-6, np.max(abs(delta)))
    im = axs[1].imshow(delta.T, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-bound, vmax=bound,
                       extent=(0, (len(delta)-1)*impulse["dt_between_samples"], 0, delta.shape[1]))
    fig.colorbar(im, ax=axs[1], shrink=.8)
    axs[1].set(title="Mediated slow response (subtract uncoupled)",
               xlabel="Physical time", ylabel="Edge i")
    errors = report["closure"]["rmse_by_model"]
    axs[2].bar(range(len(errors)), list(errors.values()))
    axs[2].set_xticks(range(len(errors)), list(errors.keys()), rotation=20)
    axs[2].set(title="Held-out slow-only forecast error", ylabel="RMSE (lower is better)")
    if out:
        fig.savefig(out, dpi=145)
        print("Saved:", out)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=120)
    parser.add_argument("--feedback", type=float, default=.93)
    parser.add_argument("--output", default="results/slow_world.json")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.trials < 2 or args.steps < 100 or args.steps % 50:
        parser.error("Require trials>=2, steps>=100 divisible by 50")
    cfg = slow_config(tau=args.tau, feedback=args.feedback)
    report = run_benchmark(args.trials, args.steps, args.output, cfg)
    if args.plot or args.show:
        plot_report(report,
                    out=str(Path(args.output).with_suffix(".png")) if args.plot else None,
                    show=args.show)


if __name__ == "__main__":
    main()
