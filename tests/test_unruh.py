"""Run: python -m unittest discover -s tests -v"""
import unittest
from dataclasses import replace
import numpy as np
from unruh import Config, World, evaluate, write_then_probe


class WorldTests(unittest.TestCase):
    def test_operator_matches_fast_force(self):
        w = World(Config(n=20, cubic=0, damping=0))
        w.x[:] = np.linspace(-0.4, 0.7, w.cfg.n)
        w.h[:] = np.linspace(-0.5, 0.5, w.cfg.n)
        expected = w.operator() @ w.x
        weights = w.edges()
        flux = weights * (np.roll(w.x, -1) - w.x)
        actual = flux - np.roll(flux, 1) - w.cfg.onsite*w.x
        np.testing.assert_allclose(expected, actual, atol=1e-12)
        np.testing.assert_allclose(w.operator(), w.operator().T)

    def test_bounded_operator_and_stability(self):
        w = World(Config(n=24), seed=4)
        force = np.zeros(w.cfg.n)
        for t in range(600):
            force.fill(0)
            force[3] = 0.8*np.sin(0.12*t)
            w.step(force, noise=0.001)
        self.assertTrue(np.all(np.isfinite(w.x)))
        self.assertLessEqual(np.max(np.abs(w.h)), 1.0)
        self.assertGreater(np.min(w.edges()), 0)
        self.assertGreaterEqual(w.field_energy(), 0)

    def test_fast_erase_keeps_only_true_slow_memory(self):
        for mode in ("coupled", "frozen", "instant", "observer"):
            w = World(mode=mode)
            for _ in range(40):
                w.step(np.eye(1, w.cfg.n, 5).ravel())
            old_h = w.h.copy()
            w.reset_fast()
            self.assertEqual(np.linalg.norm(w.x), 0)
            self.assertEqual(np.linalg.norm(w.v), 0)
            if mode == "instant":
                self.assertEqual(np.linalg.norm(w.h), 0)
            else:
                np.testing.assert_array_equal(w.h, old_h)

    def test_no_feedback_or_observer_cannot_transmit_memory(self):
        for mode, cfg in (("frozen", Config()), ("instant", Config()),
                          ("observer", Config()), ("coupled", Config(feedback=0))):
            left, _ = write_then_probe(mode, 0, 202, cfg)
            right, _ = write_then_probe(mode, 1, 202, cfg)
            np.testing.assert_array_equal(left, right)

    def test_slow_state_causal_transplant(self):
        # Same x,v=0 and probe randomness; swapping h must swap the full trace.
        cfg = Config()
        _, _, left = write_then_probe("coupled", 0, 4, cfg, return_world=True)
        _, _, right = write_then_probe("coupled", 1, 4, cfg, return_world=True)
        # The returned world is after probing. Instead compare new worlds with
        # the two stored h states, same x,v and probe RNG.
        def replay(h):
            w = World(cfg, "coupled", seed=123)
            w.h[:] = h
            f = np.zeros(cfg.n)
            out = []
            for t in range(100):
                f.fill(0)
                f[0] = 0.65*np.sin(1.1*cfg.dt*t)
                w.step(f, noise=0.008)
                out.append(w.x[[7, 12, 19]].copy())
            return np.array(out)
        a, b = replay(left.h), replay(right.h)
        self.assertGreater(np.max(np.abs(a-b)), 1e-9)
        np.testing.assert_array_equal(replay(left.h.copy()), a)
        self.assertFalse(np.array_equal(replay(np.zeros(cfg.n)), a))

    def test_held_out_decoder(self):
        full = evaluate("coupled", seed=0, train_pairs=10, test_pairs=10)
        frozen = evaluate("frozen", seed=0, train_pairs=10, test_pairs=10)
        observer = evaluate("observer", seed=0, train_pairs=10, test_pairs=10)
        instant = evaluate("instant", seed=0, train_pairs=10, test_pairs=10)
        self.assertGreaterEqual(full["accuracy"], 0.75)
        for control in (frozen, observer, instant):
            self.assertEqual(control["accuracy"], 0.5)
            self.assertEqual(control["template_distance"], 0)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            Config(feedback=1.2)
        with self.assertRaises(ValueError):
            Config(tau=0.1)


if __name__ == "__main__":
    unittest.main()
