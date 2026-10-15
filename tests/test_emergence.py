"""Unruh V2 regression tests: effective law, causal transfer, negative controls."""
import unittest
from dataclasses import replace
import numpy as np
from emergence import (slow_config, constant_drive, equilibrate_fast,
                       correlation_target, calibrate, forecast,
                       closure_gate, perturbation_gate)
from unruh import World


class EmergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = slow_config()
        cls.F = constant_drive(cls.c.n)
        cls.fit = calibrate(cls.c, cls.F)

    def test_fast_force_balance(self):
        x = equilibrate_fast(self.fit["reference"], self.c, self.F)
        w = World(self.c)
        w.h[:] = self.fit["reference"]
        residual = w.operator()@x - self.c.cubic*x**3 + self.F
        self.assertLess(np.max(np.abs(residual)), 1e-8)

    def test_driven_slow_fixed_point(self):
        target = correlation_target(self.fit["reference"], self.c, self.F)
        self.assertLess(np.max(np.abs(target-self.fit["reference"])), 1e-7)

    def test_feedback_required_for_cross_edge_jacobian(self):
        J = self.fit["jacobian"]
        off = J - np.diag(np.diag(J))
        self.assertGreater(np.max(np.abs(off)), .1)
        zero_fit = calibrate(replace(self.c, feedback=0.), self.F)
        np.testing.assert_allclose(zero_fit["jacobian"], 0, atol=1e-10)

    def test_without_pump_no_cross_edge_jacobian(self):
        fit = calibrate(self.c, np.zeros(self.c.n))
        np.testing.assert_allclose(fit["reference"], 0, atol=1e-9)
        np.testing.assert_allclose(fit["jacobian"], 0, atol=1e-9)

    def test_full_slow_law_outperforms_local_baselines(self):
        outcome = closure_gate(self.c, self.F, self.fit, trials=6, steps=600, stride=50)
        e = outcome["rmse_by_model"]
        self.assertLess(e["full"], e["nearest"])
        self.assertLess(e["full"], .85*e["diagonal"])
        self.assertLess(e["full"], e["decay"])

    def test_slow_impulse_causally_reaches_distant_edges(self):
        outcome = perturbation_gate(self.c, self.F, self.fit, steps=250, stride=25)
        self.assertGreater(outcome["max_offsite_amplitude"], .0001)
        self.assertEqual(outcome["zero_feedback_max_offsite"], 0.)

    def test_forecast_uses_h_without_mutating_initial_state(self):
        h = self.fit["reference"].copy()
        h[7] += .1
        backup = h.copy()
        for model in ("full", "nearest", "diagonal", "decay", "persistence"):
            pred = forecast(h, self.fit, self.c, steps=100, model=model, stride=10)
            self.assertEqual(pred.shape, (11, self.c.n))
            np.testing.assert_allclose(pred[0], h)
        np.testing.assert_allclose(h, backup)


if __name__ == "__main__":
    unittest.main()
