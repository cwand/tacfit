from tacfit.mc_sampling import _log_prob_uconst
import numpy as np
import unittest


class TestLogProb(unittest.TestCase):

    def test_log_prob_on_data(self):
        data = np.array([1.0, 2.0, 3.0, 2.0])
        smpl = np.array([1.0, 2.0, 3.0, 2.0])

        sigma = 1.0

        lnp = _log_prob_uconst(data, smpl, sigma)

        self.assertAlmostEqual(-3.675754, lnp, places=6)

    def test_log_prob_tissue_off_data(self):
        data = np.array([1.0, 2.0, 3.0, 2.0])
        smpl = np.array([1.1, 2.1, 2.9, 2.1])

        sigma = 0.1

        lnp = _log_prob_uconst(data, smpl, sigma)

        self.assertAlmostEqual(3.534586, lnp, places=6)
