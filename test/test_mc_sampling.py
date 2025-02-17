from tacfit.mc_sampling import _log_prob_tissue
import numpy as np
import unittest


class TestLogProb(unittest.TestCase):

    def test_log_prob_tissue_on_data(self):
        tissue_data = np.array([1.0, 2.0, 3.0, 2.0])
        tissue_smpl = np.array([1.0, 2.0, 3.0, 2.0])

        lnsigma_tissue = -4.0

        lnp = _log_prob_tissue(tissue_data, tissue_smpl, lnsigma_tissue)

        self.assertAlmostEqual(9.839339, lnp, places=6)

    def test_log_prob_tissue_off_data(self):
        tissue_data = np.array([1.0, 2.0, 3.0, 2.0])
        tissue_smpl = np.array([1.1, 2.1, 2.9, 2.1])

        lnsigma_tissue = -4.0

        lnp = _log_prob_tissue(tissue_data, tissue_smpl, lnsigma_tissue)

        self.assertAlmostEqual(-14.173933, lnp, places=6)
