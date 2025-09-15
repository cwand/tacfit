from tacfit.mc_sampling import _log_prob_uconst, _log_prob_usqrt
from tacfit.mc_sampling import _log_prob_ufrac, _init_walkers
import numpy as np
import unittest


class TestLogProbUConst(unittest.TestCase):

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


class TestLogProbUSqrt(unittest.TestCase):

    def test_log_prob_on_data(self):
        data = np.array([1.0, 2.0, 4.0, 16.0])
        smpl = np.array([1.0, 2.0, 4.0, 16.0])

        sigma = 0.1

        lnp = _log_prob_usqrt(data, smpl, sigma)

        self.assertAlmostEqual(3.108571, lnp, places=6)

    def test_log_prob_tissue_off_data(self):
        data = np.array([1.0, 2.0, 4.0, 16.0])
        smpl = np.array([1.1, 1.9, 3.0, 20.0])

        sigma = 0.1

        lnp = _log_prob_usqrt(data, smpl, sigma)

        self.assertAlmostEqual(-60.141429, lnp, places=6)

    def test_error_on_0_data(self):
        data = np.array([0.0, 2.0, 4.0, 16.0])
        smpl = np.array([0.1, 1.9, 3.0, 20.0])

        sigma = 0.1

        self.assertRaises(ZeroDivisionError,
                          _log_prob_usqrt, data, smpl, sigma)


class TestLogProbUFrac(unittest.TestCase):

    def test_log_prob_on_data(self):
        data = np.array([1.0, 2.0, 4.0, 16.0])
        smpl = np.array([1.0, 2.0, 4.0, 16.0])

        sigma = 0.1

        lnp = _log_prob_ufrac(data, smpl, sigma)

        self.assertAlmostEqual(0.682556, lnp, places=6)

    def test_log_prob_tissue_off_data(self):
        data = np.array([1.0, 2.0, 4.0, 16.0])
        smpl = np.array([1.1, 1.9, 3.0, 20.0])

        sigma = 0.1

        lnp = _log_prob_ufrac(data, smpl, sigma)

        self.assertAlmostEqual(-6.192444, lnp, places=6)

    def test_error_on_0_data(self):
        data = np.array([0.0, 2.0, 4.0, 16.0])
        smpl = np.array([0.1, 1.9, 3.0, 20.0])

        sigma = 0.1

        self.assertRaises(ZeroDivisionError,
                          _log_prob_ufrac, data, smpl, sigma)


class TestInitWalkers(unittest.TestCase):

    def test_1_dim_5_walkers(self):
        start_position = np.array([5.0])
        param_bounds = np.array([[3.0, 10.0],])
        walkers = _init_walkers(start_position, param_bounds, 5)

        self.assertEqual((5, 1), walkers.shape)
        self.assertTrue(np.all(
            np.logical_and(walkers > 4.0, walkers < 6.0)))

    def test_2_dim_4_walkers(self):
        start_position = np.array([0.0, 10.0])
        param_bounds = np.array([[-10.0, 2.0], [-10.0, 100.0]])
        walkers = _init_walkers(start_position, param_bounds, 4)

        self.assertEqual((4, 2), walkers.shape)
        self.assertTrue(np.all(
            np.logical_and(walkers[:, 0] > -1.0, walkers[:, 0] < 1.0)))
        self.assertTrue(np.all(
            np.logical_and(walkers[:, 1] > 0.0, walkers[:, 1] < 20.0)))

    def test_3_dim_50_walkers(self):
        start_position = np.array([1.0, -2.0, 10.0])
        param_bounds = np.array([[-1.0, 5.0], [-10.0, 0.0], [0.0, 15.0]])
        walkers = _init_walkers(start_position, param_bounds, 50)

        self.assertEqual((50, 3), walkers.shape)
        self.assertTrue(np.all(
            np.logical_and(walkers[:, 0] > 0.0, walkers[:, 0] < 2.0)))
        self.assertTrue(np.all(
            np.logical_and(walkers[:, 1] > -3.0, walkers[:, 1] < -1.0)))
        self.assertTrue(np.all(
            np.logical_and(walkers[:, 2] > 7.5, walkers[:, 2] < 12.5)))
