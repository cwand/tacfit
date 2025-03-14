import tacfit.model.delay as mdelay
import numpy as np
import unittest


class TestModelDelay(unittest.TestCase):

    def test_model_delay_case1(self):
        k = 0.3
        delay = 2.0

        tin = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        in_func = np.array([0.0, 10.0, 50.0, 30.0, 10.0])

        t_out = np.array([1.0, 5.0, 9.0])

        m = mdelay.model_delay(k=k, delay=delay,
                               t_in=tin, in_func=in_func,
                               t_out=t_out)

        m_exp = np.array([0.0, 5.85, 51.45])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.01))
