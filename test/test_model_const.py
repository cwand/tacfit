import tacfit.model.const as const
import numpy as np
import unittest


class TestIRFStep2(unittest.TestCase):

    def test_case_1(self):

        amp = 0.5

        t = np.array([0.0, 2.0, 2.9, 3.1, 5.5])

        m = const.irf_const(t, amp=amp)
        m_exp = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.00000001))
