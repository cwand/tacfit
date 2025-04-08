import tacfit.model.stepconst as stepconst
import tacfit.model.integrate as integrate
import numpy as np
import unittest


class TestIRFStep2(unittest.TestCase):

    def test_case_1(self):

        amp1 = 0.5
        amp2 = 0.1
        extent1 = 3.0

        t = np.array([0.0, 2.0, 2.9, 3.1, 5.5])

        m = stepconst.irf_stepconst(t, amp1=amp1, amp2=amp2, extent1=extent1)
        m_exp = np.array([0.5, 0.5, 0.5, 0.1, 0.1])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.00000001))


class TestModelStepconst(unittest.TestCase):

    def test_model_stepconst_case1(self):
        amp = 0.3
        amp2 = 0.1
        extent = 6.0

        tin = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        in_func = np.array([0.0, 10.0, 50.0, 30.0, 10.0])

        t_out = np.array([0.0, 1.0, 5.0, 9.0])

        m = integrate.model(tin, in_func, t_out, stepconst.irf_stepconst,
                            amp1=amp, amp2=amp2, extent1=extent)

        m_exp = np.array([0.0, 0.6, 26.25, 63.150])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.01))
