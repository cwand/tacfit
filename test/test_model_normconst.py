import tacfit.model.normconst as normconst
import numpy as np
import unittest


class TestIRFNormConst(unittest.TestCase):

    def test_case_1(self):

        amp1 = 0.5
        amp2 = 0.1
        extent1 = 3.0
        wid1 = 0.5

        t = np.array([0.0, 2.0, 3.0, 4.0, 10.0])

        m = normconst.irf_normconst(t, amp1=amp1, amp2=amp2, extent1=extent1,
                                    width1=wid1)
        m_exp = np.array([0.499999999605365,
                          0.490899947220728,
                          0.300000000000000,
                          0.109100052779272,
                          0.100000000000000])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.00000001))


class TestModelNormconst(unittest.TestCase):

    def test_model_normconst_case1(self):
        amp = 0.3
        amp2 = 0.1
        extent = 4.0
        width = 1.0

        tin = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        in_func = np.array([0.0, 10.0, 50.0, 30.0, 10.0])

        t_out = np.array([0.0, 1.0, 5.0, 9.0])

        m = normconst.model_normconst(amp1=amp, extent1=extent, width1=width,
                                      amp2=amp2,
                                      t_in=tin, in_func=in_func,
                                      t_out=t_out)

        m_exp = np.array([0.0, 0.6, 25.43, 49.15])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.5))
