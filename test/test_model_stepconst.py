import tacfit.model.stepconst as stepconst
import numpy as np
import unittest


class TestModelStep2(unittest.TestCase):

    def test_model_step2_case1(self):
        amp = 0.3
        amp2 = 0.1
        extent = 6.0

        tin = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        in_func = np.array([0.0, 10.0, 50.0, 30.0, 10.0])

        t_out = np.array([0.0, 1.0, 5.0, 9.0])

        m = stepconst.model_stepconst(amp1=amp, extent1=extent,
                                      amp2=amp2,
                                      t_in=tin, in_func=in_func,
                                      t_out=t_out)

        m_exp = np.array([0.0, 0.6, 26.25, 63.150])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.01))
