import tacfit.model.step2
import numpy as np
import unittest


class TestModelStep2(unittest.TestCase):

    def test_model_step2_case1(self):
        amp = 0.1
        amp2 = 0.3
        extent = 3.0
        extent2 = 6.0

        tp = np.array([0.0, 3.7, 7.1, 10.2, 13.5, 17.8])
        in_func = np.array([0.0, 572.1, 3021.5, 123.7, 50.21, 10.5])

        m = tacfit.model.step2.model_step2(amp1=amp, extent1=extent,
                                           amp2=amp2, extent2=extent2,
                                           t_in=tp, in_func=in_func,
                                           t_out=tp.copy())

        m_exp = np.array([0.0, 419.5657, 2704.4526,
                          3640.1826, 1233.5420, 81.7247])

        self.assertTrue(np.all((m - m_exp) < 1e-2))
