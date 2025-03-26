import tacfit.model.normconst as normconst
import numpy as np
import unittest


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

        print(m)

        m_exp = np.array([0.0, 0.6, 25.43, 49.15])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.5))
