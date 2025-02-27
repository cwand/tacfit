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

        m_exp = np.array([0.0, 113.415, 957.658,
                          2266.907, 1158.889, 59.808])

        self.assertTrue(np.all(np.abs(m - m_exp) < 1.0))

    def test_model_step2_t_out_single_point(self):
        amp = 0.1
        amp2 = 0.3
        extent = 3.0
        extent2 = 6.0

        tp = np.array([0.0, 3.7, 7.1, 10.2, 13.5, 17.8])
        in_func = np.array([0.0, 572.1, 3021.5, 123.7, 50.21, 10.5])

        t_out = np.array([3.7])

        m = tacfit.model.step2.model_step2(amp1=amp, extent1=extent,
                                           amp2=amp2, extent2=extent2,
                                           t_in=tp, in_func=in_func,
                                           t_out=t_out)

        m_exp = np.array([113.415])

        self.assertTrue(np.all(np.abs(m - m_exp) < 1.0))

    def test_model_step2_t_out_single_point_not_in_input_sample(self):
        amp = 0.1
        amp2 = 0.3
        extent = 3.0
        extent2 = 6.0

        tp = np.array([0.0, 3.7, 7.1, 10.2, 13.5, 17.8])
        in_func = np.array([0.0, 572.1, 3021.5, 123.7, 50.21, 10.5])

        t_out = np.array([8.0])

        m = tacfit.model.step2.model_step2(amp1=amp, extent1=extent,
                                           amp2=amp2, extent2=extent2,
                                           t_in=tp, in_func=in_func,
                                           t_out=t_out)

        m_exp = np.array([1340.227])

        self.assertTrue(np.all(np.abs(m - m_exp) < 1.0))
