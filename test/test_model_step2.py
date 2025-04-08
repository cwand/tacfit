import tacfit.model.step2 as step2
import tacfit.model.integrate as integrate
import numpy as np
import unittest


class TestIRFStep2(unittest.TestCase):

    def test_case_1(self):

        amp1 = 0.5
        amp2 = 0.1
        extent1 = 3.0
        extent2 = 5.0

        t = np.array([0.0, 2.0, 3.1, 4.0, 5.5])

        m = step2.irf_step2(t, amp1=amp1, amp2=amp2, extent1=extent1,
                            extent2=extent2)
        m_exp = np.array([0.5, 0.5, 0.1, 0.1, 0.0])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.00000001))


class TestModelStep2(unittest.TestCase):

    def test_model_step2_case1(self):
        amp = 0.3
        amp2 = 0.1
        extent = 3.0
        extent2 = 6.0

        tp = np.array([0.0, 3.7, 7.1, 10.2, 13.5, 17.8])
        in_func = np.array([0.0, 572.1, 3021.5, 123.7, 50.21, 10.5])

        m = integrate.model(tp, in_func, tp.copy(), step2.irf_step2,
                            amp1=amp, amp2=amp2, extent1=extent,
                            extent2=extent2)

        m_exp = np.array([0.0,
                          3.099390702417348e+02,
                          1.871925533051459e+03,
                          1.976325476685502e+03,
                          4.529879150757237e+02,
                          39.417721301572087])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.001))

    def test_model_step2_t_out_single_point(self):
        amp = 0.1
        amp2 = 0.3
        extent = 3.0
        extent2 = 6.0

        tp = np.array([0.0, 3.7, 7.1, 10.2, 13.5, 17.8])
        in_func = np.array([0.0, 572.1, 3021.5, 123.7, 50.21, 10.5])

        t_out = np.array([3.7])

        m = integrate.model(tp, in_func, t_out, step2.irf_step2,
                            amp1=amp, amp2=amp2, extent1=extent,
                            extent2=extent2)

        m_exp = np.array([113.415])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.001))

    def test_model_step2_t_out_single_point_not_in_input_sample(self):
        amp = 0.1
        amp2 = 0.3
        extent = 3.0
        extent2 = 6.0

        tp = np.array([0.0, 3.7, 7.1, 10.2, 13.5, 17.8])
        in_func = np.array([0.0, 572.1, 3021.5, 123.7, 50.21, 10.5])

        t_out = np.array([8.0])

        m = integrate.model(tp, in_func, t_out, step2.irf_step2,
                            amp1=amp, amp2=amp2, extent1=extent,
                            extent2=extent2)

        m_exp = np.array([1340.227])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.001))
