import tacfit.model.stepnorm as stepnorm
import tacfit.model.integrate as integrate
import numpy as np
import unittest


class TestIRFStepNorm(unittest.TestCase):

    def test_case_1(self):

        amp1 = 0.5
        amp2 = 0.1
        extent1 = 3.0
        extent2 = 10.0
        wid2 = 1.0

        t = np.array([0.0, 2.9, 3.1, 9.7, 11.0])

        m = stepnorm.irf_stepnorm(t, amp1=amp1, amp2=amp2, extent1=extent1,
                                  extent2=extent2, width2=wid2)
        m_exp = np.array([0.500000000000000,
                          0.499999999999938,
                          0.099999999999740,
                          0.061791142218895,
                          0.015865525393146])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.0000001))


class TestModelStepnorm(unittest.TestCase):

    def test_model_stepnorm_case1(self):
        amp1 = 0.3
        amp2 = 0.1
        extent1 = 4.0
        extent2 = 20
        width2 = 3.0

        tin = np.array([0.0, 2.5,  5.0,  7.5,  10.0, 15.0, 20.0, 30.0, 40.0])
        in_fnc = np.array([0.0, 10.0, 50.0, 30.0, 10.0, 8.0,  6.0,  5.0,  3.0])

        t_out = np.array([0.0, 1.0, 5.0, 9.0, 12.0, 15.0,
                          18.0, 21.0, 25.0, 28.0, 31.0])

        m = integrate.model(tin, in_fnc, t_out,
                            stepnorm.irf_stepnorm,  # type: ignore
                            amp1=amp1, amp2=amp2, extent1=extent1,
                            extent2=extent2, width2=width2)

        m_exp = np.array([0.0,
                          0.599999999993591,
                          25.849988266065488,
                          49.549943215866591,
                          36.708635516964897,
                          35.257596102595855,
                          36.175818939187614,
                          35.249291410756392,
                          29.193831488608613,
                          22.718937741431429,
                          18.175410261905824])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.01))
