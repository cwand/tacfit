import tacfit.model.stepnorm as stepnorm
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


class TestSplitArrays(unittest.TestCase):

    def test_split_no_problems(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([0.0, 1.0, 1.0, 2.0])
        t = 3.5
        tc = 1.0

        arr1, arr2, arr3, arr4 = stepnorm._split_arrays(ts, fs, t, tc)
        arr1_exp = np.array([0.0, 1.0, 2.0, 2.5])
        arr2_exp = np.array([0.0, 0.0, 1.0, 1.0])
        arr3_exp = np.array([2.5, 3.0, 3.5])
        arr4_exp = np.array([1.0, 1.0, 1.5])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))

    def test_split_tc_larger_than_t(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([0.0, 1.0, 1.0, 2.0])
        t = 2.5
        tc = 3.0

        arr1, arr2, arr3, arr4 = stepnorm._split_arrays(ts, fs, t, tc)
        arr1_exp = np.array([0.0, 0.0])
        arr2_exp = np.array([0.0, 0.0])
        arr3_exp = np.array([0.0, 1.0, 2.0, 2.5])
        arr4_exp = np.array([0.0, 0.0, 1.0, 1.0])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))

    def test_split_t_smaller_than_first_time(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([0.0, 1.0, 1.0, 2.0])
        t = 0.5
        tc = 1.0

        arr1, arr2, arr3, arr4 = stepnorm._split_arrays(ts, fs, t, tc)
        arr1_exp = np.array([0.0, 0.0])
        arr2_exp = np.array([0.0, 0.0])
        arr3_exp = np.array([0.0, 0.5])
        arr4_exp = np.array([0.0, 0.0])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))

    def test_split_interp_left(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([1.0, 2.0, 1.0, 2.0])
        t = 1.5
        tc = 1.0

        arr1, arr2, arr3, arr4 = stepnorm._split_arrays(ts, fs, t, tc)
        arr1_exp = np.array([0.0, 0.5])
        arr2_exp = np.array([0.0, 0.0])
        arr3_exp = np.array([0.5, 1.0, 1.5])
        arr4_exp = np.array([0.0, 1.0, 1.5])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))

    def test_split_interp_left_all(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([1.0, 2.0, 1.0, 2.0])
        t = 0.5
        tc = 1.0

        arr1, arr2, arr3, arr4 = stepnorm._split_arrays(ts, fs, t, tc)
        arr1_exp = np.array([0.0, 0.0])
        arr2_exp = np.array([0.0, 0.0])
        arr3_exp = np.array([0.0, 0.5])
        arr4_exp = np.array([0.0, 0.0])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))


class TestModelNormconst(unittest.TestCase):

    def test_model_normconst_case1(self):
        amp1 = 0.3
        amp2 = 0.1
        extent1 = 4.0
        extent2 = 20
        width2 = 3.0

        tin = np.array([0.0, 2.5,  5.0,  7.5,  10.0, 15.0, 20.0, 30.0, 40.0])
        in_fnc = np.array([0.0, 10.0, 50.0, 30.0, 10.0, 8.0,  6.0,  5.0,  3.0])

        t_out = np.array([0.0, 1.0, 5.0, 9.0, 12.0, 15.0,
                          18.0, 21.0, 25.0, 28.0, 31.0])

        m = stepnorm.model_stepnorm(amp1=amp1, extent1=extent1, width2=width2,
                                    amp2=amp2, extent2=extent2,
                                    t_in=tin, in_func=in_fnc,
                                    t_out=t_out)

        m_exp = np.array([0.0, 0.6, 25.85, 49.54, 36.71, 35.26,
                          36.18, 35.25, 29.19, 22.72, 18.18])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.5))
