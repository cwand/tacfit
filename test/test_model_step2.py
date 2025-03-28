import tacfit.model.step2 as step2
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

class TestSplitArrays(unittest.TestCase):

    def test_split_no_problems(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([0.0, 1.0, 1.0, 2.0])
        t = 3.5
        tc1 = 2.0
        tc2 = 1.0

        arr1, arr2, arr3, arr4 = step2._split_arrays(ts, fs, t, tc1, tc2)
        arr1_exp = np.array([1.5, 2.0, 2.5])
        arr2_exp = np.array([0.5, 1.0, 1.0])
        arr3_exp = np.array([2.5, 3.0, 3.5])
        arr4_exp = np.array([1.0, 1.0, 1.5])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))

    def test_split_tc1_and_tc2_larger_than_t(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([0.0, 1.0, 1.0, 2.0])
        t = 2.5
        tc1 = 4.0
        tc2 = 3.0

        arr1, arr2, arr3, arr4 = step2._split_arrays(ts, fs, t, tc1, tc2)
        arr1_exp = np.array([0.0, 0.0])
        arr2_exp = np.array([0.0, 0.0])
        arr3_exp = np.array([0.0, 1.0, 2.0, 2.5])
        arr4_exp = np.array([0.0, 0.0, 1.0, 1.0])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))

    def test_split_tc1_larger_than_t(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([0.0, 1.0, 1.0, 2.0])
        t = 2.5
        tc1 = 3.0
        tc2 = 2.0

        arr1, arr2, arr3, arr4 = step2._split_arrays(ts, fs, t, tc1, tc2)
        arr1_exp = np.array([0.0, 0.5])
        arr2_exp = np.array([0.0, 0.0])
        arr3_exp = np.array([0.5, 1.0, 2.0, 2.5])
        arr4_exp = np.array([0.0, 0.0, 1.0, 1.0])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))

    def test_split_t_smaller_than_first_time(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([0.0, 1.0, 1.0, 2.0])
        t = 0.5
        tc1 = 2.0
        tc2 = 1.0

        arr1, arr2, arr3, arr4 = step2._split_arrays(ts, fs, t, tc1, tc2)
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
        tc1 = 1.0
        tc2 = 0.5

        arr1, arr2, arr3, arr4 = step2._split_arrays(ts, fs, t, tc1, tc2)
        arr1_exp = np.array([0.5, 1.0])
        arr2_exp = np.array([0.0, 1.0])
        arr3_exp = np.array([1.0, 1.5])
        arr4_exp = np.array([1.0, 1.5])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))

    def test_split_interp_left_all(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([1.0, 2.0, 1.0, 2.0])
        t = 0.5
        tc1 = 2.0
        tc2 = 1.0

        arr1, arr2, arr3, arr4 = step2._split_arrays(ts, fs, t, tc1, tc2)
        arr1_exp = np.array([0.0, 0.0])
        arr2_exp = np.array([0.0, 0.0])
        arr3_exp = np.array([0.0, 0.5])
        arr4_exp = np.array([0.0, 0.0])

        self.assertTrue(np.all(np.abs(arr1 - arr1_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr2 - arr2_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr3 - arr3_exp) < 0.0000001))
        self.assertTrue(np.all(np.abs(arr4 - arr4_exp) < 0.0000001))


class TestModelStep2(unittest.TestCase):

    def test_model_step2_case1(self):
        amp = 0.1
        amp2 = 0.3
        extent = 3.0
        extent2 = 6.0

        tp = np.array([0.0, 3.7, 7.1, 10.2, 13.5, 17.8])
        in_func = np.array([0.0, 572.1, 3021.5, 123.7, 50.21, 10.5])

        m = step2.model_step2(amp1=amp, extent1=extent,
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

        m = step2.model_step2(amp1=amp, extent1=extent,
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

        m = step2.model_step2(amp1=amp, extent1=extent,
                              amp2=amp2, extent2=extent2,
                              t_in=tp, in_func=in_func,
                              t_out=t_out)

        m_exp = np.array([1340.227])

        self.assertTrue(np.all(np.abs(m - m_exp) < 1.0))
