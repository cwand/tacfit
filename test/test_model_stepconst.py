import tacfit.model.stepconst as stepconst
import numpy as np
import unittest


class TestSplitArrays(unittest.TestCase):

    def test_split_no_problems(self):
        ts = np.array([1.0, 2.0, 3.0, 4.0])
        fs = np.array([0.0, 1.0, 1.0, 2.0])
        t = 3.5
        tc = 1.0

        arr1, arr2, arr3, arr4 = stepconst._split_arrays(ts, fs, t, tc)
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

        arr1, arr2, arr3, arr4 = stepconst._split_arrays(ts, fs, t, tc)
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

        arr1, arr2, arr3, arr4 = stepconst._split_arrays(ts, fs, t, tc)
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

        arr1, arr2, arr3, arr4 = stepconst._split_arrays(ts, fs, t, tc)
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

        arr1, arr2, arr3, arr4 = stepconst._split_arrays(ts, fs, t, tc)
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
