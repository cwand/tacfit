import tacfit.model.norm2 as norm2
import numpy as np
import unittest


class TestIRFNorm2(unittest.TestCase):

    def test_case_1(self):

        amp1 = 0.5
        amp2 = 0.1
        extent1 = 3.0
        wid1 = 0.5
        extent2 = 10.0
        wid2 = 1.0

        t = np.array([0.0, 2.9, 3.1, 5.0, 9.0, 13.0, 15.0, 16.0, 19.0])

        m = norm2.irf_norm2(t, amp1=amp1, amp2=amp2, extent1=extent1,
                            width1=wid1, extent2=extent2, width2=wid2)

        print(m)

        m_exp = np.array([0.499999999605365,
                          0.331703883775579,
                          0.268296116224099,
                          0.100012639831576,
                          0.084134474606854,
                          0.000134989803163,
                          0.000000028665157,
                          0.000000000098659,
                          0])

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.0000000001))


class TestModelNorm2(unittest.TestCase):

    def test_model_norm2_case1(self):
        amp1 = 0.5
        amp2 = 0.1
        extent1 = 3.0
        wid1 = 0.5
        extent2 = 10.0
        wid2 = 1.0

        tin = np.array([0.0, 2.5,  5.0,  7.5,  10.0, 12.5, 15.0, 17.5, 20.0])
        in_func = np.array([0.0, 10.0, 50.0, 30.0, 10.0, 8.0,  6.0,  5.0,  3.0])

        t_out = np.array([0.0, 2.9, 3.1, 5.0, 9.0, 13.0, 15.0, 16.0, 19.0])

        m = norm2.model_norm2(amp1=amp1, extent1=extent1, width1=wid1,
                              amp2=amp2, extent2=extent2, width2=wid2,
                              t_in=tin, in_func=in_func,
                              t_out=t_out)

        m_exp = np.array([0.0,
                          8.818130754879661,
                          10.553878183719061,
                          40.304796745769025,
                          58.727997717576024,
                          34.552743504689403,
                          27.480718253032361,
                          23.089609674874218,
                          13.647991536815924])

        print(m)

        self.assertTrue(np.all(np.abs(m - m_exp) < 0.0001))
