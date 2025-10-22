import unittest
import os
import numpy as np
import tacfit.core


class TestSaveLoadDict(unittest.TestCase):

    def test_save_load_table(self):
        tac = tacfit.core.load_table(
            os.path.join('test', 'data', 'ex_tac.txt'))

        tacq_exp = np.array([0.0, 1.0, 2.0])
        self.assertFalse(np.any(tac['tacq'] - tacq_exp))

        a_exp = np.array([1.0, 3.0, 3.5])
        self.assertFalse(np.any(tac['a'] - a_exp))

        b_exp = np.array([-1.3, 0.1, -2.0])
        self.assertFalse(np.any(tac['b'] - b_exp))


class TestCorrectedInputFunction(unittest.TestCase):

    def test_trivial(self):
        ot = np.array([0.0, 1.5, 3.0, 6.0,  9.0, 18.0])
        od = np.array([0.0, 1.0, 5.0, 7.0, 10.0,  5.0])

        ct, cd = tacfit.create_corrected_input_function(ot, od)

        self.assertTrue(np.all((ct - ot) == 0.0))
        self.assertTrue(np.all((cd - od) == 0.0))

    def test_copy_change_original(self):
        ot = np.array([0.0, 1.5, 3.0, 6.0,  9.0, 18.0])
        od = np.array([0.0, 1.0, 5.0, 7.0, 10.0,  5.0])

        ct, cd = tacfit.create_corrected_input_function(ot, od)

        ot[0] = 1.0
        od[0] = 0.5

        self.assertEqual(ct[0], 0.0)
        self.assertEqual(cd[0], 0.0)

    def test_copy_change_corrected(self):
        ot = np.array([0.0, 1.5, 3.0, 6.0,  9.0, 18.0])
        od = np.array([0.0, 1.0, 5.0, 7.0, 10.0,  5.0])

        ct, cd = tacfit.create_corrected_input_function(ot, od)

        ct[0] = 1.0
        cd[0] = 0.5

        self.assertEqual(ot[0], 0.0)
        self.assertEqual(od[0], 0.0)

    def test_delay(self):
        ot = np.array([0.0, 1.5, 3.0, 6.0,  9.0, 18.0])
        od = np.array([0.0, 1.0, 5.0, 7.0, 10.0,  5.0])

        ct, cd = tacfit.create_corrected_input_function(ot, od, delay=1.2)

        ct_exp = np.array([1.2, 2.7, 4.2, 7.2, 10.2, 19.2])
        cd_exp = np.array([0.0, 1.0, 5.0, 7.0, 10.0,  5.0])

        self.assertTrue(np.all((ct - ct_exp) == 0.0))
        self.assertTrue(np.all((cd - cd_exp) == 0.0))


class TestCalcWeights(unittest.TestCase):

    def test_equal_weights(self):
        duration = np.array([3.0, 3.0, 6.0, 9.0])
        tac = np.array([100, 100, 200, 300])

        weights = tacfit.calc_weights(duration, tac)
        self.assertAlmostEqual(float(weights[0]), 0.173205, places=6)
        self.assertAlmostEqual(float(weights[1]), 0.173205, places=6)
        self.assertAlmostEqual(float(weights[2]), 0.173205, places=6)
        self.assertAlmostEqual(float(weights[3]), 0.173205, places=6)

    def test_unequal_weights(self):
        duration = np.array([3.0, 3.0, 6.0, 9.0])
        tac = np.array([100, 150, 350, 500])

        weights = tacfit.calc_weights(duration, tac)
        self.assertAlmostEqual(float(weights[0]), 0.173205, places=6)
        self.assertAlmostEqual(float(weights[1]), 0.141421, places=6)
        self.assertAlmostEqual(float(weights[2]), 0.130931, places=6)
        self.assertAlmostEqual(float(weights[3]), 0.134164, places=6)
