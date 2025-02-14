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
