import os
import unittest
import tacfit
import numpy as np


class TestLoadDict(unittest.TestCase):

    def test_save_load_table(self):
        tac = tacfit.load_table(os.path.join('test', 'data', 'ex_tac.txt'))

        tac_exp = np.array([0.0, 1.0, 2.0])
        self.assertFalse(np.any(tac['tacq'] - tac_exp))

        a_exp = np.array([1.0, 3.0, 3.5])
        self.assertFalse(np.any(tac['A'] - a_exp))

        two_exp = np.array([-1.3, 0.1, -2.0])
        self.assertFalse(np.any(tac['2'] - two_exp))
