import numpy as np
import unittest
from rec_sys import exercise2 as ex
from rec_sys import cf_algorithms_to_complete as cfa

class test_sparse_cosine_sim(unittest.TestCase):
    def test_centered_cosine_sim(self):
        vector_x = np.array([i + 1 for i in range(0, 100)])
        vector_y = np.array([100 - j for j in range(0, 100)])
        centered_cos_sim = ex.centered_cosine_sim(vector_x, vector_y)
        self.assertAlmostEqual(centered_cos_sim, -1.0)

    def test_centered_cosine_sim2(self):
        c1 = [2, 3, 4, 5, 6]
        c2 = np.linspace(0, 90, 10)
        c = set([i + j for i in c1 for j in c2])
        list_x = [np.nan if i in c else i + 1 for i in range(0, 100)]
        list_y = [100 - x for x in list_x]
        centered_cos_sim = ex.centered_cosine_sim(np.array(list_x), np.array(list_y))
        self.assertAlmostEqual(centered_cos_sim, -1.0)

    def test_fast_centered_cosine_sim(self):
        matrix = np.array(([1, 1, 0], [1, np.nan, 3], [0, 2, 0]))
        vector = np.array([-1, -1, 0])
        sim = ex.fast_centered_cosine_sim(matrix, vector).toarray().flatten()
        expected = cfa.fast_cosine_sim(cfa.center_and_nan_to_zero(matrix), cfa.center_and_nan_to_zero(vector))
        self.assertAlmostEqual(sim[0], expected[0])
        self.assertAlmostEqual(sim[1], expected[1])
        self.assertAlmostEqual(sim[2], expected[2])

#only execute unittests if this file specifically is run.
if __name__ == '__main__':
    unittest.main() #looks for tesetcases and runs them