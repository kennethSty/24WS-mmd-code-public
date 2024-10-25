import numpy as np
import unittest
from rec_sys import exercise2 as ex
from rec_sys import cf_algorithms_to_complete as cfa
from scipy.sparse import csr_matrix

class test_sparse_cosine_sim(unittest.TestCase):
    def test_centered_cosine_sim(self):
        vector_x = csr_matrix([i + 1 for i in range(0, 100)])
        vector_y = csr_matrix([100 - j for j in range(0, 100)])
        centered_cos_sim = ex.centered_cosine_sim(vector_x, vector_y)
        similarity_score = centered_cos_sim.data[0]
        self.assertAlmostEqual(similarity_score, -1.0)

    def test_centered_cosine_sim2(self):
        c1 = [2, 3, 4, 5, 6]
        c2 = np.linspace(0, 90, 10)
        c = set([i + j for i in c1 for j in c2])
        list_x = [np.nan if i in c else i + 1 for i in range(0, 100)]
        list_y = [100 - x for x in list_x]
        centered_cos_sim = ex.centered_cosine_sim(csr_matrix(list_x), csr_matrix(list_y))
        similarity_score = centered_cos_sim.data[0]
        self.assertAlmostEqual(similarity_score, -1.0)

    def test_fast_centered_cosine_sim(self):
        matrix = csr_matrix(np.array(([1, 1, 0], [1, np.nan, 3], [0, 2, 0])))
        vector = csr_matrix(np.array([-1, -1, 0]).reshape(3, 1))
        sparse_cleaned_matrix = ex.sparse_center_and_nan_to_zero(matrix)
        sparse_cleaned_vector = ex.sparse_center_and_nan_to_zero(vector)
        normal_cleaned_matrix = cfa.center_and_nan_to_zero(matrix.toarray())
        normal_cleaned_vector = cfa.center_and_nan_to_zero(vector.toarray())
        sim = ex.fast_centered_cosine_sim(sparse_cleaned_matrix, sparse_cleaned_vector).toarray().flatten()
        expected = cfa.fast_cosine_sim(normal_cleaned_matrix,
                                       normal_cleaned_vector).flatten()
        self.assertAlmostEqual(sim[0], expected[0])
        self.assertAlmostEqual(sim[1], expected[1])
        self.assertAlmostEqual(sim[2], expected[2])

#only execute unittests if this file specifically is run.
if __name__ == '__main__':
    unittest.main() #looks for tesetcases and runs them