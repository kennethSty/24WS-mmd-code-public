import unittest
from rec_sys import cf_config
from rec_sys import sparse_cf_algorithms_to_complete as scfa
from rec_sys import cf_algorithms_to_complete as cfa
from rec_sys import cf_data as cfd
class test_exercise3(unittest.TestCase):

    def test_rate_all_items(self):
        um_sparse = cfd.get_sparse_um_by_name(cf_config ,"sparse_test")
        true_ratings = cfa.rate_all_items(um_sparse.toarray(), 0, 4)
        computed_ratings = scfa.rate_all_items(um_sparse, 0, 4)
        self.assertAlmostEqual(true_ratings[1][0], computed_ratings[1][0])


