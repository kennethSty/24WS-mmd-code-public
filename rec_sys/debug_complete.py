import numpy as np

import cf_algorithms_to_complete as cf

matrix = np.arange(100).reshape(10, 10)
matrix = matrix.astype(float)
matrix[1, 3] = np.nan
ratings = cf.rate_all_items(matrix, 3, 4)