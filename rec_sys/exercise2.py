import numpy as np
from scipy.sparse.linalg import norm
from scipy.sparse import csr_array, csr_matrix


def centered_cosine_sim(row_array, col_array):
    row_mean = np.nanmean(row_array)
    col_mean = np.nanmean(col_array)
    sparse_row = csr_array(np.nan_to_num(row_array - row_mean))
    sparse_col = csr_array(np.nan_to_num(col_array - col_mean))
    dot = sparse_row.dot(sparse_col)
    return dot / (norm(sparse_row) * norm(sparse_col))

def fast_centered_cosine_sim(matrix, vector, axis = 0):

    means_matrix = np.nanmean(matrix, axis)
    matrix_centered = csr_matrix(np.nan_to_num(matrix - means_matrix))
    norms_centered_matrix = norm(matrix_centered, axis=axis)
    matrix_centered_normalized = matrix_centered / norms_centered_matrix

    mean_vector = np.nanmean(vector)
    centered_vector = csr_matrix(np.nan_to_num(vector - mean_vector)).transpose() #transpose to column vector
    dot = matrix_centered_normalized.transpose().dot(centered_vector)

    return dot / norm(centered_vector)
