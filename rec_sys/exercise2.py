import numpy as np
from scipy.sparse.linalg import norm
from scipy.sparse import csr_matrix


def centered_cosine_sim(sparse_row, sparse_col):
    #sparse_row and sparse_col are csr_matrix
    row_mean = np.nanmean(sparse_row.toarray())
    col_mean = np.nanmean(sparse_col.toarray())
    sparse_row = csr_matrix(np.nan_to_num(sparse_row.toarray() - row_mean))
    sparse_col = csr_matrix(np.nan_to_num(sparse_col.toarray() - col_mean))
    dot = sparse_row.dot(sparse_col.transpose())
    return dot / (norm(sparse_row.transpose()) * norm(sparse_col.transpose()))

def sparse_center_and_nan_to_zero(sparse_matrix, axis = 0):
    means_matrix = np.nanmean(sparse_matrix.toarray(), axis)
    centered_matrix = sparse_matrix.toarray() - means_matrix
    centered_matrix = csr_matrix(np.nan_to_num(centered_matrix))
    return csr_matrix(centered_matrix)

def fast_centered_cosine_sim(centered_sparse_matrix, centered_vector, axis = 0):
    norms_matrix = norm(centered_sparse_matrix, axis=axis)
    norm_of_vector = norm(centered_vector, axis=axis)
    matrix_normed = centered_sparse_matrix / norms_matrix
    vector_normed = centered_vector / norm_of_vector
    dot = matrix_normed.transpose().dot(vector_normed)

    return dot / norm_of_vector
