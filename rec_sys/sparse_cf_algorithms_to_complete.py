# Artur Andrzejak, October 2024
# Algorithms for collaborative filtering

import numpy as np
from exercise2 import fast_centered_cosine_sim, sparse_center_and_nan_to_zero
from scipy.sparse import csr_array, csc_array, csr_matrix
from scipy.sparse.linalg import norm


# Implement the CF from the lecture 1
def rate_all_items(orig_sparse_matrix, user_index, neighborhood_size):
    print(f"\n>>> CF computation for UM w/ shape: "
          + f"{orig_sparse_matrix.shape}, user_index: {user_index}, neighborhood_size: {neighborhood_size}\n")

    clean_sparse_matrix = sparse_center_and_nan_to_zero(orig_sparse_matrix)
    user_col = clean_sparse_matrix[:, user_index]
    # Compute the cosine similarity between the user and all other users
    similarities = fast_centered_cosine_sim(clean_sparse_matrix, user_col)


    def rate_one_item(item_index):
        # If the user has already rated the item, return the rating
        if not np.isnan(orig_sparse_matrix[item_index, user_index]):
            return orig_sparse_matrix[item_index, user_index]
        # Find the indices of users who rated the item
        users_who_rated = np.where(np.isnan(orig_sparse_matrix[item_index, :].toarray()) == False)[1]
        # From those, get indices of users with the highest similarity (watch out: result indices are rel. to users_who_rated)
        best_among_who_rated = np.argsort(similarities[users_who_rated].toarray().flatten())
        # Select top neighborhood_size of them
        best_among_who_rated = best_among_who_rated[-neighborhood_size:]
        # Convert the indices back to the original utility matrix indices
        best_among_who_rated = users_who_rated[best_among_who_rated]
        # Retain only those indices where the similarity is not nan
        best_among_who_rated = best_among_who_rated[np.isnan(similarities[best_among_who_rated].toarray().flatten()) == False]
        if best_among_who_rated.size > 0:
            # Compute the rating of the item
            ratings_of_selected_users = clean_sparse_matrix[item_index, best_among_who_rated]
            similarities_of_selected_users = similarities[best_among_who_rated]
            rating_of_item = np.dot(ratings_of_selected_users, similarities_of_selected_users) \
                    / np.sum(np.abs(similarities_of_selected_users))
        else:
            rating_of_item = np.nan
        print(f"item_idx: {item_index}, neighbors: {best_among_who_rated}, rating: {rating_of_item}")
        return rating_of_item.data

    num_items = orig_sparse_matrix.shape[0]

    # Get all ratings
    ratings = list(map(rate_one_item, range(num_items)))
    return ratings

