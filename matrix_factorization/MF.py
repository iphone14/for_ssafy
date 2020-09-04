import numpy as np

def matrix_factorization(R, k, iteration):

    user_count, item_count = R.shape

    weight_P = np.random.normal(size=(user_count, k))
    weight_Q = np.random.normal(size=(item_count, k))

    bias_P = np.zeros(user_count)
    bias_Q = np.zeros(item_count)

    for iter in range(iteration):
        for u in range(user_count):
            for i in range(item_count):
                r = R[u, i]
                if r >= 0:
                    error = r - prediction(weight_P[u, :], weight_Q[i, :], bias_P[u], bias_Q[i])

                    weight_delta_Q, bias_delta_Q = gradient(error, weight_P[u, :])
                    weight_delta_P, bias_delta_P = gradient(error, weight_Q[i, :])

                    weight_P[u, :] += weight_delta_P
                    bias_P[u] += bias_delta_P

                    weight_Q[i, :] += weight_delta_Q
                    bias_Q[i] += bias_delta_Q

    return weight_P.dot(weight_Q.T) + bias_P[:, np.newaxis] + bias_Q[np.newaxis:, ]

def gradient(error, weight):

    learning_rate = 0.01

    weight_delta = learning_rate * error * weight

    bias_delta = learning_rate * error

    return weight_delta, bias_delta


def prediction(P, Q, b_P, b_Q):

	return P.dot(Q.T) + b_P + b_Q



iteration = 5000
k = 3
R = np.array([
    [2, 8, 9, 1, 8],
    [8, 2, 1, 8, 1],
    [1, 5, -1, 1, 7],
    [7, 2, 1, 8, 1],
    [1, 8, 9, 2, 9],
    [9, 1, 2, -1, 2],
    [6, 1, 2, 7, 2]])

predicted_R = matrix_factorization(R, k, iteration)

print(predicted_R)
