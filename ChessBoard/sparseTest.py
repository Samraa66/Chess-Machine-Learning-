import tensorflow as tf
import numpy as np
import time


# Original methods using dense numpy arrays
def generate_initial_white_pawns_dense():
    white_pawns = np.zeros((8, 8), dtype=int)
    white_pawns[6, :] = 1
    return white_pawns


def generate_initial_white_rooks_dense():
    white_rooks = np.zeros((8, 8), dtype=int)
    white_rooks[7, 0] = 1
    white_rooks[7, 7] = 1
    return white_rooks


def generate_initial_white_knights_dense():
    white_knights = np.zeros((8, 8), dtype=int)
    white_knights[7, 1] = 1
    white_knights[7, 6] = 1
    return white_knights


def generate_initial_white_bishops_dense():
    white_bishops = np.zeros((8, 8), dtype=int)
    white_bishops[7, 2] = 1
    white_bishops[7, 5] = 1
    return white_bishops


def generate_initial_white_queen_dense():
    white_queen = np.zeros((8, 8), dtype=int)
    white_queen[7, 3] = 1
    return white_queen


def generate_initial_white_king_dense():
    white_king = np.zeros((8, 8), dtype=int)
    white_king[7, 4] = 1
    return white_king


def generate_initial_black_pawns_dense():
    black_pawns = np.zeros((8, 8), dtype=int)
    black_pawns[1, :] = 1
    return black_pawns


def generate_initial_black_rooks_dense():
    black_rooks = np.zeros((8, 8), dtype=int)
    black_rooks[0, 0] = 1
    black_rooks[0, 7] = 1
    return black_rooks


def generate_initial_black_knights_dense():
    black_knights = np.zeros((8, 8), dtype=int)
    black_knights[0, 1] = 1
    black_knights[0, 6] = 1
    return black_knights


def generate_initial_black_bishops_dense():
    black_bishops = np.zeros((8, 8), dtype=int)
    black_bishops[0, 2] = 1
    black_bishops[0, 5] = 1
    return black_bishops


def generate_initial_black_queen_dense():
    black_queen = np.zeros((8, 8), dtype=int)
    black_queen[0, 3] = 1
    return black_queen


def generate_initial_black_king_dense():
    black_king = np.zeros((8, 8), dtype=int)
    black_king[0, 4] = 1
    return black_king


def generate_board_dense():
    white_pawns = generate_initial_white_pawns_dense()
    white_rooks = generate_initial_white_rooks_dense()
    white_knights = generate_initial_white_knights_dense()
    white_bishops = generate_initial_white_bishops_dense()
    white_queen = generate_initial_white_queen_dense()
    white_king = generate_initial_white_king_dense()

    black_pawns = generate_initial_black_pawns_dense()
    black_rooks = generate_initial_black_rooks_dense()
    black_knights = generate_initial_black_knights_dense()
    black_bishops = generate_initial_black_bishops_dense()
    black_queen = generate_initial_black_queen_dense()
    black_king = generate_initial_black_king_dense()

    board = np.stack([
        white_pawns, white_rooks, white_knights, white_bishops, white_queen, white_king,
        black_pawns, black_rooks, black_knights, black_bishops, black_queen, black_king
    ], axis=-1)

    return board


# New methods using sparse tensors
def generate_initial_white_pawns_sparse():
    return [(6, i, 0) for i in range(8)]


def generate_initial_white_rooks_sparse():
    return [(7, 0, 1), (7, 7, 1)]


def generate_initial_white_knights_sparse():
    return [(7, 1, 2), (7, 6, 2)]


def generate_initial_white_bishops_sparse():
    return [(7, 2, 3), (7, 5, 3)]


def generate_initial_white_queen_sparse():
    return [(7, 3, 4)]


def generate_initial_white_king_sparse():
    return [(7, 4, 5)]


def generate_initial_black_pawns_sparse():
    return [(1, i, 6) for i in range(8)]


def generate_initial_black_rooks_sparse():
    return [(0, 0, 7), (0, 7, 7)]


def generate_initial_black_knights_sparse():
    return [(0, 1, 8), (0, 6, 8)]


def generate_initial_black_bishops_sparse():
    return [(0, 2, 9), (0, 5, 9)]


def generate_initial_black_queen_sparse():
    return [(0, 3, 10)]


def generate_initial_black_king_sparse():
    return [(0, 4, 11)]


def generate_board_sparse():
    indices = []
    indices += generate_initial_white_pawns_sparse()
    indices += generate_initial_white_rooks_sparse()
    indices += generate_initial_white_knights_sparse()
    indices += generate_initial_white_bishops_sparse()
    indices += generate_initial_white_queen_sparse()
    indices += generate_initial_white_king_sparse()
    indices += generate_initial_black_pawns_sparse()
    indices += generate_initial_black_rooks_sparse()
    indices += generate_initial_black_knights_sparse()
    indices += generate_initial_black_bishops_sparse()
    indices += generate_initial_black_queen_sparse()
    indices += generate_initial_black_king_sparse()

    indices = np.array(indices, dtype=int)
    values = np.ones(len(indices), dtype=int)

    sparse_board = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[8, 8, 12])
    sparse_board = tf.sparse.reorder(sparse_board)

    return sparse_board


# Benchmarking the two methods
def benchmark_method(method, name, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        _ = method()
    duration = time.time() - start_time
    print(f"{name} took {duration:.4f} seconds for {iterations} iterations")


# Run benchmarks
print("Benchmarking Dense Method")
benchmark_method(generate_board_dense, "Dense Method")

print("Benchmarking Sparse Method")
benchmark_method(generate_board_sparse, "Sparse Method")

import tensorflow as tf
import numpy as np
import time


# Original methods using dense numpy arrays
def generate_initial_white_pawns_dense():
    white_pawns = np.zeros((8, 8), dtype=int)
    white_pawns[6, :] = 1
    return white_pawns


def generate_initial_white_rooks_dense():
    white_rooks = np.zeros((8, 8), dtype=int)
    white_rooks[7, 0] = 1
    white_rooks[7, 7] = 1
    return white_rooks


def generate_initial_white_knights_dense():
    white_knights = np.zeros((8, 8), dtype=int)
    white_knights[7, 1] = 1
    white_knights[7, 6] = 1
    return white_knights


def generate_initial_white_bishops_dense():
    white_bishops = np.zeros((8, 8), dtype=int)
    white_bishops[7, 2] = 1
    white_bishops[7, 5] = 1
    return white_bishops


def generate_initial_white_queen_dense():
    white_queen = np.zeros((8, 8), dtype=int)
    white_queen[7, 3] = 1
    return white_queen


def generate_initial_white_king_dense():
    white_king = np.zeros((8, 8), dtype=int)
    white_king[7, 4] = 1
    return white_king


def generate_initial_black_pawns_dense():
    black_pawns = np.zeros((8, 8), dtype=int)
    black_pawns[1, :] = 1
    return black_pawns


def generate_initial_black_rooks_dense():
    black_rooks = np.zeros((8, 8), dtype=int)
    black_rooks[0, 0] = 1
    black_rooks[0, 7] = 1
    return black_rooks


def generate_initial_black_knights_dense():
    black_knights = np.zeros((8, 8), dtype=int)
    black_knights[0, 1] = 1
    black_knights[0, 6] = 1
    return black_knights


def generate_initial_black_bishops_dense():
    black_bishops = np.zeros((8, 8), dtype=int)
    black_bishops[0, 2] = 1
    black_bishops[0, 5] = 1
    return black_bishops


def generate_initial_black_queen_dense():
    black_queen = np.zeros((8, 8), dtype=int)
    black_queen[0, 3] = 1
    return black_queen


def generate_initial_black_king_dense():
    black_king = np.zeros((8, 8), dtype=int)
    black_king[0, 4] = 1
    return black_king


def generate_board_dense():
    white_pawns = generate_initial_white_pawns_dense()
    white_rooks = generate_initial_white_rooks_dense()
    white_knights = generate_initial_white_knights_dense()
    white_bishops = generate_initial_white_bishops_dense()
    white_queen = generate_initial_white_queen_dense()
    white_king = generate_initial_white_king_dense()

    black_pawns = generate_initial_black_pawns_dense()
    black_rooks = generate_initial_black_rooks_dense()
    black_knights = generate_initial_black_knights_dense()
    black_bishops = generate_initial_black_bishops_dense()
    black_queen = generate_initial_black_queen_dense()
    black_king = generate_initial_black_king_dense()

    board = np.stack([
        white_pawns, white_rooks, white_knights, white_bishops, white_queen, white_king,
        black_pawns, black_rooks, black_knights, black_bishops, black_queen, black_king
    ], axis=-1)

    return board


# New methods using sparse tensors
def generate_initial_white_pawns_sparse():
    return [(6, i, 0) for i in range(8)]


def generate_initial_white_rooks_sparse():
    return [(7, 0, 1), (7, 7, 1)]


def generate_initial_white_knights_sparse():
    return [(7, 1, 2), (7, 6, 2)]


def generate_initial_white_bishops_sparse():
    return [(7, 2, 3), (7, 5, 3)]


def generate_initial_white_queen_sparse():
    return [(7, 3, 4)]


def generate_initial_white_king_sparse():
    return [(7, 4, 5)]


def generate_initial_black_pawns_sparse():
    return [(1, i, 6) for i in range(8)]


def generate_initial_black_rooks_sparse():
    return [(0, 0, 7), (0, 7, 7)]


def generate_initial_black_knights_sparse():
    return [(0, 1, 8), (0, 6, 8)]


def generate_initial_black_bishops_sparse():
    return [(0, 2, 9), (0, 5, 9)]


def generate_initial_black_queen_sparse():
    return [(0, 3, 10)]


def generate_initial_black_king_sparse():
    return [(0, 4, 11)]


def generate_board_sparse():
    indices = []
    indices += generate_initial_white_pawns_sparse()
    indices += generate_initial_white_rooks_sparse()
    indices += generate_initial_white_knights_sparse()
    indices += generate_initial_white_bishops_sparse()
    indices += generate_initial_white_queen_sparse()
    indices += generate_initial_white_king_sparse()
    indices += generate_initial_black_pawns_sparse()
    indices += generate_initial_black_rooks_sparse()
    indices += generate_initial_black_knights_sparse()
    indices += generate_initial_black_bishops_sparse()
    indices += generate_initial_black_queen_sparse()
    indices += generate_initial_black_king_sparse()

    indices = np.array(indices, dtype=int)
    values = np.ones(len(indices), dtype=int)

    sparse_board = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[8, 8, 12])
    sparse_board = tf.sparse.reorder(sparse_board)

    return sparse_board


# Function to find the indices of ones in a dense tensor
def find_ones_dense(board):
    return np.argwhere(board == 1)


# Function to find the indices of ones in a sparse tensor
def find_ones_sparse(sparse_board):
    return sparse_board.indices.numpy()


# Benchmarking the two methods
def benchmark_method(generate_method, find_method, name, iterations=1000):
    # Benchmark the generation of the board
    start_time = time.time()
    for _ in range(iterations):
        board = generate_method()
    generation_duration = time.time() - start_time

    # Benchmark finding the ones
    start_time = time.time()
    for _ in range(iterations):
        indices = find_method(board)
    finding_duration = time.time() - start_time

    print(f"{name} - Generation took {generation_duration:.4f} seconds for {iterations} iterations")
    print(f"{name} - Finding ones took {finding_duration:.4f} seconds for {iterations} iterations")


# Run benchmarks
print("Benchmarking Dense Method")
benchmark_method(generate_board_dense, find_ones_dense, "Dense Method")

print("Benchmarking Sparse Method")
benchmark_method(generate_board_sparse, find_ones_sparse, "Sparse Method")
