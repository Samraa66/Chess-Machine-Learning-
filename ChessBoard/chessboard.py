import numpy as np
def generate_initial_black_pawns():
    black_pawns = np.zeros((8, 8), dtype=int)
    black_pawns[1, :] = 1
    return  black_pawns

black_pawns = generate_initial_black_pawns()


