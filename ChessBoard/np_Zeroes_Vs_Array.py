import numpy as np
import timeit

# Define the setup for np.array
setup_array = """
import numpy as np
def initialize_with_array():
    black_pawns = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
"""

# Define the setup for np.zeros
setup_zeros = """
import numpy as np
def initialize_with_zeros():
    black_pawns = np.zeros((8, 8), dtype=int)
    black_pawns[1, :] = 1
"""

# Timing np.array initialization
time_array = timeit.timeit("initialize_with_array()", setup=setup_array, number=10000)

# Timing np.zeros initialization
time_zeros = timeit.timeit("initialize_with_zeros()", setup=setup_zeros, number=10000)

print(f"Time using np.array: {time_array} seconds")
print(f"Time using np.zeros: {time_zeros} seconds")
