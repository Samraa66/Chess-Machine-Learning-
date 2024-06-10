import tensorflow as tf
import time

# Initialize the tensor
initial_tensor = tf.zeros((8, 8, 12))

# Positions to insert ones
array_white_pawns = [(6, i, 0) for i in range(8)]

# Method 1: Using tf.tensor_scatter_nd_update
def using_tensor_scatter_nd_update():
    positions = array_white_pawns
    updates = tf.ones(len(positions))
    updated_tensor = tf.tensor_scatter_nd_update(initial_tensor, positions, updates)
    return updated_tensor

# Method 2: Using a for loop without converting to numpy
def using_for_loop():
    tensor = tf.Variable(initial_tensor)
    for pos in array_white_pawns:
        tensor[pos].assign(1)
    return tensor

# Number of iterations for benchmarking
iterations = 1000

# Measure execution time of tf.tensor_scatter_nd_update
scatter_total_time = 0
for _ in range(iterations):
    start_time = time.time()
    tensor_scatter_result = using_tensor_scatter_nd_update()
    scatter_total_time += time.time() - start_time

# Measure execution time of for loop
for_loop_total_time = 0
for _ in range(iterations):
    start_time = time.time()
    for_loop_result = using_for_loop()
    for_loop_total_time += time.time() - start_time

# Calculate average times
scatter_avg_time = scatter_total_time / iterations
for_loop_avg_time = for_loop_total_time / iterations

# Output results
print("Average execution time using tf.tensor_scatter_nd_update: {:.6f} seconds".format(scatter_avg_time))
print("Average execution time using for loop: {:.6f} seconds".format(for_loop_avg_time))

# Optionally, compare the tensors to ensure they are the same
print("Tensors are equal:", tf.reduce_all(tf.equal(tensor_scatter_result, for_loop_result)).numpy())
