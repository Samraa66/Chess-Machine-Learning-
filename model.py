import tensorflow as tf
import keras
from keras import layers
import numpy as np
import random

random.seed(10)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, num_hidden):
        super(ResBlock, self).__init__()
        self.conv1 = layers.Conv2D(num_hidden, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(num_hidden, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = tf.nn.relu(x)
        return x

class AlphaNet(keras.Model):
    def __init__(self, num_resBlocks, num_hidden, t):
        super(AlphaNet, self).__init__()
        self.startBlock = tf.keras.Sequential([
            layers.Input(shape=(8, 8, 12)),  # Adjust based on Chess Board # NEED TO MAKE LAYER, X, Y
            layers.Conv2D(num_hidden, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.backBone = [ResBlock(num_hidden) for _ in range(num_resBlocks)]

        self.policyHead = tf.keras.Sequential([
            layers.Conv2D(73, kernel_size=1, padding='same'),  # 73 due to the 73 planes for possible moves
            layers.Softmax()
        ])

        self.valueHead = tf.keras.Sequential([
            layers.Conv2D(1, kernel_size=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(256),
            layers.ReLU(),
            layers.Dense(1),
            layers.Activation('tanh')
        ])

    def call(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value



def generate_initial_board_s():
    array_white_pawns = [(6, i, 0) for i in range(8)]
    array_black_pawns = [(1, i, 6) for i in range(8)]

    indices = array_white_pawns + [
        (7, 1, 1), (7, 6, 1), (7, 2, 2), (7, 5, 2),
        (7, 0, 3), (7, 7, 3), (7, 3, 4), (7, 4, 5)
    ] + array_black_pawns + [
                  (0, 1, 7), (0, 6, 7), (0, 2, 8), (0, 5, 8),
                  (0, 0, 9), (0, 7, 9), (0, 3, 10), (0, 4, 11)
              ]

    values = np.ones(len(indices), dtype=int)
    sparse_board = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[8, 8, 12])

    return sparse_board

def get_initial_piece_position_dict_and_tensor():
    initial_piece_to_position = {}
    initial_tensor = tf.zeros((8, 8, 12))
    indices = []

    # for pawns
    initial_piece_to_position[0] = []
    array_white_pawns = []
    array_black_pawns = []
    for i in range(8):
        initial_piece_to_position[0].append((6, i)) # white
        initial_piece_to_position[6].append((1, i)) # black

        indices.append((6, i, 0))
        indices.append((1, i, 6))

    # for knights
    initial_piece_to_position[1] = [(7, 1), (7, 6)] # white
    indices.extend((7, 1, 1), (7, 6, 1))
    initial_piece_to_position[7] = [(0, 1), (0, 6)] # black
    indices.extend((0, 1, 7), (0, 6, 7))

    # for bishop
    initial_piece_to_position[2] = [(7, 2), (7, 5)] # white
    indices.extend((7, 2, 2), (7, 5, 2))
    initial_piece_to_position[8] = [(0, 2), (0, 5)] # black
    indices.extend((0, 2, 8), (0, 5, 8))

    # rook
    initial_piece_to_position[3] = [(7, 0,), (7, 7)] # white
    indices.extend((7, 0, 3), (7, 7, 3))
    initial_piece_to_position[9] = [((0, 0, 9)), ((0, 7, 9))] # black
    indices.extend((0, 0, 9), (0, 7, 9))

    # queen
    initial_piece_to_position[4] = [(7, 3)] # white
    indices.append((7, 3, 4))
    initial_piece_to_position[10] = [(0, 3)] # black
    indices.append((0, 3, 10))

    # king
    initial_piece_to_position[5] = [(7, 4)] # white
    indices.append((7, 4, 5))
    initial_piece_to_position[11] = [(0, 4)]
    indices.append((0, 4, 11))


    # generate tensor
    updates = tf.ones(len(indices), dtype=tf.float32)
    updated_tensor = tf.tensor_scatter_nd_update(initial_tensor, indices, updates)

    return updated_tensor, initial_piece_to_position

def generate_initial_board_st():
    array_white_pawns = [(6, i, 0) for i in range(8)]
    array_black_pawns = [(1, i, 6) for i in range(8)]

    indices = array_white_pawns + [
        (7, 1, 1), (7, 6, 1), (7, 2, 2), (7, 5, 2),
        (7, 0, 3), (7, 7, 3), (7, 3, 4), (7, 4, 5)
    ] + array_black_pawns + [
        (0, 1, 7), (0, 6, 7), (0, 2, 8), (0, 5, 8),
        (0, 0, 9), (0, 7, 9), (0, 3, 10), (0, 4, 11)
    ]

    # Sort indices by row and column
    indices.sort()

    values = np.ones(len(indices), dtype=int)
    sparse_board = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[8, 8, 12])

    return sparse_board


def add_batch_dimension(sparse_tensor):

    # Convert sparse tensor to dense tensor
    dense_tensor = tf.sparse.to_dense(sparse_tensor)

    # Add batch dimension using tf.expand_dims
    batch_dense_tensor = tf.expand_dims(dense_tensor, axis=0)

    return batch_dense_tensor

def print_matrices_from_tensor(dense_tensor):
    # Iterate through the 73 matrices
    for i in range(dense_tensor.shape[-1]):
        # Extract the 8x8 matrix
        matrix = dense_tensor[0, :, :, i]
        print(f"Matrix {i + 1}:\n{matrix.numpy()}\n")

# Test
num_resBlocks = 3
num_hidden = 64
t = 1
initial_board = generate_initial_board_st()
initial_board = add_batch_dimension(initial_board)

#random_tensor = tf.random.uniform(shape=(1, 8, 8, 119), minval=0, maxval=1)
model = AlphaNet(num_resBlocks, num_hidden, 8)


# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# Define loss functions
policy_loss_fn = tf.keras.losses.CategoricalCrossentropy()
value_loss_fn = tf.keras.losses.MeanSquaredError()


policy, value = model(initial_board)
print('policy')
print_matrices_from_tensor(policy)

policy_flat = tf.reshape(policy, [-1])

# Apply the softmax function OVER ALL LAYERS
policy_softmax = tf.nn.softmax(policy_flat)
# Reshape back to the original shape
policy_softmax_reshaped = tf.reshape(policy_softmax, policy.shape)
sum_of_entries = tf.reduce_sum(policy_softmax_reshaped)

print("Sum of all entries in the tensor:", sum_of_entries.numpy())

print('Value')
print(value)



'''
# Training step
with tf.GradientTape() as tape:
    policy, value = model(random_tensor) # use model x to generate tensor output rather than model.predict(x)
    # https://stackoverflow.com/questions/55308425/difference-between-modelx-and-model-predictx-in-keras ^^
    policy_loss = policy_loss_fn(random_policy_target, policy)
    value_loss = value_loss_fn(random_value_target, value)
    total_loss = policy_loss + value_loss

# Compute gradients
gradients = tape.gradient(total_loss, model.trainable_variables)

# Apply gradients
optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Print the losses
print("Policy Loss:", policy_loss.numpy())
print("Value Loss:", value_loss.numpy())

'''