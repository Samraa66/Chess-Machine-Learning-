import tensorflow as tf
import keras
from keras import layers
import numpy as np

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
            layers.Input(shape=(8, 8, 12)),  # Adjust based on Chess Board
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
            layers.Conv2D(3, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
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

    v = tf.sparse.to_dense(sparse_board)
    print(v)

    return sparse_board


def add_batch_dimension(sparse_tensor):

    # Convert sparse tensor to dense tensor
    dense_tensor = tf.sparse.reorder(sparse_tensor)
    dense_tensor = tf.sparse.to_dense(dense_tensor)

    # Add batch dimension using tf.expand_dims
    batch_dense_tensor = tf.expand_dims(dense_tensor, axis=0)

    return batch_dense_tensor

# Test
num_resBlocks = 3
num_hidden = 64
t = 1
random_tensor = generate_initial_board_st()

'''
random_tensor =  add_batch_dimension(random_tensor)
print(random_tensor)
#random_tensor = tf.random.uniform(shape=(1, 8, 8, 119), minval=0, maxval=1)
model = AlphaNet(num_resBlocks, num_hidden, 8)


# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# Define loss functions
policy_loss_fn = tf.keras.losses.CategoricalCrossentropy()
value_loss_fn = tf.keras.losses.MeanSquaredError()


policy, value = model(random_tensor)
print('policy')
print(policy)

print('Value')
print(value)
'''
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