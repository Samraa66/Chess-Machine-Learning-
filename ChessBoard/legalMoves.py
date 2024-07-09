import numpy as np
import tensorflow as tf
import chessboard

pieces_and_moves = {
    0: (((0, 3, 4, 5), (3, 4, 5)), (-1, 0)),
    1: (((0, 3, 4), (3, 4)), (-2, 0)),
    2: (((3, 4), (3, 4)), (-3, 0)),
    3: (((3, 4), (3, 4)), (-4, 0)),
    4: (((0, 3, 4), (3, 4)), (-5, 0)),
    5: (((3, 4), (3, 4)), (-6, 0)),
    6: (((3, 4), (3, 4)), (-7, 0)),
    7: (((0, 2, 4, 5), (2, 4, 5)), (-1, 1)),
    8: (((2, 4), (2, 4)), (-2, 2)),
    9: (((2, 4), (2, 4)), (-3, 3)),
    10: (((2, 4), (2, 4)), (-4, 4)),
    11: (((2, 4), (2, 4)), (-5, 5)),
    12: (((2, 4), (2, 4)), (-6, 6)),
    13: (((2, 4), (2, 4)), (-7, 7)),
    14: (((3, 4, 5), (3, 4, 5)), (0, 1)),
    15: (((3, 4, 5), (2, 4, 5)), (0, 2)),
    16: (((3, 4), (3, 4)), (0, 3)),
    17: (((3, 4), (3, 4)), (0, 4)),
    18: (((3, 4), (3, 4)), (0, 5)),
    19: (((3, 4), (3, 4)), (0, 6)),
    20: (((3, 4), (3, 4)), (0, 7)),
    21: (((2, 4, 5), (0, 2, 4, 5)), (1, 1)),
    22: (((2, 4), (2, 4)), (2, 2)),
    23: (((2, 4), (2, 4)), (3, 3)),
    24: (((2, 4), (2, 4)), (4, 4)),
    25: (((2, 4), (2, 4)), (5, 5)),
    26: (((2, 4), (2, 4)), (6, 6)),
    27: (((2, 4), (2, 4)), (7, 7)),
    28: (((3, 4, 5), (3, 4, 5)), (1, 0)),
    29: (((3, 4), (0, 3, 4, 5)), (2, 0)),
    30: (((3, 4), (0, 3, 4)), (3, 0)),
    31: (((3, 4), (3, 4)), (4, 0)),
    32: (((3, 4), (3, 4)), (5, 0)),
    33: (((3, 4), (3, 4)), (6, 0)),
    34: (((3, 4), (3, 4)), (7, 0)),
    35: (((2, 4, 5), (0, 2, 4, 5)), (1, -1)),
    36: (((2, 4), (2, 4)), (2, -2)),
    37: (((2, 4), (2, 4)), (3, -3)),
    38: (((2, 4), (2, 4)), (4, -4)),
    39: (((2, 4), (2, 4)), (5, -5)),
    40: (((2, 4), (2, 4)), (6, -6)),
    41: (((2, 4), (2, 4)), (7, -7)),
    42: (((3, 4, 5), (3, 4, 5)), (0, -1)),
    43: (((3, 4, 5), (3, 4, 5)), (0, -2)),
    44: (((3, 4), (3, 4)), (0, -3)),
    45: (((3, 4), (3, 4)), (0, -4)),
    46: (((3, 4), (3, 4)), (0, -5)),
    47: (((3, 4), (3, 4)), (0, -6)),
    48: (((3, 4), (3, 4)), (0, -7)),
    49: (((0, 2, 4, 5), (2, 4, 5)), (-1, -1)),
    50: (((2, 4), (2, 4)), (-2, -2)),
    51: (((2, 4), (2, 4)), (-3, -3)),
    52: (((2, 4), (2, 4)), (-4, -4)),
    53: (((2, 4), (2, 4)), (-5, -5)),
    54: (((2, 4), (2, 4)), (-6, -6)),
    55: (((2, 4), (2, 4)), (-7, -7)),
    56: (((1,), (1,)), (-2, 1)),
    57: (((1,), (1,)), (-1, 2)),
    58: (((1,), (1,)), (1, 2)),
    59: (((1,), (1,)), (2, 1)),
    60: (((1,), (1,)), (2, -1)),
    61: (((1,), (1,)), (1, -2)),
    62: (((1,), (1,)), (-1, -2)),
    63: (((1,), (1,)), (-2, -1)),
    64: (((1,), (1,)), ((-1, -1), (1, -1))),
    65: (((0,), (0,)), ((-1, 0), (1, 0))),
    66: (((0,), (0,)), ((-1, 1), (1, 1))),
    67: (((0,), (0,)), ((-1, -1), (1, -1))),
    68: (((0,), (0,)), ((-1, 0), (1, 0))),
    69: (((0,), (0,)), ((-1, 1), (1, 1))),
    70: (((0,), (0,)), ((-1, -1), (1, -1))),
    71: (((0,), (0,)), ((-1, 0), (1, 0))),
    72: (((0,), (0,)), ((-1, 1), (1, 1))),
}


'''
# Generate initial piece positions
white_pawns = chessboard.generate_initial_white_pawns()
white_rooks = chessboard.generate_initial_white_rooks()
white_knights = chessboard.generate_initial_white_knights()
white_bishops = chessboard.generate_initial_white_bishops()
white_queen = chessboard.generate_initial_white_queen()
white_king = chessboard.generate_initial_white_king()
black_pawns = chessboard.generate_initial_black_pawns()
black_rooks = chessboard.generate_initial_black_rooks()
black_knights = chessboard.generate_initial_black_knights()
black_bishops = chessboard.generate_initial_black_bishops()
black_queen = chessboard.generate_initial_black_queen()
black_king = chessboard.generate_initial_black_king()
game_state = tf.stack([white_pawns, white_knights, white_bishops, white_rooks, white_queen, white_king, black_pawns, black_knights, black_bishops, black_rooks, black_queen, black_king], axis=0)
game_state = tf.sparse.from_dense(game_state)
'''


class Chessboard:
    def generate_empty_board(self):
        return np.zeros([8, 8], dtype=int)

    def place_piece(self, board, row, col, piece_id):
        board[row, col] = piece_id
        return board

    def generate_board_with_piece(self, row, col, piece_id):
        board = self.generate_empty_board()
        return self.place_piece(board, row, col, piece_id)


chessboard = Chessboard()

# Generate the initial positions for each piece type
white_pawns = chessboard.generate_empty_board()
white_rooks = chessboard.generate_empty_board()
white_knights = chessboard.generate_empty_board()
white_bishops = chessboard.generate_empty_board()
white_queen = chessboard.generate_empty_board()
white_king = chessboard.generate_board_with_piece(4, 4, 1)

black_pawns = chessboard.generate_empty_board()
black_rooks = chessboard.generate_empty_board()
black_knights = chessboard.generate_empty_board()
black_bishops = chessboard.generate_empty_board()
black_queen = chessboard.generate_board_with_piece(4, 7, 1)
black_king = chessboard.generate_empty_board()

# Place the white rook at (4, 5)
white_rooks = chessboard.place_piece(white_rooks, 4, 5, 1)

# Stack the arrays to create the game state
game_state = tf.stack([
white_pawns, white_knights, white_bishops, white_rooks, white_queen, white_king, black_pawns, black_knights, black_bishops, black_rooks, black_queen, black_king
], axis=0)

# Convert the dense game state tensor to a sparse tensor
game_state = tf.sparse.from_dense(game_state)



# 5 move history gamestate
def generate_legal_moves(game_state):
    # first we check tensor for currrent players turn
    #turn = game_state[0, 0, 70]  # should be zero or one
    turn = 0
    all_indices = game_state.indices
    my_positions = []  # potentially need to do .numpy()
    your_positions = []
    # finding pieces
    if turn == 0:
        #my_pieces = game_state[:, :, 56:61]
        #your_pieces = game_state[:, :, 62:67]
        # NEED TO CHANGE TO LAYER, X, Y
        my_positions = tf.boolean_mask(all_indices, all_indices[:, 0] < 5)
        your_positions = tf.boolean_mask(all_indices, all_indices[:, 0] > 5)

    else:
        #my_pieces = game_state[:, :, 62:67]
        #your_pieces = game_state[:, :, 56:61]
        your_positions = tf.boolean_mask(all_indices, all_indices[:, 0] < 5)
        my_positions = tf.boolean_mask(all_indices, all_indices[:, 0] > 5)
        # my positions will contain locations of pieces on board

    # idea for pins, initialize all pieces with allowed squares of all 8x8 board, except for the first piece in one of 9 radial directions from king that right after has a piece of the opposite color that attacks in this direction, that piece is pinned and can only move in this specified direction.

    # legal moves should be in the form piece_x, piece_y, layer
    # these are initializing the return var.
    legal_moves = []
    game_states = []
    piece_types = []
    for i in range(73):
        if i == 58:
            continue
        # for each layer we want to check if pieces can move in layer direction

        # get both pieces that make move of type i, and displacement of move i
        piece_and_move = pieces_and_moves[i]
        # grab piece types that would make move i (w,b)
        piece_for_this_kind_of_move = piece_and_move[0]
        # for each of my pieces on the board
        for piece_type, piece_x, piece_y in my_positions:
            if piece_type in piece_for_this_kind_of_move[turn]:
                # here we know we can move this piece in this direction so test potential location
                potential_piece_x, potential_piece_y = np.add((piece_x, piece_y), piece_and_move[1])
                # first check if new location is out of bounds or my piece is blocking
                within_board = (0 <= potential_piece_x < 8
                                and 0 <= potential_piece_y < 8)
                piece_present = False
                for pos in my_positions:
                    if potential_piece_x == pos[1] and potential_piece_y == pos[2]:
                        piece_present = True
                        break

                if (within_board and not piece_present):
                    # CURRENTLY THIS MEANS MOVE IS LEGAL (SHOULD UPDATE TO CHECK PINS AND PATHS)
                    # If legal move, copy game state, make move, and
                    # legal moves should be in the form piece_x, piece_y, layer
                    legal_moves.append(
                      (piece_x, piece_y, i))  # one task could be actually
                    # manipulating the 8x8x73 matrix here

                    # first check if capture
                    piece_capture = False
                    piece_location = None
                    for _, pos in enumerate(your_positions):
                        if potential_piece_x == pos[1] and potential_piece_y == pos[2]:
                            piece_capture = True
                            #storing opp's piece location to remove his piece
                            piece_location = pos
                            break

                    # tried use chat, cause i cant find stuff on deep copying a sparse tensor so
                    # maybe conversion is the way to go
                    # (i think manipulation would also be easier),
                    # need to debug if actually deep copy
                    # copy game state
                    new_game_state = tf.sparse.to_dense(game_state)
                    new_game_state = new_game_state.numpy()

                    # update the new game state
                    new_game_state[piece_type, piece_x, piece_y] = 0
                    new_game_state[piece_type, potential_piece_x,
                               potential_piece_y] = 1

                    # if there was a captured remove the piece
                    # python none same as false? lets check by debugging if this will work as intended
                    if piece_capture and piece_location is not None:
                        new_game_state[piece_location[0], piece_location[1],
                                 piece_location[2]] = 0

                    print(new_game_state)
                    print("----------------------------------------------------------------------------------------")
                    # Convert back to sparse tensor and add to new game states
                    new_game_state_sparse = tf.sparse.from_dense(new_game_state)
                    game_states.append(new_game_state_sparse)
                    # lastly need to add piece type for legal move for visualization
                    piece_types.append(
                    i)  # one task could be getting this in the right format

      # TO DOS
      # when updating game state if capture update tensor and remove opp piece
      # is this not done^?
      # return 8x8x73 legal move tensor, list of game state tensors per move, piece type per         move
      # recursively check for pieces blocking movement through move hierarchy
      # figure out move history and concatenating it to game state for model (should be easy)
      # determine checks and checkmate and handle them ??
      # edge case: my king cannot move to a square threatened by another piece (implementation might be hell)
    # UPDATES
    # REFACTORED FOR READABILITY
    #if not (potential_piece_x>8 or potential_piece_x<0 or potential_piece_y>8 or       #potential_piece_y<0 or any(potential_piece_x == pos[0] and potential_piece_y == pos[1] for   #pos in my_positions)):

#def king_is_threatened(game_state, position ):
    #checks if the given position is threatened by any of the oponents moves
  # game_state is the game state tensor, position is the position to check.
  # function should return true if the position is at risk false if not
  #opponent_moves = generate_opponent_moves(game_state)
    #for move in opponent_moves:
      #if move[0] == position[0] and move[1] == position[1]:
# so here i checked if the x-cord of the oponents move is the same as the x-cord of the king and the same for y-cord if they both match then the king is threatend

      #  return True
   # return false

#def generate_opponent_moves(game_state):
  #opponent_moves = []
  #turn = game_state[0, 0, 70]
  #opponent_pieces = game_state[:, :, 62:67] if turn == 0 else    game_state[:, :, 56:61]

  # trying to iterate through opp pieces
  #for piece_x, piece_y, piece_type in opponent_pieces.indices:
    #moves = generate_

#def is_in_check( game_state)
  # does it make sense to say that if king_is_threatened is true then the king is in check?

def get_pinned_pieces(game_state, position):
    # position is where the king is
    all_indices = game_state.indices

    my_piece_present = False
    my_piece_location = []
    enemy_piece_present = False
    enemy_piece_location = []

    my_positions = tf.boolean_mask(all_indices, all_indices[:, 0] <= 5)
    your_positions = tf.boolean_mask(all_indices, all_indices[:, 0] > 5)

    row, col = position

    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (1, 0), (0, 1), (-1, 0),
                  (0, -1)]  # Diagonals and straight directions

    for direction in directions:
        if direction == (1,0) or direction == (1,0):
            print("la")
        my_piece_present = False
        enemy_piece_present = False
        my_piece_location = []
        enemy_piece_location = []

        d_row, d_col = direction
        current_row, current_col = row + d_row, col + d_col

        while 0 <= current_row < 8 and 0 <= current_col < 8:
            # Check for my pieces
            same_pos = tf.boolean_mask(my_positions,
                                       (my_positions[:, 1] == current_col) & (my_positions[:, 2] == current_row))
            if tf.size(same_pos) > 0:
                my_piece_present = True
                my_piece_location = same_pos[0].numpy()
                # Further logic if my piece is present

            # Check for enemy pieces
            same_pos_enemy = tf.boolean_mask(your_positions, (your_positions[:, 1] == current_col) & (
                        your_positions[:, 2] == current_row))
            if tf.size(same_pos_enemy) > 0:
                enemy_piece_present = True
                enemy_piece_location = same_pos_enemy[0].numpy()
                # Once we find an enemy piece stop checking
                break

            # Move to the next position in the specified direction
            current_row += d_row
            current_col += d_col

        if my_piece_present and enemy_piece_present:
            # this means my piece is pinned
            print("My piece is pinned at:", my_piece_location)
            print("Enemy piece causing the pin is at:", enemy_piece_location)
            break

    return my_piece_present, enemy_piece_present, my_piece_location, enemy_piece_location


#print(game_state)
#print(tf.sparse.to_dense(game_state))

get_pinned_pieces(game_state, [4, 4])

#generate_legal_moves(game_state)