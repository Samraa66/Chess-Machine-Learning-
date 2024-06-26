import tensorflow as tf

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
    56: (((1), (1)), (-2, 1)),
    57: (((1), (1)), (-1, 2)),
    58: (((1), (1)), (1, 2)),
    59: (((1), (1)), (2, 1)),
    60: (((1), (1)), (2, -1)),
    61: (((1), (1)), (1, -2)),
    62: (((1), (1)), (-1, -2)),
    63: (((1), (1)), (-2, -1)),
    64: (((1), (1)), ((-1, -1), (1, -2))),
    65: (((0), (0)), ((-1, 0), (1, 0))),
    66: (((0), (0)), ((-1, 1), (1, 1))),
    67: (((0), (0)), ((-1, -1), (1, -1))),
    68: (((0), (0)), ((-1, 0), (1, 0))),
    69: (((0), (0)), ((-1, 1), (1, 1))),
    70: (((0), (0)), ((-1, -1), (1, -1))),
    71: (((0), (0)), ((-1, 0), (1, 0))),
    72: (((0), (0)), ((-1, 1), (1, 1))),
}

print(pieces_and_moves)


# 5 move history gamestate
def generate_legal_moves(game_state):
  # first we check tensor for currrent players turn
  turn = game_state[0, 0, 70]  # should be zero or one

  # finding pieces
  if turn == 0:
    my_pieces = game_state[:, :, 56:61]
    your_pieces = game_state[:, :, 62:67]
  else:
    my_pieces = game_state[:, :, 62:67]
    your_pieces = game_state[:, :, 56:61]
  # my positions will contain locations of pieces on board
  my_positions = my_pieces.indices  # potentially need to do .numpy()
  your_positions = your_pieces.indices  # potentially need to do .numpy()
  # idea for pins, initialize all pieces with allowed squares of all 8x8 board, except for the first piece in one of 9 radial directions from king that right after has a piece of the opposite color that attacks in this direction, that piece is pinned and can only move in this specified direction.

  # legal moves should be in the form piece_x, piece_y, layer
  # these are initializing the return var.
  legal_moves = []
  game_states = []
  piece_types = []
  for i in range(73):
    # for each layer we want to check if pieces can move in layer direction

    # get both pieces that make move of type i, and displacement of move i
    piece_and_move = pieces_and_moves[i]
    # grab piece types that would make move i (w,b)
    piece_for_this_kind_of_move = piece_and_move[0]
    # for each of my pieces on the board
    for piece_x, piece_y, piece_type in my_positions:
      if piece_type in piece_for_this_kind_of_move[turn]:
        # here we know we can move this piece in this direction so test potential location
        potential_piece_x, potential_piece_y = (piece_x,
                                                piece_y) + piece_and_move[1]
        # first check if new location is out of bounds or my piece is blocking
        within_board = (0 <= potential_piece_x <= 8
                        and 0 <= potential_piece_y <= 8)
        piece_present = False
        for pos in my_positions:
          if potential_piece_x == pos[0] and potential_piece_y == pos[1]:
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
          for pos in your_positions:
            if potential_piece_x == pos[0] and potential_piece_y == pos[1]:
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
            new_game_state[piece_x, piece_y, piece_type] = 0
            new_game_state[potential_piece_x, potential_piece_y,
                           piece_type] = 1

            # if there was a captured remove the piece
            # python none same as false? lets check by debugging if this will work as intended
            if piece_capture and piece_location is not None:
              new_game_state[piece_location[0], piece_location[1],
                             piece_location[2]] = 0

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

