import numpy as np 

# Define the functions to generate initial white piece positions
def generate_initial_white_pawns():

    white_pawns = np.zeros((8,8), dtype=int)
    white_pawns[6, :] = 1
    return white_pawns
white_pawns = generate_initial_white_pawns()

def generate_initial_white_rooks():
    white_rooks = np.zeros((8,8), dtype =int)
    white_rooks[7,0] = 1
    white_rooks[7,7] = 1
    return white_rooks
white_rooks = generate_initial_white_rooks()

def generate_initial_white_knights():
    white_knights = np.zeros((8,8), dtype = int)
    white_knights[7,1] = 1
    white_knights[7,6]= 1
    return white_knights 
white_knights = generate_initial_white_knights

def generate_initial_white_bishops():
    white_bishops = np.zeros((8,8), dtype = int)
    white_bishops[7,2] = 1
    white_bishops[7,5] =1 
    return white_bishops
white_bishops = generate_initial_white_bishops

def generate_initial_white_queen():
    white_queen = np.zeros((8,8), dtype = int)
    white_queen[7,3] = 1
    
    return white_queen
white_queen = generate_initial_white_queen

def generate_initial_white_king():
    white_king = np.zeros((8,8), dtype = int)
    white_king[7,4] = 1
    
    return white_king
white_king = generate_initial_white_king

# Define the functions to generate initial black piece positions
def generate_initial_black_pawns():
    black_pawns = np.zeros((8, 8), dtype=int)
    black_pawns[1, :] = 1
    return black_pawns


def generate_initial_black_rooks():
    black_rooks = np.zeros((8, 8), dtype=int)
    black_rooks[0, 0] = 1
    black_rooks[0, 7] = 1
    return black_rooks

def generate_initial_black_knights():
    black_knights = np.zeros((8, 8), dtype=int)
    black_knights[0, 1] = 1
    black_knights[0, 6] = 1
    return black_knights

def generate_initial_black_bishops():
    black_bishops = np.zeros((8, 8), dtype=int)
    black_bishops[0, 2] = 1
    black_bishops[0, 5] = 1
    return black_bishops

def generate_initial_black_queen():
    black_queen = np.zeros((8, 8), dtype=int)
    black_queen[0, 3] = 1
    return black_queen

def generate_initial_black_king():
    black_king = np.zeros((8, 8), dtype=int)
    black_king[0, 4] = 1
    return black_king

# Generate initial piece positions
black_pawns = generate_initial_black_pawns()
black_rooks = generate_initial_black_rooks()
black_knights = generate_initial_black_knights()
black_bishops = generate_initial_black_bishops()
black_queen = generate_initial_black_queen()
black_king = generate_initial_black_king()

print(black_pawns)

