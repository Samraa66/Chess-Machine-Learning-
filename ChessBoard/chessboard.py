import numpy as np 

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
