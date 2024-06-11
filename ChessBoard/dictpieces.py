
# layer:((white, black),(move_update))

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
