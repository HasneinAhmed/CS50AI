"""
Tic Tac Toe Player
"""

import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY] * 3 for _ in range(3)]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if terminal(board):
        return None

    turns = sum(row.count(EMPTY) for row in board)
    # Since X always starts, an odd number of empty spots means X's turn
    return X if turns % 2 != 0 else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    open_spots = set()
    for r in range(3):
        for c in range(3):
            if board[r][c] is EMPTY:
                open_spots.add((r, c))
    return open_spots


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    row, col = action
    if not (0 <= row <= 2 and 0 <= col <= 2):
        raise Exception("Move out of bounds.")
    if board[row][col] is not EMPTY:
        raise Exception("Cell is already taken.")

    board_copy = copy.deepcopy(board)
    board_copy[row][col] = player(board)
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows and columns in a single loop
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not EMPTY:
            return board[0][i]

    # Check main diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    return all(cell is not EMPTY for row in board for cell in row)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    game_winner = winner(board)
    if game_winner == X:
        return 1
    if game_winner == O:
        return -1
    return 0


def _max_score(board):
    if terminal(board):
        return utility(board)
    highest = -float("inf")
    for act in actions(board):
        highest = max(highest, _min_score(result(board, act)))
    return highest


def _min_score(board):
    if terminal(board):
        return utility(board)
    lowest = float("inf")
    for act in actions(board):
        lowest = min(lowest, _max_score(result(board, act)))
    return lowest


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    # Instant check for empty board to skip full tree search on turn 1
    if board == initial_state():
        return (1, 1)

    turn = player(board)

    if turn == X:
        top_val = -float("inf")
        best_move = None
        for act in actions(board):
            score = _min_score(result(board, act))
            if score > top_val:
                top_val = score
                best_move = act
        return best_move
    else:
        low_val = float("inf")
        best_move = None
        for act in actions(board):
            score = _max_score(result(board, act))
            if score < low_val:
                low_val = score
                best_move = act
        return best_move
