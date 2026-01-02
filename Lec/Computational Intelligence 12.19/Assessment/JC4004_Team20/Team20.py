# Asalto manual play example
# JC4004 Computational Intelligence 2025-26

from RebelAI import get_best_rebel_move
from OfficerAI import get_best_officer_move

# Global configuration
USE_ITERATIVE_DEEPENING = False

class Player:
    def __init__(self):
        # Initialization (load models here if needed)
        pass

    def play_rebel(self, board):
        # Call RebelAI logic, pass configuration
        return get_best_rebel_move(board, use_iterative=USE_ITERATIVE_DEEPENING)

    def play_officer(self, board):
        # Call OfficerAI logic, pass configuration
        return get_best_officer_move(board, use_iterative=USE_ITERATIVE_DEEPENING)

    # =================================================
    # Print the board
    @staticmethod
    def print_board(board):
        # Prints the current board
        print('')
        print('  0 1 2 3 4 5 6')
        for i in range(len(board)):
            txt = str(i) + ' '
            for j in range(len(board[i])):
                if i < 3 and 1 < j < 5:
                    if board[i][j] == 'O':
                        txt += "\033[97m" + board[i][j] + " \033[00m"
                    elif board[i][j] == 'R':
                        txt += "\033[96m" + board[i][j] + " \033[00m"
                    else:
                        txt += "\033[37m" + board[i][j] + " \033[00m"
                else:
                    if board[i][j] == 'O':
                        txt += "\033[37;1m" + board[i][j] + " \033[00m"
                    elif board[i][j] == 'R':
                        txt += "\033[36m" + board[i][j] + " \033[00m"
                    else:
                        txt += "\033[90m" + board[i][j] + " \033[00m"
            print(txt)
        print('')

# ==== End of file