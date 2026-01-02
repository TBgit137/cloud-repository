# Asalto manual play example
# JC4004 Computational Intelligence 2025-26

class Player:
  
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

  # =================================================
  # Play one move as a fox
  def play_officer(self, board):

    # First, print the current board
    self.print_board(board)

    # Get player input for new officer position
    cont = True
    print("Officer plays next!")

    officer_pos = [int(i) for i in input("Give the position of the officer you want to move (row,column): ").split(',')]

    move = [officer_pos]

    while cont:      
      new_pos = [int(i) for i in input("Give the new position for the officer (row,column): ").split(',')]
      move.append(new_pos)
      if abs(move[-1][0]-move[-2][0]) > 1 or abs(move[-1][1]-move[-2][1]) > 1:
        inp = input("Do you want to make another move [y/N]? ")
        if inp != 'y' and inp != 'Y':
          cont = False
      else:
        cont = False

    return move

  # =================================================
  # Play one move as a goose
  def play_rebel(self, board):

    # First, print the current board
    self.print_board(board)

    # Get goose start position
    print("Rebel plays next!")
    rebel_pos = [int(i) for i in input("Give the position of the rebel you want to move (row,column): ").split(',')]
    move = [rebel_pos]

    # Get player input for goose next position
    new_pos = [int(i) for i in input("Give the new position for the rebel (row,column): ").split(',')]
    move.append(new_pos)

    return move

# ==== End of file