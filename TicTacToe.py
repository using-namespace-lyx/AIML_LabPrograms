def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def is_winner(board, player):
    row_win = any(all(board[i][j] == player for j in range(3)) for i in range(3))
    col_win = any(all(board[j][i] == player for j in range(3)) for i in range(3))
    diag1_win = all(board[i][i] == player for i in range(3))
    diag2_win = all(board[i][2 - i] == player for i in range(3))

    return row_win or col_win or diag1_win or diag2_win

def is_draw(board):
    return all(board[i][j] != ' ' for i in range(3) for j in range(3))

def dfs(board, maximizing_player):
    if is_winner(board, 'O'):
        return -1
    if is_winner(board, 'X'):
        return 1
    if is_draw(board):
        return 0

    eval_func = max if maximizing_player else min
    eval_value = float('-inf') if maximizing_player else float('inf')

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X' if maximizing_player else 'O'
                eval_value = eval_func(eval_value, dfs(board, not maximizing_player))
                board[i][j] = ' '

    return eval_value

def find_best_move(board):
    best_eval = float('-inf')
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'
                eval = dfs(board, False)
                board[i][j] = ' '

                if eval > best_eval:
                    best_eval, best_move = eval, (i, j)

    return best_move

# Main program
board = [[' ' for _ in range(3)] for _ in range(3)]

print("Initial Board:")
print_board(board)

while not is_winner(board, 'X') and not is_winner(board, 'O') and not is_draw(board):
    x, y = find_best_move(board)
    board[x][y] = 'X'

    print("Player X's move:")
    print_board(board)

    if is_winner(board, 'X') or is_draw(board):
        break

    x, y = map(int, input("Enter your move (row and column, separated by space): ").split())
    while board[x][y] != ' ':
        print("Invalid move. Try again.")
        x, y = map(int, input("Enter your move (row and column, separated by space): ").split())
    board[x][y] = 'O'

    print("Player O's move:")
    print_board(board)

if is_winner(board, 'X'):
    print("Player X wins!")
elif is_winner(board, 'O'):
    print("Player O wins!")
else:
    print("It's a draw!")
