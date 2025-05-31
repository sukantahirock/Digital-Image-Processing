# Define the Tic-Tac-Toe game board
class TicTacToe:
    def __init__(self, board):
        self.board = board
        self.current_player = 'X' if board.count('X') == board.count('O') else 'O'

    def print_board(self):
        print(" " + self.board[0] + " | " + self.board[1] + " | " + self.board[2] + " ")
        print("---|---|---")
        print(" " + self.board[3] + " | " + self.board[4] + " | " + self.board[5] + " ")
        print("---|---|---")
        print(" " + self.board[6] + " | " + self.board[7] + " | " + self.board[8] + " ")

    def is_terminal(self):
        # Check for terminal states: win, lose, or draw
        if (self.board[0] == self.board[1] == self.board[2] != ' ') or \
           (self.board[3] == self.board[4] == self.board[5] != ' ') or \
           (self.board[6] == self.board[7] == self.board[8] != ' ') or \
           (self.board[0] == self.board[3] == self.board[6] != ' ') or \
           (self.board[1] == self.board[4] == self.board[7] != ' ') or \
           (self.board[2] == self.board[5] == self.board[8] != ' ') or \
           (self.board[0] == self.board[4] == self.board[8] != ' ') or \
           (self.board[2] == self.board[4] == self.board[6] != ' ') or \
           ' ' not in self.board:
            return True
        return False

    def evaluate(self):
        # Evaluate the game state (heuristic)
        if self.is_terminal():
            if 'X' in self.board:
                return -1  # Player 'O' wins or draw
            elif 'O' in self.board:
                return 1  # Player 'X' wins or draw
            else:
                return 0  # Draw
        return None

    def get_children(self):
        # Generate possible next moves (child nodes)
        children = []
        for i in range(9):
            if self.board[i] == ' ':
                child_board = self.board.copy()
                child_board[i] = self.current_player
                children.append(child_board)
        return children

# Minimax algorithm with Alpha-Beta pruning
# Minimax algorithm with Alpha-Beta pruning
def minimax(node, depth, maximizingPlayer, alpha, beta):
    if depth == 0 or node.is_terminal():  # Base case: Terminal node
        return node.evaluate() or 0  # Return 0 if evaluation is None

    if maximizingPlayer:
        max_eval = float('-inf')
        for child in node.get_children():
            eval_score = minimax(TicTacToe(child), depth - 1, False, alpha, beta)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.get_children():
            eval_score = minimax(TicTacToe(child), depth - 1, True, alpha, beta)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval


# Function to find the best move using Minimax with Alpha-Beta pruning
def find_best_move(board):
    best_move = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    for child in board.get_children():
        value = minimax(TicTacToe(child), 5, False, alpha, beta)  # Depth set to 5 for example
        if value > best_value:
            best_value = value
            best_move = child
    return best_move

# Example usage
game_board = TicTacToe([' '] * 9)
game_board.print_board()
while not game_board.is_terminal():
    if game_board.current_player == 'X':
        move = int(input("Enter the position to place 'X' (1-9): ")) - 1
        if game_board.board[move] == ' ':
            game_board.board[move] = 'X'
            game_board.current_player = 'O'
    else:
        print("Computer's turn:")
        best_move = find_best_move(game_board)
        game_board.board = best_move
        game_board.current_player = 'X'
    game_board.print_board()

# Check game result
result = game_board.evaluate()
if result == 1:
    print("Player 'O' wins!")
elif result == -1:
    print("Player 'X' wins!")
else:
    print("It's a draw!")