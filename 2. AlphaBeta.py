MAX, MIN = 1000, -1000

def minmax(depth, nodeIndex, maximizing_player, values, alpha, beta):
    if depth == 3:
        return values[nodeIndex]

    if maximizing_player:
        best = MIN
        for i in range(0, 2):
            val = minmax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            print(f"At depth {depth}, node {nodeIndex}, alpha = {alpha}, beta = {beta}")
            if beta <= alpha:
                print("Pruned!")
                break
        return best
    else:
        best = MAX
        for i in range(0, 2):
            val = minmax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            print(f"At depth {depth}, node {nodeIndex}, alpha = {alpha}, beta = {beta}")
            if beta <= alpha:
                print("Pruned!")
                break
        return best

if __name__ == '__main__':
    values = list(map(int, input("Enter the leaf nodes of a game tree of depth 3: ").split()))
    print("The optimal value is ", minmax(0, 0, True, values, MIN, MAX))
