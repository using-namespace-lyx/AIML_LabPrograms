def f(x):
    # The function to be maximized
    return -(x - 2)**2 + 5

def hill_climbing(initial_x, step_size, max_iterations):
    current_x = initial_x

    for _ in range(max_iterations):
        current_value = f(current_x)
        next_x = current_x + step_size
        next_value = f(next_x)

        if next_value > current_value:
            current_x = next_x
        else:
            break  # Break if further movement doesn't increase the value

    return current_x

# Main program
initial_x_value = 0.0
step_size = 0.1
max_iterations = 50

result = hill_climbing(initial_x_value, step_size, max_iterations)

print(f"Maximum value found at x = {result}, f(x) = {f(result)}")
