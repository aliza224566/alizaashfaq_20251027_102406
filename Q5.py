def perceptron_epoch(X, y, w, b, lr) -> [list[float], float]:
    """
    Run a single epoch of the binary perceptron over data (in order).
    Labels are -1 or +1. Update rule:
      if y_i * (w·x + b) <= 0:
          w = w + lr*y_i*x
          b = b + lr*y_i
    Returns [w, b].
    """
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    # Ensure mutable copies if needed
    w = list(w)
    b = float(b)

    for xi, yi in zip(X, y):
        margin = yi * (dot(w, xi) + b)
        if margin <= 0:
            # w update
            for j in range(len(w)):
                w[j] += lr * yi * xi[j]
            # b update
            b += lr * yi
    return [w, b]
