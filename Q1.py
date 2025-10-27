def softmax_classify(W, b, X) -> list[int]:
    """
    Linear multi-class classification without external libraries.
    For each input x in X, computes logits = W @ x + b and returns argmax class index.
    Tie-breaker: choose the smallest class index.
    Shapes:
      - W: C x D (list of lists)
      - b: length C (list)
      - X: M x D (list of lists)
    """
    if not W or not X:
        return []
    C = len(W)
    D = len(W[0])
    # Basic validation (optional, but keeps behavior predictable)
    # Assume inputs are well-formed per prompt.
    out = []
    for x in X:
        # compute logits
        logits = [0.0] * C
        for c in range(C):
            s = b[c] if b else 0.0
            row = W[c]
            # dot product W[c] · x
            for j in range(D):
                s += row[j] * x[j]
            logits[c] = s
        # argmax with smallest index on ties
        best_c = 0
        best_v = logits[0]
        for c in range(1, C):
            v = logits[c]
            if v > best_v or (v == best_v and c < best_c):
                best_c, best_v = c, v
        out.append(best_c)
    return out
