def confusion_matrix(y_true, y_pred, labels) -> list[list[int]]:
    """
    Compute confusion matrix.
    Rows: true labels; Columns: predicted labels; both ordered as in `labels`.
    Returns a 2D list of counts.
    """
    n = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    # initialize matrix
    M = [[0 for _ in range(n)] for _ in range(n)]
    for yt, yp in zip(y_true, y_pred):
        if yt in index and yp in index:
            i = index[yt]
            j = index[yp]
            M[i][j] += 1
        # If a label isn't in `labels`, it's ignored per robust handling.
    return M
