def top_k_cosine(query, docs, k) -> list[int]:
    """
    Return indices of top-k documents by cosine similarity to the query vector.
    Cosine similarity = (q·d)/(||q||·||d||). If either vector has zero norm,
    treat similarity as 0. Ties broken by smaller index. If k > len(docs),
    return up to len(docs) indices.
    """
    import math

    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    def norm(v):
        return math.sqrt(sum(x*x for x in v))

    qn = norm(query)
    sims = []
    for idx, d in enumerate(docs):
        dn = norm(d)
        if qn == 0 or dn == 0:
            s = 0.0
        else:
            s = dot(query, d) / (qn * dn)
        sims.append(( -s, idx ))  # negative for descending sort, store index for tie-break

    sims.sort()  # sorts by (-sim, idx): higher sim first; ties -> smaller idx first
    k = min(k, len(docs))
    return [idx for _, idx in sims[:k]]
