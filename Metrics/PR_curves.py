import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def interpolated_pr_ir(
        query,
        documents,
        relevant_indices,
        recall_grid=None
    ):
    """
    Compute interpolated precision and recall for a single IR query.

    Parameters
    ----------
    query : str
        Query string.
    documents : list of str
        Corpus documents.
    relevant_indices : list or set of int
        Indices of documents that are relevant to the query.
    recall_grid : array-like or None
        Recall points at which to compute the interpolated curve.
        If None, use {0.0, 0.1, ..., 1.0}.

    Returns
    -------
    recall_grid : ndarray
    interpolated_precision : ndarray
    raw_recalls : ndarray
    raw_precisions : ndarray
    ranked_indices : ndarray
    """

    # Fit TF-IDF on the corpus and the query
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents + [query])
    X_docs = X[:-1]
    X_query = X[-1]

    # Cosine similarity scores
    scores = cosine_similarity(X_docs, X_query).ravel()

    # Ranking
    ranked_indices = np.argsort(scores)[::-1]
    ranked_relevance = np.array([i in relevant_indices for i in ranked_indices])

    num_relevant = len(relevant_indices)
    if num_relevant == 0:
        raise ValueError("No relevant documents were provided.")

    # Accumulate TP counts over the ranking
    tp_cum = np.cumsum(ranked_relevance)

    # Raw precision and recall arrays over ranks
    ranks = np.arange(1, len(documents) + 1)
    raw_precisions = tp_cum / ranks
    raw_recalls = tp_cum / num_relevant

    # Interpolation grid
    if recall_grid is None:
        recall_grid = np.linspace(0.0, 1.0, 11)

    interpolated_precision = np.zeros_like(recall_grid)

    for i, r in enumerate(recall_grid):
        mask = raw_recalls >= r
        interpolated_precision[i] = np.max(raw_precisions[mask]) if np.any(mask) else 0.0

    return recall_grid, interpolated_precision, raw_recalls, raw_precisions, ranked_indices

import matplotlib.pyplot as plt

def plot_ir_pr_curves(
        raw_recalls,
        raw_precisions,
        recall_grid,
        interpolated_precision,
        title=None
    ):
    """
    Plot raw and interpolated precision–recall curves.

    Parameters
    ----------
    raw_recalls : array-like
        Recall values from the ranking.
    raw_precisions : array-like
        Precision values from the ranking.
    recall_grid : array-like
        Recall grid used for interpolation.
    interpolated_precision : array-like
        Interpolated precision values.
    title : str or None
        Optional plot title.
    """

    plt.figure(figsize=(7, 5))

    # Raw PR curve (step-like)
    plt.plot(raw_recalls, raw_precisions, label="Raw PR curve", marker="o")

    # Interpolated curve
    plt.plot(recall_grid, interpolated_precision, label="Interpolated PR", marker="s")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.grid(True)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()




query = "deep learning"
documents = [
    "statistical learning theory",              # non-relevant
    "deep learning applications",               # relevant
    "classical probability models",             # non-relevant
    "transformers and deep representations",    # relevant
    "regression methods",                       # non-relevant
    "advanced neural architectures",            # relevant
]

# Relevant: indices: 1, 3, 5
relevant = {1, 3, 5}

grid, iprec, raw_rec, raw_prec, ranking = interpolated_pr_ir(
    query,
    documents,
    relevant
)
plot_ir_pr_curves(raw_rec, raw_prec, grid, iprec, title="Interpolated PR (non-monotone example)")


print(grid)
print(iprec)
print(ranking)


