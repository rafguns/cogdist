import typing
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

__all__ = [
    "as_square_matrix",
    "barycenter",
    "euclidean_distance",
    "sa_vector",
    "weighted_cosine",
]


def ensure_symmetric(M: np.ndarray) -> None:
    """Ensure that M is a square, symmetric matrix"""
    # XXX This does only a very basic check. It should be tested/profiled if a
    # more precise check is not too expensive computationally. Should be
    # sufficient for now.
    m, n = M.shape
    if m != n:
        raise ValueError("M is not square!")


def barycenter(counts: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Calculate the barycenter for the given counts and coordinates

    Arguments
    ---------
    counts : an iterable
         Counts for each point (this should have length m)
    coords : an m * n numpy.ndarray
         Array containing coordinates for m points in n dimensions

     Returns
     -------
     barycenter_coords : numpy.ndarray of length n
         Coordinates of the barycenter (n dimensions)

    """
    m, _ = coords.shape

    if len(counts) != m:
        raise ValueError(
            "'counts' should have the same number of items "
            "(now: {}) as rows of 'coords' (now: {})".format(len(counts), m)
        )

    # Transposing because of broadcasting rules
    a = coords.T * counts
    return a.sum(axis=1) / sum(counts)


def sa_vector(
    counts: np.ndarray, sim_matrix: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Calculate the similarity adapted vector for the given counts and
    similarity matrix

    Arguments
    ---------
    counts : an iterable
        Counts for each point (this should have length m)
    sim_matrix : a symmetric numpy.ndarray
        Similarity matrix
    normalize : True|False
        Whether to normalize cordinates

    Returns
    -------
    vector : numpy.ndarray of length n
        Similarity adapted vector

    """
    ensure_symmetric(sim_matrix)

    if len(counts) != len(sim_matrix):
        raise ValueError(
            "'counts' should have the same number of items "
            "(now: {}) as rows of similarity matrix (now: {})".format(
                len(counts), len(sim_matrix)
            )
        )

    raw_sa_vector = (sim_matrix * counts).sum(axis=1)
    return raw_sa_vector / raw_sa_vector.sum() if normalize else raw_sa_vector


def weighted_cosine(u: np.ndarray, v: np.ndarray, sim_matrix: np.ndarray) -> np.ndarray:
    ensure_symmetric(sim_matrix)
    if len(u) != len(v) != len(sim_matrix):
        raise ValueError("Vectors or similarity matrix of different length.")

    return (u @ sim_matrix @ v) / np.sqrt((u @ sim_matrix @ u) * (v @ sim_matrix @ v))


def as_square_matrix(
    M: np.ndarray, compare_by: typing.Union[str, typing.Callable], *args, **kwargs
) -> np.ndarray:
    """Calculate pairwise distances or similarities

    Arguments
    ---------
    M : an m * n numpy.ndarray
        m observations in n-dimensional space
    compare_by : a function to compare row vectors or a string
        The function takes at least two vectors. Extra arguments are supplied
        as args or kwargs. If string, this is passed to
        `scipy.spatial.distance.pdist` (e.g., 'euclidean', 'minkowski' etc.).

    """
    if isinstance(M, pd.DataFrame):
        idx = M.index
        M = M.as_matrix()
    else:
        idx = None

    if callable(compare_by):
        n = len(M)
        S = np.empty((n, n))
        # Also calculate diagonal, since we don't know if this is a distance or
        # similarity measure
        for i in range(n):
            for j in range(i, n):
                S[i, j] = S[j, i] = compare_by(M[i], M[j], *args, **kwargs)
    else:
        # Assume compare_by is a label like 'euclidean'
        S = squareform(pdist(M, compare_by))

    if isinstance(idx, pd.Index):
        S = pd.DataFrame(S, index=idx, columns=idx)

    return S


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> typing.Union[float, np.ndarray]:
    """Determine Euclidean distance between two vectors/arrays

    If a and b are 1D vectors, return Euclidean distance as a float.
    If a and b are 2D m * n arrays, compare then row by row and
    return 1D array of Euclidean distances (length m).

    """
    if a.shape != b.shape:
        raise ValueError("a and b should be of same shape")

    ndims = len(a.shape)
    if ndims > 2:
        raise ValueError("Only one- and two-dimensional arrays are supported")
    # For 1D vectors, axis is 0. For 2D vectors, axis is 1.
    return np.linalg.norm(a - b, axis=ndims - 1)
