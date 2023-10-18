import joblib
import numpy as np
from tqdm.auto import trange
from typing import Callable, Optional

cachedir = ".cache"
memory = joblib.Memory(cachedir, verbose=0)


def bootstrap_sample(counts: np.ndarray) -> np.ndarray:
    """Draw a bootstrapped sample of papers

    `counts` is an array detailing the number of papers per journal (or subject
    category). We draw a sample with replacement from those journals, and return
    an 'alternative' count vector.

    """
    #  First, transform into a paper array where each value denotes that paper's
    # journal.
    n_counts = counts.size
    papers = np.repeat(np.arange(n_counts), counts.astype("int64"))

    # Draw sample from papers
    sample = np.random.choice(papers, papers.size)

    # Count number of papers in each journal
    return np.bincount(sample, minlength=n_counts)


def bootstrap_replication(
    func: Callable,
    coords: np.ndarray,
    counts: np.ndarray,
    counts2: Optional[np.ndarray] = None,
    **kwargs,
):
    sample = bootstrap_sample(counts)
    if counts2 is None:
        return func(coords, sample, **kwargs)

    sample2 = bootstrap_sample(counts2)
    return func(coords, sample, sample2, **kwargs)


@memory.cache
def bootstrap_samples(
    func: Callable,
    coords: np.ndarray,
    counts: np.ndarray,
    counts2: Optional[np.ndarray] = None,
    /,
    num_samples: int = 1000,
    **kwargs,
):
    samples = np.empty((num_samples, coords.shape[1]))

    for i in trange(num_samples):
        samples[i] = bootstrap_replication(func, coords, counts, counts2, **kwargs)
    return samples


def confidence_interval(data, alpha=0.05):
    num_samples = len(data)
    stat = np.sort(data)
    return (
        stat[int((alpha / 2) * num_samples)],
        stat[int((1 - alpha / 2) * num_samples)],
    )
