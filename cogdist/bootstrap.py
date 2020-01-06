import joblib
import numpy as np
from tqdm import trange

memory = joblib.Memory(cachedir='.cache', verbose=0)


def bootstrap_sample(counts):
    # counts is an array detailing the number of papers per journal. First,
    # transform it into a paper array where each value denotes that paper's
    # journal.
    n_counts = counts.size
    papers = np.repeat(np.arange(n_counts), counts.astype('int64'))

    # Draw sample from papers
    sample = np.random.choice(papers, papers.size)

    # Count number of papers in each journal
    return np.bincount(sample, minlength=n_counts)


def bootstrap_replication(counts, coords, func, *args, **kwargs):
    sample = bootstrap_sample(counts)

    return func(sample, coords, *args, **kwargs)


@memory.cache
def bootstrap_samples(counts, coords, num_samples, func, *args, **kwargs):
    samples = np.empty((num_samples, coords.shape[1]))

    for i in trange(num_samples):
        samples[i] = bootstrap_replication(counts, coords, func, *args,
                                           **kwargs)
    return samples


def bootstrap_replication2(counts1, counts2, coords, func, *args, **kwargs):
    sample1 = bootstrap_sample(counts1)
    sample2 = bootstrap_sample(counts2)

    return func(sample1, sample2, coords, *args, **kwargs)


# XXX ugly duplication here. Seems like it should be possible to subsume the
# barycenter and SAPV CIs here as well, if we move more stuff to a separate
# function (the func to be called)
@memory.cache
def bootstrap_samples2(counts1, counts2,
                       coords, num_samples, func, *args, **kwargs):
    samples = np.empty(num_samples)

    for i in trange(num_samples):
        samples[i] = bootstrap_replication2(counts1, counts2, coords, func,
                                            *args, **kwargs)
    return samples


def confidence_interval(data, alpha=0.05):
    num_samples = len(data)
    stat = np.sort(data)
    return (stat[int((alpha / 2) * num_samples)],
            stat[int((1 - alpha / 2) * num_samples)])
