import numpy as np
import pytest

import cogdist

def test_bootstrap_sample():
    counts = np.array([5, 0, 4, 10, 0, 0])
    sample_counts = cogdist.bootstrap_sample(counts)

    assert sample_counts.sum() == counts.sum()
    assert (np.where(sample_counts == 0)[0] == np.where(counts == 0)[0]).all()
