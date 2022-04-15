import numpy as np
import pytest

import cogdist

counts = [np.array([5, 0, 4, 10, 0])]
coords2d = [np.array([[0.1, 0.5], [0.9, 0.4], [0.7, 0.1], [0.0, 1.0], [0.2, 0.55]])]

@pytest.mark.parametrize("counts", counts)
def test_bootstrap_sample(counts):
    sample_counts = cogdist.bootstrap_sample(counts)

    assert sample_counts.sum() == counts.sum()
    assert (np.where(sample_counts == 0)[0] == np.where(counts == 0)[0]).all()
