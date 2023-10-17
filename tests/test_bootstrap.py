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


def dummy(*args, **kwargs):
    return args, kwargs


@pytest.mark.parametrize("counts,coords", zip(counts, coords2d))
def test_bootstrap_replication(counts, coords):
    (counts_res, coords_res), _ = cogdist.bootstrap_replication(counts, coords, dummy)

    assert counts_res.sum() == counts.sum()
    assert (coords_res == coords).all()


@pytest.mark.parametrize("counts,coords", zip(counts, coords2d))
def test_bootstrap_replication_args_kwargs(counts, coords):
    args, kwargs = cogdist.bootstrap_replication(counts, coords, dummy, "foo", x="bar")

    assert kwargs == {"x": "bar"}
    assert args[2] ==  "foo"
    assert args[0].sum() == counts.sum()
    assert (args[1] == coords).all()
