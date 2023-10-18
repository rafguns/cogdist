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


@pytest.mark.parametrize("coords,counts", zip(coords2d, counts))
def test_bootstrap_replication(coords, counts):
    (coords_res, counts_res), _ = cogdist.bootstrap_replication(dummy, coords, counts)

    assert (coords_res == coords).all()
    assert counts_res.sum() == counts.sum()


@pytest.mark.parametrize("coords, counts", zip(coords2d, counts))
def test_bootstrap_replication_kwargs(counts, coords):
    args, kwargs = cogdist.bootstrap_replication(
        dummy, coords, counts, y="foo", x="bar"
    )

    assert (args[0] == coords).all()
    assert args[1].sum() == counts.sum()
    assert kwargs == {"x": "bar", "y": "foo"}


@pytest.mark.parametrize("coords,counts1,counts2", zip(coords2d, counts, counts))
def test_bootstrap_replication2(coords, counts1, counts2):
    (coords_res, counts1_res, counts2_res), _ = cogdist.bootstrap_replication(
        dummy, coords, counts1, counts2
    )

    assert (coords_res == coords).all()
    assert counts1_res.sum() == counts1.sum()
    assert counts2_res.sum() == counts2.sum()
