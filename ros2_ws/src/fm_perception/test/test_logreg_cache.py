"""
Purpose: Pin the behaviour of the fitted-probe cache in dinov2_terrain_node.
         Refitting the linear probe from the feature cache cost 10.3 s of every
         launch on the Pi (measured 2026-07-29) to reproduce a byte-identical
         result, which is a fifth of the whole startup budget. The cache must
         reuse that fit, and must refuse to reuse it when anything that shaped
         it changes -- a stale probe silently classifying terrain would be far
         worse than a slow boot.
Inputs:  None; builds a small synthetic feature cache in a temp dir.
Outputs: pytest results.
How to run:
    cd ros2_ws && python3 -m pytest src/fm_perception/test/test_logreg_cache.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "fm_perception")
)

from dinov2_terrain_node import fit_or_load_logreg, logreg_cache_path


@pytest.fixture
def feature_cache(tmp_path):
    """A tiny stand-in for dinov2_reg_small_train_1000shot.npz."""
    rng = np.random.default_rng(0)
    n_per_class = 12
    feats, labels = [], []
    for c in range(3):
        centre = rng.normal(size=384)
        for _ in range(n_per_class):
            feats.append(centre + 0.1 * rng.normal(size=384))
            labels.append(c)
    path = tmp_path / "features.npz"
    np.savez(path, feats=np.array(feats, dtype=np.float32),
             labels=np.array(labels))
    return str(path)


def test_first_call_fits_and_second_call_loads(feature_cache):
    clf_a, fitted_a = fit_or_load_logreg(feature_cache, n_shot=10,
                                         class_weight_balanced=False)
    assert fitted_a is True, "nothing was cached yet, so it had to fit"
    assert os.path.exists(logreg_cache_path(feature_cache))

    clf_b, fitted_b = fit_or_load_logreg(feature_cache, n_shot=10,
                                         class_weight_balanced=False)
    assert fitted_b is False, "the second call must reuse the cached fit"
    # Same probe, not merely a probe with the same shape.
    np.testing.assert_allclose(clf_a.coef_, clf_b.coef_, rtol=0, atol=0)
    np.testing.assert_allclose(clf_a.intercept_, clf_b.intercept_,
                               rtol=0, atol=0)


def test_a_different_n_shot_refits(feature_cache):
    fit_or_load_logreg(feature_cache, n_shot=10, class_weight_balanced=False)
    _, fitted = fit_or_load_logreg(feature_cache, n_shot=5,
                                   class_weight_balanced=False)
    assert fitted is True


def test_a_different_class_weight_refits(feature_cache):
    fit_or_load_logreg(feature_cache, n_shot=10, class_weight_balanced=False)
    _, fitted = fit_or_load_logreg(feature_cache, n_shot=10,
                                   class_weight_balanced=True)
    assert fitted is True


def test_a_changed_feature_cache_refits(feature_cache):
    fit_or_load_logreg(feature_cache, n_shot=10, class_weight_balanced=False)
    # Regenerate the features. The probe on disk now describes data that no
    # longer exists, which is the case that has to be caught.
    rng = np.random.default_rng(99)
    np.savez(feature_cache,
             feats=rng.normal(size=(36, 384)).astype(np.float32),
             labels=np.repeat([0, 1, 2], 12))
    _, fitted = fit_or_load_logreg(feature_cache, n_shot=10,
                                   class_weight_balanced=False)
    assert fitted is True


def test_a_corrupt_cache_refits_instead_of_raising(feature_cache):
    fit_or_load_logreg(feature_cache, n_shot=10, class_weight_balanced=False)
    with open(logreg_cache_path(feature_cache), "wb") as fh:
        fh.write(b"not an npz")
    clf, fitted = fit_or_load_logreg(feature_cache, n_shot=10,
                                     class_weight_balanced=False)
    assert fitted is True
    assert clf is not None


def test_a_cache_that_cannot_be_written_still_returns_a_probe(feature_cache,
                                                              monkeypatch):
    # The Pi runs this out of a bind-mounted workspace, and a node that
    # refuses to boot because it could not write an optimisation would be a
    # worse failure than the slow boot the cache exists to avoid.
    import dinov2_terrain_node as node_mod

    def refuse(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(node_mod.np, "savez", refuse)
    clf, fitted = fit_or_load_logreg(feature_cache, n_shot=10,
                                     class_weight_balanced=False)
    assert clf is not None
    assert fitted is True
