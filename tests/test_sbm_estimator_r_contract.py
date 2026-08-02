from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pybspcov.estimators as estimator_module
from pybspcov import SBMSPCov
from pybspcov.kernels import BMState, SBMCompactStructure, SBMPackedChainResult
from pybspcov.kernels.bm import pack_lower_triangle_column_major

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "r" / "bspcov-1.0.3"


def _load_csv(name: str) -> np.ndarray:
    return np.atleast_2d(
        np.genfromtxt(
            FIXTURE_DIR / name,
            delimiter=",",
            dtype=np.float64,
            missing_values="NA",
            filling_values=np.nan,
        )
    )


class _RunnerCall(TypedDict):
    keys: jax.Array
    states: BMState
    scatter: jax.Array
    n_observations: jax.Array
    a: jax.Array
    b: jax.Array
    diagonal_rate: jax.Array
    tau1sq: jax.Array
    structure: SBMCompactStructure
    burnin: int
    n_samples: int


class _CutoffCall(TypedDict):
    key: jax.Array
    n_observations: int
    correlation: float | jax.Array
    false_negative_rate: float | jax.Array
    n_simulations: int
    dtype: str


def _require_x64() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("R double-precision parity requires JAX_ENABLE_X64=1")


def _load_metadata() -> dict[str, Any]:
    with (FIXTURE_DIR / "sbm_screening_metadata.json").open() as stream:
        return cast(dict[str, Any], json.load(stream))


def _accepted_result(
    keys: jax.Array,
    states: BMState,
    *,
    burnin: int,
    n_samples: int,
    covariance: jax.Array | None = None,
    phi: jax.Array | None = None,
    accepted: bool = True,
) -> SBMPackedChainResult:
    chain_count = keys.shape[0]
    if covariance is None:
        covariance = pack_lower_triangle_column_major(states.covariance)
    else:
        covariance = jnp.broadcast_to(covariance, (chain_count, covariance.shape[-1]))
    if phi is None:
        phi = pack_lower_triangle_column_major(states.phi)
    else:
        phi = jnp.broadcast_to(phi, (chain_count, phi.shape[-1]))
    return SBMPackedChainResult(
        final_state=states,
        covariance=jnp.broadcast_to(
            covariance[:, None, :],
            (chain_count, n_samples, covariance.shape[-1]),
        ),
        phi=jnp.broadcast_to(
            phi[:, None, :],
            (chain_count, n_samples, phi.shape[-1]),
        ),
        accepted=jnp.full(
            (chain_count, burnin + n_samples),
            accepted,
            dtype=jnp.bool_,
        ),
    )


def _install_passthrough_runner(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[_RunnerCall] | None = None,
) -> None:
    def runner(
        keys: jax.Array,
        states: BMState,
        scatter: jax.Array,
        n_observations: jax.Array,
        a: jax.Array,
        b: jax.Array,
        diagonal_rate: jax.Array,
        tau1sq: jax.Array,
        structure: SBMCompactStructure,
        *,
        burnin: int,
        n_samples: int,
    ) -> SBMPackedChainResult:
        if calls is not None:
            calls.append(
                {
                    "keys": keys,
                    "states": states,
                    "scatter": scatter,
                    "n_observations": n_observations,
                    "a": a,
                    "b": b,
                    "diagonal_rate": diagonal_rate,
                    "tau1sq": tau1sq,
                    "structure": structure,
                    "burnin": burnin,
                    "n_samples": n_samples,
                }
            )
        return _accepted_result(
            keys,
            states,
            burnin=burnin,
            n_samples=n_samples,
        )

    monkeypatch.setattr(estimator_module, "_compile_sbm_chains", lambda: runner)


def _small_x(dtype: jnp.dtype = jnp.float64) -> jax.Array:
    return jnp.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 1.0],
            [-1.0, 3.0, 2.0],
            [0.0, 5.0, 4.0],
        ],
        dtype=dtype,
    )


def test_sbmspcov_wires_r_defaults_and_isolates_shared_screening_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_x64()
    runner_calls: list[_RunnerCall] = []
    cutoff_calls: list[_CutoffCall] = []
    structure_calls: list[SBMCompactStructure] = []
    active_mask = jnp.asarray(
        [[False, True, False], [True, False, True], [False, True, False]]
    )
    screening_cutoff = jnp.asarray(0.375, dtype=jnp.float64)
    original_prepare = cast(
        Callable[[jax.Array, jax.Array], SBMCompactStructure],
        vars(estimator_module)["prepare_sbm_compact_structure"],
    )

    def cutoff_spy(
        key: jax.Array,
        *,
        n_observations: int,
        correlation: float | jax.Array,
        false_negative_rate: float | jax.Array,
        n_simulations: int,
        dtype: str,
    ) -> jax.Array:
        cutoff_calls.append(
            {
                "key": key,
                "n_observations": n_observations,
                "correlation": correlation,
                "false_negative_rate": false_negative_rate,
                "n_simulations": n_simulations,
                "dtype": dtype,
            }
        )
        return screening_cutoff

    def prepare_spy(
        mask: jax.Array,
        other_indices: jax.Array,
    ) -> SBMCompactStructure:
        assert jnp.array_equal(mask, active_mask)
        structure = original_prepare(mask, other_indices)
        structure_calls.append(structure)
        return structure

    monkeypatch.setattr(
        estimator_module,
        "pairwise_jeffreys_bayes_factors",
        lambda _: jnp.asarray(
            [[jnp.nan, jnp.nan, jnp.nan], [1.0, jnp.nan, jnp.nan], [0.1, 2.0, jnp.nan]],
            dtype=jnp.float64,
        ),
    )
    monkeypatch.setattr(estimator_module, "estimate_fnr_cutoff", cutoff_spy)
    monkeypatch.setattr(estimator_module, "prepare_sbm_compact_structure", prepare_spy)
    _install_passthrough_runner(monkeypatch, runner_calls)
    master_key = jax.random.key(401)
    model = SBMSPCov(n_samples=2, burnin=1, n_chains=3, dtype="float64")

    model.fit(_small_x(), key=master_key)

    assert len(cutoff_calls) == 1
    assert len(structure_calls) == 1
    assert len(runner_calls) == 1
    screening_key, sampler_key = jax.random.split(master_key)
    np.testing.assert_array_equal(
        jax.random.key_data(cutoff_calls[0]["key"]),
        jax.random.key_data(screening_key),
    )
    assert cutoff_calls[0]["n_observations"] == 4
    assert float(cutoff_calls[0]["correlation"]) == 0.25
    assert float(cutoff_calls[0]["false_negative_rate"]) == 0.05
    assert cutoff_calls[0]["n_simulations"] == 1000
    assert cutoff_calls[0]["dtype"] == "float64"

    runner_call = runner_calls[0]
    np.testing.assert_array_equal(
        jax.random.key_data(runner_call["keys"]),
        jax.random.key_data(jax.random.split(sampler_key, 3)),
    )
    np.testing.assert_array_equal(
        runner_call["scatter"],
        np.asarray(
            [[6.0, 7.0, 3.0], [7.0, 54.0, 36.0], [3.0, 36.0, 30.0]],
            dtype=np.float64,
        ),
    )
    assert int(runner_call["n_observations"]) == 4
    assert float(runner_call["a"]) == 0.5
    assert float(runner_call["b"]) == 0.5
    assert float(runner_call["diagonal_rate"]) == 1.0
    assert float(runner_call["tau1sq"]) == pytest.approx(math.log(3.0) / (3**2 * 4))
    assert runner_call["structure"] is structure_calls[0]
    assert structure_calls[0].other_indices.shape == (3, 2)
    assert structure_calls[0].active_positions.ndim == 2
    states = runner_call["states"]
    assert isinstance(states, BMState)
    assert states.covariance.shape == (3, 3, 3)
    np.testing.assert_array_equal(model.screening_mask_, active_mask)


def test_sbmspcov_fnr_flow_matches_bspcov_1_0_3_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_x64()
    metadata = _load_metadata()
    x = jnp.asarray(_load_csv("sbm_screening_x.csv"), dtype=jnp.float64)
    fixture_scores = jnp.asarray(
        _load_csv("sbm_screening_pairwise_bf.csv"),
        dtype=jnp.float64,
    )
    fixture_cutoff = jnp.asarray(metadata["fnr"]["cutoff"], dtype=jnp.float64)
    expected_mask = _load_csv("sbm_screening_fnr_active_mask.csv").astype(bool)
    initial = jnp.asarray(
        _load_csv("sbm_screening_initial_covariance.csv"),
        dtype=jnp.float64,
    )
    expected_initial = _load_csv("sbm_screening_fnr_covariance.csv")
    score_inputs: list[jax.Array] = []
    cutoff_inputs: list[jax.Array] = []

    def fixture_score_spy(observations: jax.Array) -> jax.Array:
        score_inputs.append(observations)
        return fixture_scores

    def fixture_cutoff_spy(key: jax.Array, **_: object) -> jax.Array:
        cutoff_inputs.append(key)
        return fixture_cutoff

    monkeypatch.setattr(
        estimator_module,
        "pairwise_jeffreys_bayes_factors",
        fixture_score_spy,
    )
    monkeypatch.setattr(estimator_module, "estimate_fnr_cutoff", fixture_cutoff_spy)
    _install_passthrough_runner(monkeypatch)
    master_key = jax.random.key(409)
    model = SBMSPCov(n_samples=1, burnin=0, dtype="float64")

    model.fit(x, key=master_key, initial_covariance=initial)

    assert len(score_inputs) == 1
    assert len(cutoff_inputs) == 1
    np.testing.assert_array_equal(score_inputs[0], x)
    np.testing.assert_array_equal(
        jax.random.key_data(cutoff_inputs[0]),
        jax.random.key_data(jax.random.split(master_key)[0]),
    )
    np.testing.assert_array_equal(model.screening_mask_, expected_mask)
    assert model.screening_cutoff_ is not None
    assert float(model.screening_cutoff_) == metadata["fnr"]["cutoff"]
    np.testing.assert_array_equal(model.initial_covariance_, expected_initial)
    assert model.diagnostics_.screening_jitter == 0.0
    assert model.diagnostics_.n_active_edges == 3
    assert model.diagnostics_.n_screened_edges == 3


def test_sbmspcov_correlation_flow_uses_r_type7_retained_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_x64()
    x = jnp.asarray(_load_csv("sbm_screening_x.csv"), dtype=jnp.float64)
    expected = np.asarray(
        [
            [False, True, False, True],
            [True, False, False, False],
            [False, False, False, False],
            [True, False, False, False],
        ]
    )
    original = cast(
        Callable[[jax.Array, float | jax.Array], jax.Array],
        vars(estimator_module)["correlation_screening_mask"],
    )
    retained: list[float] = []

    def correlation_spy(
        observations: jax.Array,
        retained_fraction: float | jax.Array,
    ) -> jax.Array:
        retained.append(float(retained_fraction))
        return original(observations, retained_fraction)

    def unexpected_fnr(*_: object, **__: object) -> jax.Array:
        raise AssertionError("correlation screening must not execute the FNR path")

    monkeypatch.setattr(estimator_module, "correlation_screening_mask", correlation_spy)
    monkeypatch.setattr(estimator_module, "estimate_fnr_cutoff", unexpected_fnr)
    monkeypatch.setattr(
        estimator_module,
        "pairwise_jeffreys_bayes_factors",
        unexpected_fnr,
    )
    _install_passthrough_runner(monkeypatch)
    model = SBMSPCov(
        n_samples=1,
        burnin=0,
        cutoff_method="correlation",
        retained_fraction=0.3,
        dtype="float64",
    )

    model.fit(x, key=jax.random.key(419))

    assert retained == [0.3]
    np.testing.assert_array_equal(model.screening_mask_, expected)
    assert model.screening_cutoff_ is None
    assert model.diagnostics_.n_active_edges == 2
    assert model.diagnostics_.n_screened_edges == 4


def test_sbmspcov_screens_support_before_applying_upstream_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_x64()
    active_mask = jnp.asarray(
        [[False, True, True], [True, False, False], [True, False, False]]
    )
    initial = jnp.asarray(
        [[1.0, 0.8, 0.8], [0.8, 1.0, 0.3], [0.8, 0.3, 1.0]],
        dtype=jnp.float64,
    )
    screened = jnp.asarray(
        [[1.0, 0.8, 0.8], [0.8, 1.0, 0.0], [0.8, 0.0, 1.0]],
        dtype=jnp.float64,
    )
    expected_jitter = 0.1323708498984763
    runner_calls: list[_RunnerCall] = []
    monkeypatch.setattr(
        estimator_module,
        "correlation_screening_mask",
        lambda *_args, **_kwargs: active_mask,
    )
    _install_passthrough_runner(monkeypatch, runner_calls)
    model = SBMSPCov(
        n_samples=1,
        burnin=0,
        cutoff_method="correlation",
        retained_fraction=0.5,
        dtype="float64",
    )

    model.fit(
        _small_x(),
        key=jax.random.key(421),
        initial_covariance=initial,
    )

    assert model.diagnostics_.screening_jitter == pytest.approx(
        expected_jitter,
        rel=0.0,
        abs=2e-15,
    )
    expected = screened + expected_jitter * jnp.eye(3, dtype=jnp.float64)
    np.testing.assert_allclose(
        model.initial_covariance_, expected, rtol=0.0, atol=2e-15
    )
    states = runner_calls[0]["states"]
    assert isinstance(states, BMState)
    np.testing.assert_allclose(states.covariance[0], expected, rtol=0.0, atol=2e-15)
    assert model.initial_covariance_[1, 2] == 0.0


def test_sbmspcov_jitters_at_inclusive_r_eigenvalue_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_x64()
    active_mask = jnp.zeros((3, 3), dtype=jnp.bool_)
    initial = jnp.diag(jnp.asarray([1e-15, 1.0, 2.0], dtype=jnp.float64))
    monkeypatch.setattr(
        estimator_module,
        "correlation_screening_mask",
        lambda *_args, **_kwargs: active_mask,
    )
    _install_passthrough_runner(monkeypatch)
    model = SBMSPCov(
        n_samples=1,
        burnin=0,
        cutoff_method="correlation",
        retained_fraction=0.0,
        dtype="float64",
    )

    model.fit(
        _small_x(),
        key=jax.random.key(431),
        initial_covariance=initial,
    )

    expected_jitter = 0.001 - 1e-15
    assert model.diagnostics_.screening_jitter == pytest.approx(
        expected_jitter,
        rel=0.0,
        abs=1e-18,
    )
    np.testing.assert_allclose(
        jnp.diag(model.initial_covariance_),
        jnp.asarray([0.001, 1.0 + expected_jitter, 2.0 + expected_jitter]),
        rtol=0.0,
        atol=1e-18,
    )


def test_sbmspcov_publishes_exact_zero_covariance_and_unit_phi_off_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_x64()
    active_mask = jnp.asarray(
        [
            [False, True, False, True],
            [True, False, False, True],
            [False, False, False, False],
            [True, True, False, False],
        ]
    )
    packed_covariance = jnp.asarray(
        [4.0, 0.25, 0.0, 0.5, 4.0, 0.0, 0.75, 4.0, 0.0, 4.0],
        dtype=jnp.float64,
    )
    packed_phi = jnp.asarray(
        [1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0],
        dtype=jnp.float64,
    )

    def runner(
        keys: jax.Array,
        states: BMState,
        *_: object,
        burnin: int,
        n_samples: int,
    ) -> SBMPackedChainResult:
        return _accepted_result(
            keys,
            states,
            burnin=burnin,
            n_samples=n_samples,
            covariance=packed_covariance,
            phi=packed_phi,
        )

    monkeypatch.setattr(
        estimator_module,
        "correlation_screening_mask",
        lambda *_args, **_kwargs: active_mask,
    )
    monkeypatch.setattr(estimator_module, "_compile_sbm_chains", lambda: runner)
    x = jnp.asarray(_load_csv("sbm_screening_x.csv"), dtype=jnp.float64)
    model = SBMSPCov(
        n_samples=2,
        burnin=0,
        cutoff_method="correlation",
        retained_fraction=0.5,
        dtype="float64",
    )

    model.fit(x, key=jax.random.key(433))

    excluded = (~active_mask) & ~jnp.eye(4, dtype=jnp.bool_)
    assert jnp.all(model.posterior_samples_[..., excluded] == 0.0)
    assert jnp.all(model.phi_samples_[..., excluded] == 1.0)
    assert jnp.all(model.posterior_samples_[..., active_mask] != 0.0)
    assert jnp.all(model.phi_samples_[..., active_mask] != 1.0)


def test_sbmspcov_rejected_refit_preserves_every_published_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_x64()
    active_mask = jnp.asarray(
        [[False, True, False], [True, False, False], [False, False, False]]
    )
    reject_next = False

    def runner(
        keys: jax.Array,
        states: BMState,
        *_: object,
        burnin: int,
        n_samples: int,
    ) -> SBMPackedChainResult:
        return _accepted_result(
            keys,
            states,
            burnin=burnin,
            n_samples=n_samples,
            accepted=not reject_next,
        )

    monkeypatch.setattr(
        estimator_module,
        "correlation_screening_mask",
        lambda *_args, **_kwargs: active_mask,
    )
    monkeypatch.setattr(estimator_module, "_compile_sbm_chains", lambda: runner)
    model = SBMSPCov(
        n_samples=1,
        burnin=0,
        cutoff_method="correlation",
        retained_fraction=0.5,
        dtype="float64",
    )
    x = _small_x()
    model.fit(x, key=jax.random.key(439))
    published_names = (
        "posterior_samples_packed_",
        "phi_samples_packed_",
        "covariance_",
        "diagnostics_",
        "screening_mask_",
        "screening_cutoff_",
        "initial_covariance_",
        "n_features_in_",
        "n_observations_",
        "dtype_",
        "device_",
    )
    previous = {name: getattr(model, name) for name in published_names}
    reject_next = True

    with pytest.raises(RuntimeError, match="SBM sampling rejected 1 sweep"):
        model.fit(x * 2.0, key=jax.random.key(443))

    for name, value in previous.items():
        assert getattr(model, name) is value
