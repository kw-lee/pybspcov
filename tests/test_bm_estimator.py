import jax
import jax.numpy as jnp
import pytest

import pybspcov.estimators as estimator_module
from pybspcov import BMSPCov
from pybspcov.kernels import BMPackedChainResult, BMState


def _centered_case(dtype: jnp.dtype) -> jax.Array:
    return jnp.asarray(
        [
            [-1.0, 0.5, 0.2],
            [-0.4, -0.7, 0.1],
            [0.2, 0.1, -0.8],
            [0.5, -0.2, 0.6],
            [0.9, 0.4, -0.3],
            [-0.2, -0.1, 0.2],
        ],
        dtype=dtype,
    )


def test_bm_chain_compiler_is_reused_across_fits() -> None:
    assert (
        estimator_module._compile_bm_chains() is estimator_module._compile_bm_chains()
    )


def test_bmspcov_fit_returns_self_and_exposes_device_arrays() -> None:
    model = BMSPCov(n_samples=2, burnin=1, n_chains=2, dtype="float32")
    assert not hasattr(model, "covariance_")
    assert not hasattr(model, "posterior_samples_")
    assert not hasattr(model, "diagnostics_")

    fitted = model.fit(_centered_case(jnp.float32), key=jax.random.key(17))

    assert fitted is model
    assert model.posterior_samples_packed_.shape == (2, 2, 6)
    assert model.phi_samples_packed_.shape == (2, 2, 6)
    assert "posterior_samples_" not in vars(model)
    assert "phi_samples_" not in vars(model)
    assert model.posterior_samples_.shape == (2, 2, 3, 3)
    assert model.covariance_.shape == (3, 3)
    assert model.covariance_.dtype == jnp.float32
    assert model.posterior_samples_.dtype == jnp.float32
    assert model.phi_samples_.shape == (2, 2, 3, 3)
    assert jnp.all(jnp.isfinite(model.posterior_samples_))
    assert jnp.allclose(
        model.posterior_samples_,
        jnp.swapaxes(model.posterior_samples_, -1, -2),
    )
    assert jnp.all(jnp.linalg.eigvalsh(model.posterior_samples_) > 0.0)

    assert jnp.allclose(
        model.covariance_,
        jnp.mean(model.posterior_samples_, axis=(0, 1)),
    )
    assert model.diagnostics_.accepted.shape == (2, 3)
    assert model.diagnostics_.n_sweeps == 6
    assert model.diagnostics_.n_rejected_sweeps == 0
    assert model.diagnostics_.n_initial_repairs == 0
    assert model.diagnostics_.initial_variance_floor > 0.0
    assert model.n_features_in_ == 3
    assert model.n_observations_ == 6
    assert model.dtype_ == jnp.dtype("float32")
    assert model.posterior_samples_packed_.devices() == {model.device_}
    assert model.phi_samples_packed_.devices() == {model.device_}


def test_bmspcov_is_reproducible_and_splits_independent_chain_keys() -> None:
    x = _centered_case(jnp.float32)
    first = BMSPCov(n_samples=2, burnin=1, n_chains=2, dtype="float32")
    second = BMSPCov(n_samples=2, burnin=1, n_chains=2, dtype="float32")

    first.fit(x, key=jax.random.key(23))
    second.fit(x, key=jax.random.key(23))

    assert jnp.array_equal(first.posterior_samples_, second.posterior_samples_)
    assert not jnp.allclose(
        first.posterior_samples_[0],
        first.posterior_samples_[1],
    )


def test_bmspcov_preserves_explicit_initial_covariance() -> None:
    x = _centered_case(jnp.float32)
    initial = jnp.diag(jnp.asarray([1.5, 1.25, 0.75], dtype=jnp.float32))
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")

    model.fit(x, key=jax.random.key(29), initial_covariance=initial)

    assert jnp.array_equal(model.initial_covariance_, initial)


def test_bmspcov_default_initialization_stabilizes_zero_scale_columns() -> None:
    x = jnp.asarray(
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        dtype=jnp.float32,
    )
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")

    model.fit(x, key=jax.random.key(31))

    assert jnp.all(jnp.linalg.eigvalsh(model.initial_covariance_) > 0.0)
    assert model.diagnostics_.n_initial_repairs == 1
    assert model.diagnostics_.initial_variance_floor > 0.0


def test_bmspcov_default_initialization_matches_r_sample_variances() -> None:
    x = _centered_case(jnp.float32)
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")

    model.fit(x, key=jax.random.key(33))

    expected = jnp.diag(jnp.var(x, axis=0, ddof=1))
    assert jnp.allclose(model.initial_covariance_, expected)


def test_bmspcov_wires_upstream_r_defaults_without_centering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, jax.Array] = {}

    def capture_runner(
        keys: jax.Array,
        states: BMState,
        scatter: jax.Array,
        other_indices: jax.Array,
        n_observations: jax.Array,
        a: jax.Array,
        b: jax.Array,
        diagonal_rate: jax.Array,
        tau1sq: jax.Array,
        *,
        burnin: int,
        n_samples: int,
    ) -> BMPackedChainResult:
        captured.update(
            scatter=scatter,
            other_indices=other_indices,
            n_observations=n_observations,
            a=a,
            b=b,
            diagonal_rate=diagonal_rate,
            tau1sq=tau1sq,
            initial_covariance=states.covariance,
        )
        covariance = jnp.broadcast_to(
            jnp.asarray([4.0, 0.0, 4.0], dtype=scatter.dtype),
            (keys.shape[0], n_samples, 3),
        )
        phi = jnp.ones_like(covariance)
        accepted = jnp.ones(
            (keys.shape[0], burnin + n_samples),
            dtype=jnp.bool_,
        )
        return BMPackedChainResult(states, covariance, phi, accepted)

    monkeypatch.setattr(
        estimator_module,
        "_compile_bm_chains",
        lambda: capture_runner,
    )
    x = jnp.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=jnp.float32,
    )
    model = BMSPCov(n_samples=1, burnin=1, dtype="float32")

    model.fit(x, key=jax.random.key(34))

    assert jnp.array_equal(
        captured["scatter"],
        jnp.asarray([[35.0, 44.0], [44.0, 56.0]], dtype=jnp.float32),
    )
    assert jnp.array_equal(
        captured["other_indices"],
        jnp.asarray([[1], [0]], dtype=jnp.int32),
    )
    assert int(captured["n_observations"]) == 3
    assert float(captured["a"]) == 0.5
    assert float(captured["b"]) == 0.5
    assert float(captured["diagonal_rate"]) == 1.0
    assert float(captured["tau1sq"]) == pytest.approx(10_000.0 / (3 * 2**4))
    expected_initial = jnp.asarray([jnp.eye(2, dtype=jnp.float32) * 4.0])
    assert jnp.array_equal(captured["initial_covariance"], expected_initial)
    assert jnp.array_equal(
        model.posterior_samples_packed_[0, 0],
        jnp.asarray([4.0, 0.0, 4.0], dtype=jnp.float32),
    )


def test_bmspcov_preserves_float64_when_x64_is_enabled() -> None:
    if not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    model = BMSPCov(n_samples=1, burnin=0, dtype="float64")

    model.fit(_centered_case(jnp.float64), key=jax.random.key(35))

    assert model.posterior_samples_.dtype == jnp.float64
    assert model.covariance_.dtype == jnp.float64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_samples": 0}, "n_samples must be positive"),
        ({"burnin": -1}, "burnin must be non-negative"),
        ({"n_chains": 0}, "n_chains must be positive"),
        ({"n_samples": True}, "n_samples must be an integer"),
        ({"dtype": "float16"}, "dtype must be 'float32' or 'float64'"),
        ({"device": "tpu"}, "device must be 'cpu', 'gpu', or 'cuda'"),
    ],
)
def test_bmspcov_validates_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        BMSPCov(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("x", "message"),
    [
        (jnp.ones((4,)), "two-dimensional"),
        (jnp.ones((4, 1)), "at least two columns"),
        (jnp.ones((0, 2)), "at least two rows"),
        (jnp.ones((1, 2)), "at least two rows"),
        (jnp.asarray([[1.0, jnp.nan], [2.0, 3.0]]), "finite"),
    ],
)
def test_bmspcov_validates_x(x: jax.Array, message: str) -> None:
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")

    with pytest.raises(ValueError, match=message):
        model.fit(x, key=jax.random.key(37))

    assert not hasattr(model, "covariance_")


@pytest.mark.parametrize(
    ("initial", "message"),
    [
        (jnp.eye(2), "shape"),
        (jnp.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "symmetric"),
        (jnp.diag(jnp.asarray([1.0, 0.0, 1.0])), "positive definite"),
        (
            jnp.asarray([[1.0, 0.0, 0.0], [0.0, jnp.inf, 0.0], [0.0, 0.0, 1.0]]),
            "finite",
        ),
    ],
)
def test_bmspcov_validates_initial_covariance(
    initial: jax.Array,
    message: str,
) -> None:
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")

    with pytest.raises(ValueError, match=message):
        model.fit(
            _centered_case(jnp.float32),
            key=jax.random.key(41),
            initial_covariance=initial,
        )


def test_bmspcov_requires_typed_scalar_key() -> None:
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")
    x = _centered_case(jnp.float32)

    with pytest.raises(TypeError, match="typed JAX key"):
        model.fit(x, key=jax.random.PRNGKey(43))
    with pytest.raises(ValueError, match="scalar key"):
        model.fit(x, key=jax.random.split(jax.random.key(43), 2))


def test_bmspcov_float64_requires_x64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(estimator_module, "_x64_enabled", lambda: False)
    model = BMSPCov(n_samples=1, burnin=0)

    with pytest.raises(RuntimeError, match="JAX_ENABLE_X64=1"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(47))


def test_bmspcov_revalidates_mutated_configuration_before_fit() -> None:
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")
    model.device = "tpu"

    with pytest.raises(ValueError, match="device must be"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(49))

    assert not hasattr(model, "covariance_")


def test_bmspcov_explicit_gpu_request_fails_without_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(platform: str) -> list[jax.Device]:
        assert platform == "cuda"
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(estimator_module, "_available_devices", unavailable)
    model = BMSPCov(
        n_samples=1,
        burnin=0,
        dtype="float32",
        device="gpu",
    )

    with pytest.raises(RuntimeError, match="CUDA device was explicitly requested"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(53))


def test_bmspcov_does_not_silently_publish_rejected_chains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = estimator_module._compile_bm_chains

    def rejecting_compiler() -> object:
        run = original()

        def reject_last(*args: object, **kwargs: object) -> object:
            result = run(*args, **kwargs)
            rejected = result.accepted.at[0, -1].set(False)
            return result._replace(accepted=rejected)

        return reject_last

    monkeypatch.setattr(estimator_module, "_compile_bm_chains", rejecting_compiler)
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")

    with pytest.raises(RuntimeError, match="rejected 1 sweep"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(59))

    assert not hasattr(model, "covariance_")


def test_bmspcov_failed_refit_preserves_previous_fitted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BMSPCov(n_samples=1, burnin=0, dtype="float32")
    x = _centered_case(jnp.float32)
    model.fit(x, key=jax.random.key(61))
    previous_samples = model.posterior_samples_packed_
    original = estimator_module._compile_bm_chains

    def rejecting_compiler() -> object:
        run = original()

        def reject_last(*args: object, **kwargs: object) -> object:
            result = run(*args, **kwargs)
            return result._replace(accepted=result.accepted.at[0, -1].set(False))

        return reject_last

    monkeypatch.setattr(estimator_module, "_compile_bm_chains", rejecting_compiler)

    with pytest.raises(RuntimeError, match="rejected 1 sweep"):
        model.fit(x, key=jax.random.key(63))

    assert model.posterior_samples_packed_ is previous_samples
