import jax
import jax.numpy as jnp
import pytest

import pybspcov.estimators as estimator_module
from pybspcov import SBMDiagnostics, SBMSPCov


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


def _correlation_model(**kwargs: object) -> SBMSPCov:
    configuration: dict[str, object] = {
        "n_samples": 1,
        "burnin": 0,
        "cutoff_method": "correlation",
        "retained_fraction": 0.0,
        "dtype": "float32",
    }
    configuration.update(kwargs)
    return SBMSPCov(**configuration)  # type: ignore[arg-type]


def test_sbm_chain_compiler_is_reused_across_fits() -> None:
    assert (
        estimator_module._compile_sbm_chains() is estimator_module._compile_sbm_chains()
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sbmspcov_fit_exposes_lazy_packed_device_arrays(dtype_name: str) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    model = _correlation_model(
        n_samples=2,
        burnin=1,
        n_chains=2,
        dtype=dtype_name,
    )
    assert not hasattr(model, "covariance_")
    assert not hasattr(model, "posterior_samples_")
    assert not hasattr(model, "diagnostics_")

    fitted = model.fit(
        _centered_case(jnp.dtype(dtype_name)),
        key=jax.random.key(101),
    )

    assert fitted is model
    assert model.posterior_samples_packed_.shape == (2, 2, 6)
    assert model.phi_samples_packed_.shape == (2, 2, 6)
    assert "posterior_samples_" not in vars(model)
    assert "phi_samples_" not in vars(model)
    assert model.posterior_samples_.shape == (2, 2, 3, 3)
    assert model.phi_samples_.shape == (2, 2, 3, 3)
    assert model.covariance_.shape == (3, 3)
    assert model.covariance_.dtype == jnp.dtype(dtype_name)
    tolerance = 2e-5 if dtype_name == "float32" else 1e-11
    assert jnp.all(jnp.isfinite(model.posterior_samples_))
    assert jnp.allclose(
        model.posterior_samples_,
        jnp.swapaxes(model.posterior_samples_, -1, -2),
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.all(jnp.linalg.eigvalsh(model.posterior_samples_) > 0.0)
    assert jnp.all(jnp.isfinite(model.covariance_))
    assert jnp.allclose(
        model.covariance_,
        model.covariance_.T,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.all(jnp.linalg.eigvalsh(model.covariance_) > 0.0)
    assert jnp.allclose(
        model.covariance_,
        jnp.mean(model.posterior_samples_, axis=(0, 1)),
    )
    assert model.posterior_samples_packed_.devices() == {model.device_}
    assert model.phi_samples_packed_.devices() == {model.device_}
    assert model.screening_mask_.devices() == {model.device_}
    assert model.screening_cutoff_ is None
    assert model.diagnostics_.accepted.shape == (2, 3)
    assert model.diagnostics_.n_sweeps == 6
    assert model.diagnostics_.n_rejected_sweeps == 0
    assert model.diagnostics_.n_active_edges == 0
    assert model.diagnostics_.n_screened_edges == 3
    assert model.n_features_in_ == 3
    assert model.n_observations_ == 6
    assert model.dtype_ == jnp.dtype(dtype_name)


def test_sbmspcov_zero_width_support_is_reproducible_across_chains() -> None:
    x = _centered_case(jnp.float32)
    first = _correlation_model(n_samples=2, burnin=1, n_chains=2)
    second = _correlation_model(n_samples=2, burnin=1, n_chains=2)

    first.fit(x, key=jax.random.key(103))
    second.fit(x, key=jax.random.key(103))

    assert first.screening_mask_.shape == (3, 3)
    assert not bool(jnp.any(first.screening_mask_))
    assert jnp.array_equal(first.posterior_samples_, second.posterior_samples_)
    assert not jnp.allclose(
        first.posterior_samples_[0],
        first.posterior_samples_[1],
    )
    off_diagonal = ~jnp.eye(3, dtype=jnp.bool_)
    assert jnp.all(first.posterior_samples_[..., off_diagonal] == 0.0)


def test_sbmspcov_fnr_fit_publishes_cutoff_and_shared_support() -> None:
    model = SBMSPCov(
        n_samples=1,
        burnin=0,
        n_chains=2,
        n_cutoff_simulations=8,
        dtype="float32",
    )

    model.fit(_centered_case(jnp.float32), key=jax.random.key(107))

    assert isinstance(model.screening_cutoff_, jax.Array)
    assert model.screening_cutoff_.shape == ()
    assert model.screening_cutoff_.dtype == jnp.float32
    assert model.screening_cutoff_.devices() == {model.device_}
    assert model.screening_mask_.shape == (3, 3)
    assert jnp.array_equal(model.screening_mask_, model.screening_mask_.T)
    assert not bool(jnp.any(jnp.diag(model.screening_mask_)))


def test_sbmspcov_stores_screened_jittered_initial_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_mask = jnp.asarray(
        [[False, True, True], [True, False, False], [True, False, False]]
    )
    monkeypatch.setattr(
        estimator_module,
        "correlation_screening_mask",
        lambda *_args, **_kwargs: active_mask,
    )
    initial = jnp.full((3, 3), 0.9, dtype=jnp.float32).at[jnp.diag_indices(3)].set(1.0)
    model = SBMSPCov(
        n_samples=1,
        burnin=0,
        cutoff_method="correlation",
        retained_fraction=0.5,
        dtype="float32",
    )

    model.fit(
        _centered_case(jnp.float32),
        key=jax.random.key(109),
        initial_covariance=initial,
    )

    assert model.diagnostics_.screening_jitter > 0.0
    assert jnp.all(jnp.linalg.eigvalsh(model.initial_covariance_) > 0.0)
    assert model.initial_covariance_[1, 2] == 0.0
    assert model.initial_covariance_[0, 1] == initial[0, 1]
    assert jnp.allclose(
        jnp.diag(model.initial_covariance_),
        jnp.diag(initial) + model.diagnostics_.screening_jitter,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_samples": 0}, "n_samples must be positive"),
        ({"burnin": -1}, "burnin must be non-negative"),
        ({"n_chains": 0}, "n_chains must be positive"),
        ({"n_cutoff_simulations": 0}, "n_cutoff_simulations must be positive"),
        ({"cutoff_method": "rank"}, "cutoff_method must be"),
        ({"fnr_correlation": 1.1}, "fnr_correlation must be between"),
        ({"false_negative_rate": -0.1}, "false_negative_rate must be between"),
        ({"retained_fraction": 1.1}, "retained_fraction must be between"),
        ({"dtype": "float16"}, "dtype must be 'float32' or 'float64'"),
        ({"device": "tpu"}, "device must be 'cpu', 'gpu', or 'cuda'"),
    ],
)
def test_sbmspcov_validates_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SBMSPCov(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("fnr_correlation", jnp.asarray([0.25])),
        ("false_negative_rate", jnp.asarray([0.05])),
        ("retained_fraction", jnp.asarray([0.2])),
    ],
)
def test_sbmspcov_requires_scalar_screening_probabilities(
    name: str,
    value: jax.Array,
) -> None:
    with pytest.raises(ValueError, match=f"{name} must be a scalar"):
        SBMSPCov(**{name: value})  # type: ignore[arg-type]


def test_sbmspcov_revalidates_mutated_configuration_before_fit() -> None:
    model = _correlation_model()
    model.retained_fraction = -0.1

    with pytest.raises(ValueError, match="retained_fraction must be between"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(113))

    assert not hasattr(model, "covariance_")


def test_sbmspcov_requires_typed_scalar_key() -> None:
    model = _correlation_model()
    x = _centered_case(jnp.float32)

    with pytest.raises(TypeError, match="typed JAX key"):
        model.fit(x, key=jax.random.PRNGKey(127))
    with pytest.raises(ValueError, match="scalar key"):
        model.fit(x, key=jax.random.split(jax.random.key(127), 2))


def test_sbmspcov_float64_requires_x64() -> None:
    if jax.config.x64_enabled:
        pytest.skip("this contract requires an x64-disabled process")
    model = SBMSPCov(n_samples=1, burnin=0)

    with pytest.raises(RuntimeError, match="JAX_ENABLE_X64=1"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(129))


def test_sbmspcov_does_not_publish_rejected_chains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = estimator_module._compile_sbm_chains

    def rejecting_compiler() -> object:
        run = original()

        def reject_last(*args: object, **kwargs: object) -> object:
            result = run(*args, **kwargs)
            return result._replace(accepted=result.accepted.at[0, -1].set(False))

        return reject_last

    monkeypatch.setattr(estimator_module, "_compile_sbm_chains", rejecting_compiler)
    model = _correlation_model()

    with pytest.raises(RuntimeError, match="SBM sampling rejected 1 sweep"):
        model.fit(_centered_case(jnp.float32), key=jax.random.key(131))

    assert not hasattr(model, "covariance_")


def test_sbmspcov_failed_refit_preserves_previous_fitted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _correlation_model()
    x = _centered_case(jnp.float32)
    model.fit(x, key=jax.random.key(137))
    previous_samples = model.posterior_samples_packed_
    original = estimator_module._compile_sbm_chains

    def rejecting_compiler() -> object:
        run = original()

        def reject_last(*args: object, **kwargs: object) -> object:
            result = run(*args, **kwargs)
            return result._replace(accepted=result.accepted.at[0, -1].set(False))

        return reject_last

    monkeypatch.setattr(estimator_module, "_compile_sbm_chains", rejecting_compiler)

    with pytest.raises(RuntimeError, match="SBM sampling rejected 1 sweep"):
        model.fit(x, key=jax.random.key(139))

    assert model.posterior_samples_packed_ is previous_samples
    assert isinstance(model.diagnostics_, SBMDiagnostics)
