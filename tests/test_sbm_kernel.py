import jax
import jax.numpy as jnp
import pytest

from pybspcov import kernels
from pybspcov.kernels import bm as bm_kernel
from pybspcov.kernels import sbm as sbm_kernel
from pybspcov.kernels.bm import bm_column_parameters, bm_sweep, initialize_bm_state
from pybspcov.kernels.sbm import (
    initialize_sbm_state,
    sample_sbm_chain,
    sbm_column_parameters,
    sbm_sweep,
    validate_sbm_active_mask,
)
from pybspcov.sampling.gig import GIGSample


def _column_case(dtype: jnp.dtype) -> tuple[jax.Array, ...]:
    covariance = jnp.asarray(
        [
            [2.0, 0.2, 0.0, 0.1],
            [0.2, 1.5, -0.15, 0.0],
            [0.0, -0.15, 1.2, 0.25],
            [0.1, 0.0, 0.25, 1.8],
        ],
        dtype=dtype,
    )
    x = jnp.asarray(
        [
            [-1.0, 0.5, 0.2, -0.1],
            [-0.4, -0.7, 0.1, 0.3],
            [0.2, 0.1, -0.8, 0.4],
            [0.5, -0.2, 0.6, -0.5],
            [0.9, 0.4, -0.3, 0.2],
            [-0.2, -0.1, 0.2, -0.3],
        ],
        dtype=dtype,
    )
    tau = jnp.asarray(
        [
            [0.4, 0.3, 0.2, 0.5],
            [0.3, 0.4, 0.6, 0.7],
            [0.2, 0.6, 0.4, 0.8],
            [0.5, 0.7, 0.8, 0.4],
        ],
        dtype=dtype,
    )
    active_mask = jnp.asarray(
        [
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, False],
            [True, False, False, False],
        ]
    )
    return covariance, jnp.linalg.inv(covariance), x.T @ x, tau, active_mask


def _sbm_rollback_case() -> tuple[sbm_kernel.BMState, tuple[jax.Array, ...]]:
    dtype = jnp.float32
    covariance, _, scatter, _, active_mask = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = initialize_sbm_state(covariance, tau1sq, active_mask)
    arguments = (
        scatter,
        jnp.asarray(
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
            dtype=jnp.int32,
        ),
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        active_mask,
    )
    return state, arguments


def _assert_state_array_equal(
    actual: sbm_kernel.BMState,
    expected: sbm_kernel.BMState,
) -> None:
    for field_name, actual_value, expected_value in zip(
        sbm_kernel.BMState._fields,
        actual,
        expected,
        strict=True,
    ):
        assert jnp.array_equal(actual_value, expected_value), field_name


def _accepted_batch_gig(
    keys: jax.Array,
    lambda_: jax.Array,
    chi: jax.Array,
    psi: jax.Array,
) -> GIGSample:
    del keys, lambda_, psi
    return GIGSample(
        value=jnp.ones_like(chi),
        accepted=jnp.ones_like(chi, dtype=jnp.bool_),
        iterations=jnp.ones_like(chi, dtype=jnp.int32),
    )


def test_sbm_kernel_api_is_exported() -> None:
    assert kernels.SBMColumnParameters is sbm_kernel.SBMColumnParameters
    assert kernels.SBMChainResult is sbm_kernel.SBMChainResult
    assert kernels.initialize_sbm_state is sbm_kernel.initialize_sbm_state
    assert kernels.sbm_column_parameters is sbm_kernel.sbm_column_parameters
    assert kernels.sbm_sweep is sbm_kernel.sbm_sweep
    assert kernels.sample_sbm_chain is sbm_kernel.sample_sbm_chain
    assert kernels.validate_sbm_active_mask is sbm_kernel.validate_sbm_active_mask


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sbm_column_parameters_match_explicit_active_submatrix(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    tolerance = 2e-5 if dtype_name == "float32" else 1e-11
    covariance, precision, scatter, tau, active_mask = _column_case(dtype)
    column = jnp.asarray(1, dtype=jnp.int32)
    other_indices = jnp.asarray([0, 2, 3], dtype=jnp.int32)
    gamma = jnp.asarray(0.9, dtype=dtype)
    diagonal_rate = jnp.asarray(1.0, dtype=dtype)

    parameters = jax.jit(sbm_column_parameters)(
        covariance=covariance,
        precision=precision,
        scatter=scatter,
        tau=tau,
        active_mask=active_mask,
        column=column,
        other_indices=other_indices,
        n_observations=jnp.asarray(6, dtype=dtype),
        diagonal_rate=diagonal_rate,
        gamma=gamma,
    )

    precision_block = precision[jnp.ix_(other_indices, other_indices)]
    precision_cross = precision[other_indices, column]
    conditional_precision = (
        precision_block
        - jnp.outer(precision_cross, precision_cross) / precision[column, column]
    )
    scatter_block = scatter[jnp.ix_(other_indices, other_indices)]
    scatter_cross = scatter[other_indices, column]
    active_positions = jnp.asarray([0, 1], dtype=jnp.int32)
    reduced_rows = conditional_precision[active_positions, :]
    reduced_scatter = reduced_rows @ scatter_cross
    quadratic = reduced_rows @ scatter_block @ reduced_rows.T
    beta = covariance[other_indices[active_positions], column]
    expected_chi = (
        beta @ quadratic @ beta - 2.0 * beta @ reduced_scatter + scatter[column, column]
    )
    expected_precision = (
        quadratic / gamma
        + jnp.diag(1.0 / tau[other_indices[active_positions], column])
        + diagonal_rate
        * conditional_precision[jnp.ix_(active_positions, active_positions)]
    )
    expected_mean = jnp.linalg.solve(expected_precision, reduced_scatter) / gamma

    assert parameters.active_count == 2
    assert jnp.array_equal(parameters.active, jnp.asarray([True, True, False]))
    assert jnp.allclose(
        parameters.conditional_precision,
        conditional_precision,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        parameters.gamma_chi, expected_chi, rtol=tolerance, atol=tolerance
    )
    assert jnp.allclose(
        parameters.beta_precision[jnp.ix_(active_positions, active_positions)],
        expected_precision,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        parameters.beta_mean[active_positions],
        expected_mean,
        rtol=tolerance,
        atol=tolerance,
    )
    assert parameters.beta_precision[2, 2] == 1.0
    assert jnp.all(parameters.beta_precision[2, :2] == 0.0)
    assert jnp.all(parameters.beta_precision[:2, 2] == 0.0)
    assert parameters.beta_mean[2] == 0.0
    assert parameters.beta_precision.dtype == dtype
    assert parameters.beta_mean.dtype == dtype


def test_sbm_column_parameters_reduce_to_bm_when_every_edge_is_active() -> None:
    dtype = jnp.float64
    if not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    covariance, precision, scatter, tau, _ = _column_case(dtype)
    column = jnp.asarray(2, dtype=jnp.int32)
    other_indices = jnp.asarray([0, 1, 3], dtype=jnp.int32)
    common = {
        "covariance": covariance,
        "precision": precision,
        "scatter": scatter,
        "tau": tau,
        "column": column,
        "other_indices": other_indices,
        "n_observations": jnp.asarray(6, dtype=dtype),
        "diagonal_rate": jnp.asarray(1.0, dtype=dtype),
        "gamma": jnp.asarray(0.9, dtype=dtype),
    }

    bm_parameters = bm_column_parameters(**common)
    sbm_parameters = sbm_column_parameters(
        **common,
        active_mask=~jnp.eye(4, dtype=jnp.bool_),
    )

    assert sbm_parameters.active_count == 3
    for name in (
        "conditional_precision",
        "conditional_scatter",
        "quadratic",
        "gamma_lambda",
        "gamma_chi",
        "gamma_psi",
        "beta_precision",
        "beta_mean",
    ):
        assert jnp.allclose(getattr(sbm_parameters, name), getattr(bm_parameters, name))


def test_sbm_column_parameters_handle_a_fully_screened_column() -> None:
    dtype = jnp.float32
    covariance, precision, scatter, tau, _ = _column_case(dtype)
    parameters = jax.jit(sbm_column_parameters)(
        covariance=covariance,
        precision=precision,
        scatter=scatter,
        tau=tau,
        active_mask=jnp.zeros((4, 4), dtype=jnp.bool_),
        column=jnp.asarray(1, dtype=jnp.int32),
        other_indices=jnp.asarray([0, 2, 3], dtype=jnp.int32),
        n_observations=jnp.asarray(6, dtype=dtype),
        diagonal_rate=jnp.asarray(1.0, dtype=dtype),
        gamma=jnp.asarray(0.9, dtype=dtype),
    )

    assert parameters.active_count == 0
    assert jnp.all(parameters.beta_mean == 0.0)
    assert jnp.array_equal(parameters.beta_precision, jnp.eye(3, dtype=dtype))
    assert jnp.allclose(parameters.gamma_chi, scatter[1, 1])


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sbm_sweep_reduces_to_bm_when_every_edge_is_active(dtype_name: str) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    covariance, _, scatter, _, _ = _column_case(dtype)
    tau1sq = jnp.asarray(0.2, dtype=dtype)
    full_mask = ~jnp.eye(4, dtype=jnp.bool_)
    other_indices = jnp.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=jnp.int32
    )
    bm_state = initialize_bm_state(covariance, tau1sq)
    sbm_state = initialize_sbm_state(covariance, tau1sq, full_mask)
    arguments = (
        scatter,
        other_indices,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
    )

    bm_result = jax.jit(bm_sweep)(jax.random.key(71), bm_state, *arguments)
    sbm_result = jax.jit(sbm_sweep)(
        jax.random.key(71), sbm_state, *arguments, full_mask
    )

    assert sbm_result.accepted == bm_result.accepted
    for actual, expected in zip(sbm_result.state, bm_result.state, strict=True):
        assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sbm_sweep_preserves_screened_zeros_and_dense_inverse(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    tolerance = 3e-5 if dtype_name == "float32" else 1e-10
    covariance, _, scatter, _, active_mask = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    other_indices = jnp.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=jnp.int32
    )
    state = initialize_sbm_state(covariance, tau1sq, active_mask)
    arguments = (
        scatter,
        other_indices,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        active_mask,
    )
    run_sweep = jax.jit(sbm_sweep)

    result = run_sweep(jax.random.key(83), state, *arguments)
    repeated = run_sweep(jax.random.key(83), state, *arguments)

    excluded = (~active_mask) & (~jnp.eye(4, dtype=jnp.bool_))
    assert result.accepted
    assert jnp.all(result.state.covariance[excluded] == 0.0)
    assert jnp.all(result.state.phi[excluded] == 1.0)
    assert jnp.all(result.state.psi[excluded] == 1.0)
    assert jnp.all(result.state.tau[excluded] == tau1sq)
    assert jnp.allclose(result.state.covariance, result.state.covariance.T)
    assert jnp.all(jnp.linalg.eigvalsh(result.state.covariance) > 0.0)
    assert jnp.allclose(
        result.state.precision,
        jnp.linalg.inv(result.state.covariance),
        rtol=tolerance,
        atol=tolerance,
    )
    for actual, expected in zip(result.state, repeated.state, strict=True):
        assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sbm_sweep_fully_screened_skips_phi_gig(
    dtype_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    covariance, _, scatter, _, _ = _column_case(dtype)
    active_mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    other_indices = jnp.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=jnp.int32
    )
    state = initialize_sbm_state(covariance, tau1sq, active_mask)

    def fail_if_phi_gig_runs(*args: object, **kwargs: object) -> None:
        raise AssertionError("fully screened columns must skip batched phi GIG")

    monkeypatch.setattr(sbm_kernel, "_sample_gig_batch", fail_if_phi_gig_runs)
    with jax.disable_jit():
        result = sbm_sweep(
            jax.random.key(97),
            state,
            scatter,
            other_indices,
            jnp.asarray(6),
            jnp.asarray(0.5, dtype=dtype),
            jnp.asarray(0.5, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
            tau1sq,
            active_mask,
        )

    off_diagonal = ~jnp.eye(4, dtype=jnp.bool_)
    assert result.accepted
    assert not jnp.array_equal(
        jnp.diag(result.state.covariance),
        jnp.diag(state.covariance),
    )
    assert jnp.all(result.state.covariance[off_diagonal] == 0.0)
    assert jnp.all(result.state.precision[off_diagonal] == 0.0)
    assert jnp.array_equal(result.state.phi, state.phi)
    assert jnp.array_equal(result.state.psi, state.psi)
    assert jnp.array_equal(result.state.tau, state.tau)
    assert jnp.allclose(
        result.state.precision,
        jnp.linalg.inv(result.state.covariance),
        rtol=2e-6 if dtype_name == "float32" else 1e-12,
        atol=2e-6 if dtype_name == "float32" else 1e-12,
    )


def test_sbm_sweep_handles_a_fully_screened_graph() -> None:
    dtype = jnp.float32
    covariance, _, scatter, _, _ = _column_case(dtype)
    active_mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    other_indices = jnp.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=jnp.int32
    )
    state = initialize_sbm_state(covariance, tau1sq, active_mask)

    result = jax.jit(sbm_sweep)(
        jax.random.key(97),
        state,
        scatter,
        other_indices,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        active_mask,
    )

    off_diagonal = ~jnp.eye(4, dtype=jnp.bool_)
    assert result.accepted
    assert jnp.all(result.state.covariance[off_diagonal] == 0.0)
    assert jnp.all(result.state.precision[off_diagonal] == 0.0)
    assert jnp.all(result.state.phi == 1.0)
    assert jnp.all(result.state.psi == 1.0)
    assert jnp.all(result.state.tau == tau1sq)
    assert jnp.allclose(
        result.state.precision,
        jnp.linalg.inv(result.state.covariance),
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize(
    ("proposal", "expected"),
    [(5e-7, 1e-6), (1e-6, 1e-6), (2e-6, 2e-6)],
)
def test_sbm_sweeps_floor_only_accepted_gamma_like_r(
    monkeypatch: pytest.MonkeyPatch,
    dtype_name: str,
    proposal: float,
    expected: float,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    proposed_gamma = jnp.asarray(proposal, dtype=dtype)

    def accepted_gamma(*_: object) -> GIGSample:
        return GIGSample(
            value=proposed_gamma,
            accepted=jnp.asarray(True),
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    monkeypatch.setattr(sbm_kernel, "sample_gig", accepted_gamma)
    covariance = jnp.eye(2, dtype=dtype)
    active_mask = jnp.zeros((2, 2), dtype=jnp.bool_)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = initialize_sbm_state(covariance, tau1sq, active_mask)
    other_indices = jnp.asarray([[1], [0]], dtype=jnp.int32)
    shared = (
        covariance,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
    )

    masked = sbm_sweep(
        jax.random.key(103),
        state,
        shared[0],
        other_indices,
        *shared[1:],
        active_mask,
    )
    compact = sbm_kernel.compact_sbm_sweep(
        jax.random.key(103),
        state,
        *shared,
        sbm_kernel.prepare_sbm_compact_structure(active_mask, other_indices),
    )

    expected_diagonal = jnp.full((2,), expected, dtype=dtype)
    assert masked.accepted
    assert compact.accepted
    assert masked.state.covariance.dtype == dtype
    assert compact.state.covariance.dtype == dtype
    assert masked.state.covariance.device == state.covariance.device
    assert compact.state.covariance.device == state.covariance.device
    assert jnp.array_equal(jnp.diag(masked.state.covariance), expected_diagonal)
    for compact_value, masked_value in zip(compact.state, masked.state, strict=True):
        assert jnp.array_equal(compact_value, masked_value)


def test_sbm_sweep_rolls_back_all_state_when_gamma_draw_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arguments = _sbm_rollback_case()

    def reject_gamma(
        key: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del key, lambda_, psi
        return GIGSample(
            value=jnp.full_like(chi, 5e-7),
            accepted=jnp.asarray(False),
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    monkeypatch.setattr(sbm_kernel, "sample_gig", reject_gamma)
    monkeypatch.setattr(sbm_kernel, "_sample_gig_batch", _accepted_batch_gig)
    run_sweep = jax.jit(
        lambda key, initial_state, *sweep_arguments: sbm_kernel.sbm_sweep(
            key,
            initial_state,
            *sweep_arguments,
        )
    )

    result = run_sweep(jax.random.key(107), state, *arguments)

    assert not result.accepted
    _assert_state_array_equal(result.state, state)


def test_sample_sbm_chain_carries_input_state_through_rejected_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, arguments = _sbm_rollback_case()

    def reject_gamma(
        key: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del key, lambda_, psi
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=jnp.asarray(False),
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    monkeypatch.setattr(sbm_kernel, "sample_gig", reject_gamma)
    monkeypatch.setattr(sbm_kernel, "_sample_gig_batch", _accepted_batch_gig)
    run_chain = jax.jit(
        lambda key, initial_state: sbm_kernel.sample_sbm_chain(
            key,
            initial_state,
            *arguments,
            burnin=1,
            n_samples=2,
        )
    )

    result = run_chain(jax.random.key(109), state)

    assert jnp.array_equal(result.accepted, jnp.asarray([False, False, False]))
    _assert_state_array_equal(result.final_state, state)
    assert jnp.array_equal(
        result.covariance,
        jnp.broadcast_to(state.covariance, result.covariance.shape),
    )
    assert jnp.array_equal(
        result.phi,
        jnp.broadcast_to(state.phi, result.phi.shape),
    )


def test_all_active_sbm_and_bm_sweeps_match_when_phi_draw_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtype = jnp.float32
    covariance, _, scatter, _, _ = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    active_mask = ~jnp.eye(4, dtype=jnp.bool_)
    bm_state = initialize_bm_state(covariance, tau1sq)
    sbm_state = initialize_sbm_state(covariance, tau1sq, active_mask)
    arguments = (
        scatter,
        jnp.asarray(
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
            dtype=jnp.int32,
        ),
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
    )

    def accept_gamma(
        key: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del key, lambda_, psi
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=jnp.asarray(True),
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    def reject_phi(
        keys: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del keys, lambda_, psi
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=jnp.zeros_like(chi, dtype=jnp.bool_),
            iterations=jnp.ones_like(chi, dtype=jnp.int32),
        )

    monkeypatch.setattr(bm_kernel, "sample_gig", accept_gamma)
    monkeypatch.setattr(sbm_kernel, "sample_gig", accept_gamma)
    monkeypatch.setattr(bm_kernel, "_sample_gig_batch", reject_phi)
    monkeypatch.setattr(sbm_kernel, "_sample_gig_batch", reject_phi)
    run_bm = jax.jit(lambda key, state: bm_kernel.bm_sweep(key, state, *arguments))
    run_sbm = jax.jit(
        lambda key, state: sbm_kernel.sbm_sweep(
            key,
            state,
            *arguments,
            active_mask,
        )
    )

    sweep_key = jax.random.key(113)
    bm_result = run_bm(sweep_key, bm_state)
    sbm_result = run_sbm(sweep_key, sbm_state)

    assert not bm_result.accepted
    assert not sbm_result.accepted
    _assert_state_array_equal(bm_result.state, bm_state)
    _assert_state_array_equal(sbm_result.state, sbm_state)
    _assert_state_array_equal(sbm_result.state, bm_result.state)


def test_sample_sbm_chain_matches_sequential_sweeps() -> None:
    dtype = jnp.float32
    covariance, _, scatter, _, active_mask = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    other_indices = jnp.asarray(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=jnp.int32
    )
    initial_state = initialize_sbm_state(covariance, tau1sq, active_mask)
    arguments = (
        scatter,
        other_indices,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        active_mask,
    )
    key = jax.random.key(101)
    sweep_keys = jax.random.split(key, 3)

    expected_state = initial_state
    expected_covariance = []
    expected_phi = []
    expected_accepted = []
    for sweep_key in sweep_keys:
        sweep = sbm_sweep(sweep_key, expected_state, *arguments)
        expected_state = sweep.state
        expected_covariance.append(expected_state.covariance)
        expected_phi.append(expected_state.phi)
        expected_accepted.append(sweep.accepted)

    run_chain = jax.jit(
        sample_sbm_chain,
        static_argnames=("burnin", "n_samples"),
    )
    result = run_chain(
        key,
        initial_state,
        *arguments,
        burnin=1,
        n_samples=2,
    )

    assert result.covariance.shape == (2, 4, 4)
    assert result.phi.shape == (2, 4, 4)
    assert result.accepted.shape == (3,)
    for actual, expected in zip(result.final_state, expected_state, strict=True):
        assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(result.covariance, jnp.stack(expected_covariance[1:]))
    assert jnp.array_equal(result.phi, jnp.stack(expected_phi[1:]))

    assert jnp.array_equal(result.accepted, jnp.stack(expected_accepted))


def test_validate_sbm_active_mask_accepts_the_screening_contract() -> None:
    _, _, _, _, active_mask = _column_case(jnp.float32)

    validated = validate_sbm_active_mask(active_mask, dimension=4)

    assert jnp.array_equal(validated, active_mask)
    assert validated.dtype == jnp.bool_


@pytest.mark.parametrize(
    ("active_mask", "error_type", "message"),
    [
        (jnp.zeros((4, 3), dtype=jnp.bool_), ValueError, "shape"),
        (jnp.zeros((4, 4), dtype=jnp.float32), TypeError, "boolean"),
        (
            jnp.asarray(
                [
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            ),
            ValueError,
            "symmetric",
        ),
        (jnp.eye(4, dtype=jnp.bool_), ValueError, "diagonal"),
    ],
)
def test_validate_sbm_active_mask_rejects_invalid_masks(
    active_mask: jax.Array,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        validate_sbm_active_mask(active_mask, dimension=4)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_initialize_sbm_state_repairs_screening_induced_indefiniteness(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    covariance = jnp.asarray(
        [[1.0, 0.8, 0.5], [0.8, 1.0, 0.8], [0.5, 0.8, 1.0]],
        dtype=dtype,
    )
    active_mask = jnp.asarray(
        [[False, True, False], [True, False, True], [False, True, False]]
    )
    screened = jnp.where(active_mask | jnp.eye(3, dtype=jnp.bool_), covariance, 0.0)
    minimum_eigenvalue = jnp.linalg.eigvalsh(screened)[0]
    expected_covariance = screened + (
        -minimum_eigenvalue + jnp.asarray(0.001, dtype=dtype)
    ) * jnp.eye(3, dtype=dtype)

    state = initialize_sbm_state(
        covariance,
        jnp.asarray(0.25, dtype=dtype),
        active_mask,
    )

    tolerance = 2e-6 if dtype_name == "float32" else 1e-12
    assert jnp.allclose(
        state.covariance, expected_covariance, rtol=tolerance, atol=tolerance
    )
    assert jnp.all(jnp.linalg.eigvalsh(state.covariance) > 0.0)
    assert jnp.allclose(
        state.precision,
        jnp.linalg.inv(state.covariance),
        rtol=2e-5 if dtype_name == "float32" else 1e-11,
        atol=2e-5 if dtype_name == "float32" else 1e-11,
    )
