from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.extend import core as jax_core

from pybspcov import kernels
from pybspcov.kernels import sbm
from pybspcov.sampling.gig import GIGSample

ACTIVE_MASK = jnp.asarray(
    [
        [False, True, False, True],
        [True, False, True, False],
        [False, True, False, False],
        [True, False, False, False],
    ]
)
OTHER_INDICES = jnp.asarray(
    [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
    dtype=jnp.int32,
)


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
    return covariance, jnp.linalg.inv(covariance), x.T @ x, tau


def _compact_parameters(
    *,
    dtype: jnp.dtype,
    column: int,
    structure: sbm.SBMCompactStructure,
) -> sbm.SBMCompactColumnParameters:
    covariance, precision, scatter, tau = _column_case(dtype)
    return sbm.compact_sbm_column_parameters(
        covariance=covariance,
        precision=precision,
        scatter=scatter,
        tau=tau,
        column=jnp.asarray(column, dtype=jnp.int32),
        structure=structure,
        n_observations=jnp.asarray(6),
        diagonal_rate=jnp.asarray(1.0, dtype=dtype),
        gamma=jnp.asarray(0.9, dtype=dtype),
    )


def test_compact_sbm_public_api_exports_the_structure_and_conditionals() -> None:
    assert kernels.SBMCompactStructure is sbm.SBMCompactStructure
    assert kernels.SBMCompactColumnParameters is sbm.SBMCompactColumnParameters
    assert kernels.prepare_sbm_compact_structure is sbm.prepare_sbm_compact_structure
    assert kernels.compact_sbm_column_parameters is sbm.compact_sbm_column_parameters
    assert kernels.compact_sbm_sweep is sbm.compact_sbm_sweep
    assert kernels.sample_compact_sbm_chain is sbm.sample_compact_sbm_chain


def test_prepare_sbm_compact_structure_maps_literal_active_lanes() -> None:
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)

    assert jnp.array_equal(structure.other_indices, OTHER_INDICES)
    assert jnp.array_equal(
        structure.active_positions,
        jnp.asarray([[0, 2], [0, 1], [1, 0], [0, 0]], dtype=jnp.int32),
    )
    assert jnp.array_equal(
        structure.lane_mask,
        jnp.asarray([[True, True], [True, True], [True, False], [True, False]]),
    )


def test_prepare_sbm_compact_structure_preserves_zero_width() -> None:
    structure = sbm.prepare_sbm_compact_structure(
        jnp.zeros((4, 4), dtype=jnp.bool_),
        OTHER_INDICES,
    )

    assert jnp.array_equal(structure.other_indices, OTHER_INDICES)
    assert structure.active_positions.shape == (4, 0)
    assert structure.active_positions.dtype == jnp.int32
    assert structure.lane_mask.shape == (4, 0)
    assert structure.lane_mask.dtype == jnp.bool_


def test_prepare_sbm_compact_structure_preserves_other_indices_device() -> None:
    cpu_devices = jax.devices("cpu")
    target_device = cpu_devices[-1]
    other_indices = jax.device_put(OTHER_INDICES, target_device)

    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, other_indices)

    expected_devices = other_indices.devices()
    for field_name, value in zip(
        sbm.SBMCompactStructure._fields,
        structure,
        strict=True,
    ):
        assert value.devices() == expected_devices, field_name


@pytest.mark.parametrize(
    ("other_indices", "exception", "message"),
    [
        (jnp.zeros((4, 2), dtype=jnp.int32), ValueError, "shape"),
        (
            jnp.asarray(
                [[1, 1, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                dtype=jnp.int32,
            ),
            ValueError,
            "exactly once",
        ),
        (
            jnp.asarray(
                [[0, 1, 2], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                dtype=jnp.int32,
            ),
            ValueError,
            "exclude its column",
        ),
        (
            jnp.asarray(
                [[1, 2, 4], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
                dtype=jnp.int32,
            ),
            ValueError,
            "between zero",
        ),
        (OTHER_INDICES.astype(jnp.float32), TypeError, "integer"),
    ],
)
def test_prepare_sbm_compact_structure_rejects_invalid_index_tables(
    other_indices: jax.Array,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        sbm.prepare_sbm_compact_structure(ACTIVE_MASK, other_indices)


def test_compact_sbm_structure_keeps_positions_bound_to_its_index_table() -> None:
    canonical_structure = sbm.prepare_sbm_compact_structure(
        ACTIVE_MASK,
        OTHER_INDICES,
    )
    reversed_structure = sbm.prepare_sbm_compact_structure(
        ACTIVE_MASK,
        OTHER_INDICES[:, ::-1],
    )

    canonical = _compact_parameters(
        dtype=jnp.float32,
        column=1,
        structure=canonical_structure,
    )
    reversed_result = _compact_parameters(
        dtype=jnp.float32,
        column=1,
        structure=reversed_structure,
    )

    assert jnp.allclose(
        canonical.gamma_chi,
        reversed_result.gamma_chi,
        rtol=2e-5,
        atol=2e-5,
    )
    assert jnp.allclose(
        canonical.beta_mean,
        reversed_result.beta_mean[::-1],
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_compact_sbm_conditionals_match_masked_dense_active_lanes(
    dtype_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    tolerance = 2e-5 if dtype_name == "float32" else 1e-11
    covariance, precision, scatter, tau = _column_case(dtype)
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)
    column = jnp.asarray(2, dtype=jnp.int32)
    common = {
        "covariance": covariance,
        "precision": precision,
        "scatter": scatter,
        "tau": tau,
        "column": column,
        "n_observations": jnp.asarray(6),
        "diagonal_rate": jnp.asarray(1.0, dtype=dtype),
        "gamma": jnp.asarray(0.9, dtype=dtype),
    }

    masked = sbm.sbm_column_parameters(
        **common,
        active_mask=ACTIVE_MASK,
        other_indices=OTHER_INDICES[2],
    )
    compact = jax.jit(sbm.compact_sbm_column_parameters)(
        **common,
        structure=structure,
    )

    assert jnp.array_equal(compact.lane_mask, jnp.asarray([True, False]))
    assert compact.active_count == 1
    assert jnp.allclose(
        compact.conditional_precision,
        masked.conditional_precision,
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.conditional_scatter[0],
        masked.conditional_scatter[1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.quadratic[0, 0],
        masked.quadratic[1, 1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.beta_precision[0, 0],
        masked.beta_precision[1, 1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.beta_mean[0],
        masked.beta_mean[1],
        rtol=tolerance,
        atol=tolerance,
    )
    assert jnp.allclose(
        compact.gamma_chi,
        masked.gamma_chi,
        rtol=tolerance,
        atol=tolerance,
    )
    assert compact.conditional_scatter[1] == 0.0
    assert compact.beta_mean[1] == 0.0
    assert jnp.array_equal(
        compact.beta_precision[1],
        jnp.asarray([0.0, 1.0], dtype=dtype),
    )
    assert jnp.array_equal(
        compact.beta_precision[:, 1],
        jnp.asarray([0.0, 1.0], dtype=dtype),
    )
    assert compact.beta_precision.dtype == dtype
    assert compact.beta_mean.dtype == dtype


def test_compact_sbm_conditionals_support_jit_of_vmap_over_columns() -> None:
    dtype = jnp.float32
    covariance, precision, scatter, tau = _column_case(dtype)
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)

    def column_parameters(column: jax.Array) -> sbm.SBMCompactColumnParameters:
        return sbm.compact_sbm_column_parameters(
            covariance=covariance,
            precision=precision,
            scatter=scatter,
            tau=tau,
            column=column,
            structure=structure,
            n_observations=jnp.asarray(6),
            diagonal_rate=jnp.asarray(1.0, dtype=dtype),
            gamma=jnp.asarray(0.9, dtype=dtype),
        )

    parameters = jax.jit(jax.vmap(column_parameters))(jnp.arange(4, dtype=jnp.int32))

    assert parameters.conditional_scatter.shape == (4, 2)
    assert parameters.beta_precision.shape == (4, 2, 2)
    assert jnp.array_equal(parameters.lane_mask, structure.lane_mask)
    assert jnp.array_equal(parameters.active_count, jnp.asarray([2, 2, 1, 1]))


def test_compact_sbm_conditionals_preserve_zero_width() -> None:
    dtype = jnp.float32
    structure = sbm.prepare_sbm_compact_structure(
        jnp.zeros((4, 4), dtype=jnp.bool_),
        OTHER_INDICES,
    )

    parameters = jax.jit(
        lambda: _compact_parameters(dtype=dtype, column=1, structure=structure)
    )()

    assert parameters.active_count == 0
    assert parameters.conditional_scatter.shape == (0,)
    assert parameters.quadratic.shape == (0, 0)
    assert parameters.beta_precision.shape == (0, 0)
    assert parameters.beta_mean.shape == (0,)
    assert parameters.gamma_chi == _column_case(dtype)[2][1, 1]


def _nested_jaxprs(value: object) -> list[jax_core.Jaxpr]:
    found: list[jax_core.Jaxpr] = []
    if isinstance(value, jax_core.ClosedJaxpr):
        found.append(value.jaxpr)
    elif isinstance(value, jax_core.Jaxpr):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_nested_jaxprs(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.extend(_nested_jaxprs(item))
    return found


def test_compact_sbm_conditionals_factor_and_form_quadratic_at_compact_width() -> None:
    dtype = jnp.float32
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)
    closed = jax.make_jaxpr(
        lambda: _compact_parameters(dtype=dtype, column=1, structure=structure)
    )()
    pending = [closed.jaxpr]
    factor_shapes: list[tuple[int, ...]] = []
    matrix_products: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    while pending:
        current = pending.pop()
        for equation in current.eqns:
            if equation.primitive.name in {"cholesky", "lu", "triangular_solve"}:
                factor_shapes.extend(
                    variable.aval.shape
                    for variable in equation.invars
                    if hasattr(variable.aval, "shape") and len(variable.aval.shape) == 2
                )
            if equation.primitive.name == "dot_general":
                operand_shapes = [
                    variable.aval.shape
                    for variable in equation.invars[:2]
                    if hasattr(variable.aval, "shape")
                ]
                if len(operand_shapes) == 2 and all(
                    len(shape) == 2 for shape in operand_shapes
                ):
                    matrix_products.append((operand_shapes[0], operand_shapes[1]))
            for parameter in equation.params.values():
                pending.extend(_nested_jaxprs(parameter))

    assert (2, 2) in factor_shapes
    assert (3, 3) not in factor_shapes
    assert ((3, 3), (3, 3)) not in matrix_products


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("mask_name", ["partial", "full"])
def test_compact_sbm_sweep_matches_successful_masked_dense_fixed_key(
    dtype_name: str,
    mask_name: str,
) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    tolerance = 4e-5 if dtype_name == "float32" else 2e-10
    covariance, _, scatter, _ = _column_case(dtype)
    active_mask = (
        ACTIVE_MASK if mask_name == "partial" else ~jnp.eye(4, dtype=jnp.bool_)
    )
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, active_mask)
    structure = sbm.prepare_sbm_compact_structure(active_mask, OTHER_INDICES)
    shared = (
        scatter,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
    )
    key = jax.random.key(211)

    masked = jax.jit(sbm.sbm_sweep)(
        key, state, scatter, OTHER_INDICES, *shared[1:], active_mask
    )
    compact = jax.jit(sbm.compact_sbm_sweep)(key, state, *shared, structure)

    assert masked.accepted
    assert compact.accepted
    for actual, expected in zip(compact.state, masked.state, strict=True):
        assert jnp.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    excluded = (~active_mask) & (~jnp.eye(4, dtype=jnp.bool_))
    assert jnp.all(compact.state.covariance[excluded] == 0.0)
    assert jnp.all(jnp.linalg.eigvalsh(compact.state.covariance) > 0.0)
    assert jnp.allclose(
        compact.state.precision,
        jnp.linalg.inv(compact.state.covariance),
        rtol=tolerance,
        atol=tolerance,
    )


def test_compact_sbm_sweep_uses_structure_owned_other_indices() -> None:
    dtype = jnp.float32
    covariance, _, scatter, _ = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, ACTIVE_MASK)
    reversed_indices = OTHER_INDICES[:, ::-1]
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, reversed_indices)
    key = jax.random.key(211)

    masked = jax.jit(sbm.sbm_sweep)(
        key,
        state,
        scatter,
        reversed_indices,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        ACTIVE_MASK,
    )
    compact = jax.jit(sbm.compact_sbm_sweep)(
        key,
        state,
        scatter,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        structure,
    )

    assert masked.accepted
    assert compact.accepted
    for actual, expected in zip(compact.state, masked.state, strict=True):
        assert jnp.allclose(actual, expected, rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("compiled", [False, True])
def test_compact_sbm_sweep_zero_width_skips_phi_gig_and_preserves_scales(
    compiled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtype = jnp.float32
    covariance, _, scatter, _ = _column_case(dtype)
    active_mask = jnp.zeros((4, 4), dtype=jnp.bool_)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, active_mask)
    scale_pattern = jnp.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ],
        dtype=dtype,
    )
    state = state._replace(
        phi=jnp.asarray(1.5, dtype=dtype) + scale_pattern,
        psi=jnp.asarray(2.5, dtype=dtype) + scale_pattern,
        tau=jnp.asarray(0.25, dtype=dtype) + scale_pattern,
    )
    structure = sbm.prepare_sbm_compact_structure(active_mask, OTHER_INDICES)

    def fail_if_phi_gig_runs(*args: object, **kwargs: object) -> None:
        raise AssertionError("zero-width compact sweep must skip phi GIG")

    monkeypatch.setattr(sbm, "_sample_gig_batch", fail_if_phi_gig_runs)

    def run() -> object:
        return sbm.compact_sbm_sweep(
            jax.random.key(223),
            state,
            scatter,
            jnp.asarray(6),
            jnp.asarray(0.5, dtype=dtype),
            jnp.asarray(0.5, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
            tau1sq,
            structure,
        )

    if compiled:
        result = jax.jit(run)()
    else:
        with jax.disable_jit():
            result = run()

    assert result.accepted
    assert jnp.array_equal(result.state.phi, state.phi)
    assert jnp.array_equal(result.state.psi, state.psi)
    assert jnp.array_equal(result.state.tau, state.tau)


def test_compact_sbm_sweep_factors_only_compact_width() -> None:
    dtype = jnp.float32
    covariance, _, scatter, _ = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, ACTIVE_MASK)
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)
    closed = jax.make_jaxpr(sbm.compact_sbm_sweep)(
        jax.random.key(227),
        state,
        scatter,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        structure,
    )
    pending = [closed.jaxpr]
    factor_shapes: list[tuple[int, ...]] = []
    matrix_products: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    while pending:
        current = pending.pop()
        for equation in current.eqns:
            if equation.primitive.name in {"cholesky", "lu", "triangular_solve"}:
                factor_shapes.extend(
                    variable.aval.shape
                    for variable in equation.invars
                    if hasattr(variable.aval, "shape") and len(variable.aval.shape) == 2
                )
            if equation.primitive.name == "dot_general":
                operand_shapes = [
                    variable.aval.shape
                    for variable in equation.invars[:2]
                    if hasattr(variable.aval, "shape")
                ]
                if len(operand_shapes) == 2 and all(
                    len(shape) == 2 for shape in operand_shapes
                ):
                    matrix_products.append((operand_shapes[0], operand_shapes[1]))
            for parameter in equation.params.values():
                pending.extend(_nested_jaxprs(parameter))

    assert (2, 2) in factor_shapes
    assert (3, 3) not in factor_shapes
    assert ((3, 3), (3, 3)) not in matrix_products


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_sample_compact_sbm_chain_matches_explicit_sweeps(dtype_name: str) -> None:
    if dtype_name == "float64" and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX_ENABLE_X64=1")
    dtype = getattr(jnp, dtype_name)
    covariance, _, scatter, _ = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, ACTIVE_MASK)
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)
    arguments = (
        scatter,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        structure,
    )
    key = jax.random.key(229)
    expected_state = state
    expected_covariance = []
    expected_phi = []
    expected_accepted = []
    run_sweep = jax.jit(sbm.compact_sbm_sweep)
    for sweep_key in jax.random.split(key, 5):
        sweep = run_sweep(sweep_key, expected_state, *arguments)
        expected_state = sweep.state
        expected_covariance.append(expected_state.covariance)
        expected_phi.append(expected_state.phi)
        expected_accepted.append(sweep.accepted)

    result = jax.jit(
        sbm.sample_compact_sbm_chain,
        static_argnames=("burnin", "n_samples"),
    )(
        key,
        state,
        *arguments,
        burnin=2,
        n_samples=3,
    )

    assert result.covariance.shape == (3, 4, 4)
    assert result.phi.shape == (3, 4, 4)
    assert result.accepted.shape == (5,)
    assert jnp.array_equal(result.covariance, jnp.stack(expected_covariance[2:]))
    assert jnp.array_equal(result.phi, jnp.stack(expected_phi[2:]))
    assert jnp.array_equal(result.accepted, jnp.stack(expected_accepted))
    for actual, expected in zip(result.final_state, expected_state, strict=True):
        assert jnp.array_equal(actual, expected)


def _compact_column_gamma_key(sweep_key: jax.Array, column: int) -> jax.Array:
    current_key = sweep_key
    gamma_key = sweep_key
    for _ in range(column + 1):
        current_key, gamma_key, _, _, _ = jax.random.split(current_key, 5)
    return gamma_key


def test_sample_compact_sbm_chain_self_transitions_then_continues_after_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtype = jnp.float32
    covariance, _, scatter, _ = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, ACTIVE_MASK)
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)
    key = jax.random.key(233)
    arguments = (
        scatter,
        jnp.asarray(6),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        tau1sq,
        structure,
    )
    sweep_keys = jax.random.split(key, 3)
    rejected_key_data = jax.random.key_data(
        _compact_column_gamma_key(sweep_keys[1], column=1)
    )

    def reject_one_gamma(
        draw_key: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del lambda_, psi
        accepted = ~jnp.all(jax.random.key_data(draw_key) == rejected_key_data)
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=accepted,
            iterations=jnp.asarray(1, dtype=jnp.int32),
        )

    def accept_phi(
        draw_keys: jax.Array,
        lambda_: jax.Array,
        chi: jax.Array,
        psi: jax.Array,
    ) -> GIGSample:
        del draw_keys, lambda_, psi
        return GIGSample(
            value=jnp.ones_like(chi),
            accepted=jnp.ones_like(chi, dtype=jnp.bool_),
            iterations=jnp.ones_like(chi, dtype=jnp.int32),
        )

    monkeypatch.setattr(sbm, "sample_gig", reject_one_gamma)
    monkeypatch.setattr(sbm, "_sample_gig_batch", accept_phi)

    def isolated_sweep(
        sweep_key: jax.Array,
        current_state: sbm.BMState,
    ) -> sbm.BMSweepResult:
        return sbm.compact_sbm_sweep(sweep_key, current_state, *arguments)

    def isolated_chain(
        chain_key: jax.Array,
        initial_state: sbm.BMState,
    ) -> sbm.SBMChainResult:
        return sbm.sample_compact_sbm_chain(
            chain_key,
            initial_state,
            *arguments,
            burnin=0,
            n_samples=3,
        )

    run_sweep = jax.jit(isolated_sweep)
    first = run_sweep(sweep_keys[0], state)
    rejected = run_sweep(sweep_keys[1], first.state)
    expected_final = run_sweep(sweep_keys[2], rejected.state)

    assert first.accepted
    assert not rejected.accepted
    assert expected_final.accepted

    result = jax.jit(isolated_chain)(key, state)

    assert jnp.array_equal(result.accepted, jnp.asarray([True, False, True]))
    assert jnp.array_equal(result.covariance[1], result.covariance[0])
    assert jnp.array_equal(result.phi[1], result.phi[0])
    assert not jnp.array_equal(result.covariance[2], result.covariance[1])
    for rejected_value, first_value in zip(rejected.state, first.state, strict=True):
        assert jnp.array_equal(rejected_value, first_value)
    for actual, expected in zip(result.final_state, expected_final.state, strict=True):
        assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("burnin", "n_samples", "message"),
    [
        (-1, 1, "burnin must be non-negative"),
        (0, 0, "n_samples must be positive"),
    ],
)
def test_sample_compact_sbm_chain_validates_static_lengths(
    burnin: int,
    n_samples: int,
    message: str,
) -> None:
    dtype = jnp.float32
    covariance, _, scatter, _ = _column_case(dtype)
    tau1sq = jnp.asarray(0.15, dtype=dtype)
    state = sbm.initialize_sbm_state(covariance, tau1sq, ACTIVE_MASK)
    structure = sbm.prepare_sbm_compact_structure(ACTIVE_MASK, OTHER_INDICES)

    with pytest.raises(ValueError, match=message):
        sbm.sample_compact_sbm_chain(
            jax.random.key(239),
            state,
            scatter,
            jnp.asarray(6),
            jnp.asarray(0.5, dtype=dtype),
            jnp.asarray(0.5, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
            tau1sq,
            structure,
            burnin=burnin,
            n_samples=n_samples,
        )
