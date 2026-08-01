"""Pure JAX kernels shared by the BM and SBM samplers."""

from pybspcov.kernels.covariance import update_covariance_column

__all__ = ["update_covariance_column"]
