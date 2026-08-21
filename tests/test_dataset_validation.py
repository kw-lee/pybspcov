import numpy as np
import pytest

from pybspcov import preprocess_colon


def test_preprocess_colon_rejects_zero_tissue_labels() -> None:
    colon = np.arange(1.0, 25.0).reshape(6, 4)
    tissues = np.asarray([1, 1, -1, -1, 0, 1])

    with pytest.raises(ValueError, match="finite non-zero signed labels"):
        preprocess_colon(colon, tissues, n_features=2)
