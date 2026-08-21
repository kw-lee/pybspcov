# Example datasets

Small compressed copies of all three upstream data objects ship with the
wheel, so loading never requires a network connection.

```python
from pybspcov import load_colon, load_sp500, preprocess_colon

colon = load_colon()
processed = preprocess_colon(colon.data, colon.target)
assert processed.X.shape == (62, 50)

prices = load_sp500()
assert prices.data.dtype.names == ("symbol", "date", "adjusted", "sector")
```

Loaders return `DatasetBunch`, a dictionary whose keys are also accessible as
attributes. Each bunch includes its description, upstream version, source,
and resource checksum. `load_colon(return_X_y=True)` returns the array pair
directly. Install the `data` extra to request pandas frames with
`as_frame=True`.

`load_colon` follows the Python samples-by-features convention and therefore
transposes the upstream R `colon` object. The raw expression data remain
untransformed until `preprocess_colon` is called. Tissue labels must be finite non-zero signed values.

`preprocess_sp500` follows the upstream sector-then-symbol column order and
includes the first observed month using `quantmod::periodReturn` semantics. A
committed R fixture validates monthly returns and the fixed-factor residual
path. The optional automatic factor count uses the documented spectral-ratio
selector rather than `hdbinseg`'s `ah` criterion; pass `n_factors` for a
reproducible cross-language factor rank.
