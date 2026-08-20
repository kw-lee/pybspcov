# Visualization

Install the optional analysis dependencies:

```bash
pip install "pybspcov[analysis]"
```

```python
from pybspcov import (
    plot_posterior_mean,
    plot_quantiles,
    plot_trace,
    save_quantile_plot,
)

plot_trace(model, row=0, column=1)
figure, axis = plot_posterior_mean(model)
figure, axes = plot_quantiles(model, probs=[0.025, 0.5, 0.975])
save_quantile_plot(model, "quantiles.png")
```

All helpers return Matplotlib objects and do not open a GUI. Cross-validation
results provide `result.plot()`, which delegates to `plot_cv`.
