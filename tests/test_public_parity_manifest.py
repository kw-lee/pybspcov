from pathlib import Path

import pybspcov


def test_every_bspcov_1_0_3_public_feature_has_api_and_documentation() -> None:
    mapping = {
        "bandPPP": "BandPPP",
        "bmspcov": "BMSPCov",
        "cv.bandPPP": "cross_validate_band_ppp",
        "cv.thresPPP": "cross_validate_threshold_ppp",
        "estimate": "estimate()",
        "plot.bspcov": "plot_trace",
        "plot.postmean.bspcov": "plot_posterior_mean",
        "plot.quantile.bspcov": "plot_quantiles",
        "proc_colon": "preprocess_colon",
        "proc_SP500": "preprocess_sp500",
        "quantile.bspcov": "quantile()",
        "save_quantile_plot": "save_quantile_plot",
        "sbmspcov": "SBMSPCov",
        "summary.bspcov": "summary()",
        "thresPPP": "ThresholdPPP",
        "colon": "load_colon",
        "tissues": "load_colon",
        "SP500": "load_sp500",
    }
    documentation = Path("docs/source/parity.md").read_text(encoding="utf-8")

    for r_name, python_name in mapping.items():
        assert f"`{r_name}`" in documentation
        assert f"`{python_name}`" in documentation

    root_symbols = {
        value.removesuffix("()")
        for value in mapping.values()
        if value not in {"estimate()", "quantile()", "summary()"}
    }
    for symbol in root_symbols:
        assert hasattr(pybspcov, symbol), symbol
