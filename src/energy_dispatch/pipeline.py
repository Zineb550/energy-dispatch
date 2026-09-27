"""
Single entry point: regenerates processed data, forecasts, and scenario
results end to end (spec section 12).

    python -m energy_dispatch.pipeline
"""

from __future__ import annotations

import logging

from energy_dispatch import data, evaluate

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    logger.info("Step 1/4: building processed dataset")
    df = data.build_processed_dataset()

    logger.info("Step 2/4: running rolling-origin forecast backtest")
    # forecast_table = forecast.rolling_origin_backtest(features.build_feature_matrix(df), ...)
    raise NotImplementedError(
        "pipeline wiring is stubbed until forecast.py and evaluate.py are implemented"
    )


if __name__ == "__main__":
    main()
