import pandas as pd
import numpy as np


def fetch_data(tickers, start, end):
    """Fetch price data for given tickers.

    Parameters
    ----------
    tickers : list[str]
        Asset identifiers. Example: ['CSI300', 'CGB10Y', 'Cu', 'Au'].
        In production these map to Wind tickers like '000300.SH',
        'CGB10Y.YLD', 'Cu.SHF', 'Au.SHF'.
    start, end : str or datetime-like
        Start and end date.

    Returns
    -------
    DataFrame
        Price dataframe indexed by date.

    Notes
    -----
    For reproducibility this function generates synthetic geometric
    Brownian motion data. Replace the marked block with real data
    from WindPy:

    >>> # from WindPy import w
    >>> # w.start()
    >>> # data = w.wsd('000300.SH', 'close', start, end)
    >>> # ...
    """
    dates = pd.date_range(start, end, freq='B')
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(index=dates)
    for i, t in enumerate(tickers):
        mu = 0.05 + 0.02 * i
        sigma = 0.15 + 0.05 * i
        steps = rng.normal((mu - 0.5 * sigma ** 2) / 252, sigma / np.sqrt(252), len(dates))
        prices[t] = 100 * np.exp(np.cumsum(steps))
    return prices
