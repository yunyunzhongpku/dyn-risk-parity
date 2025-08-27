import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data import fetch_data
from signals import trend_signal
from portfolio import allocate_weights
from backtest import run_backtest


def test_fetch_data_shape():
    prices = fetch_data(['a', 'b'], '2020-01-01', '2020-12-31')
    assert list(prices.columns) == ['a', 'b']
    assert len(prices) > 200


def test_trend_signal():
    prices = fetch_data(['a', 'b'], '2020-01-01', '2020-12-31')
    sig = trend_signal(prices, 'mom')
    assert sig.shape == prices.shape
    assert set(sig.values.flatten()) <= {0, 1, float('nan')}


def test_allocate_constraints():
    prices = fetch_data(['a', 'b', 'c', 'd'], '2020-01-01', '2020-06-30')
    w = allocate_weights(prices, regime='tvp')
    sums = w.sum(axis=1).replace(0, np.nan).dropna()
    assert (sums.round(5) == 1).all()
    assert (w >= 0).all().all()
    assert (w <= 0.5 + 1e-6).all().all()


def test_backtest_runs():
    prices = fetch_data(['a', 'b', 'c', 'd'], '2020-01-01', '2020-06-30')
    w = allocate_weights(prices, regime='tvp', rebalance='W')
    result = run_backtest(prices, w)
    assert 'nav' in result.columns
