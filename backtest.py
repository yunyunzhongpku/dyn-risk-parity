import pandas as pd
import numpy as np
from portfolio import risk_weights


def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame, cost: float = 0.0005):
    returns = prices.pct_change().reindex(weights.index).fillna(0)
    w = weights.copy()
    w_prev = w.shift(1).fillna(0)
    trades = w.diff().abs().sum(axis=1) * cost
    port_ret = (w_prev * returns).sum(axis=1) - trades
    nav = (1 + port_ret).cumprod()
    return pd.DataFrame({'ret': port_ret, 'nav': nav, 'trades': trades})


def equal_weight(prices: pd.DataFrame):
    n = prices.shape[1]
    return pd.DataFrame(np.ones((len(prices), n)) / n, index=prices.index, columns=prices.columns)


def static_rp(prices: pd.DataFrame):
    cov = prices.pct_change().dropna().cov()
    w = risk_weights(cov, method='erc')
    weights = pd.DataFrame(np.tile(w.values, (len(prices), 1)), index=prices.index, columns=prices.columns)
    return weights


def performance_stats(port_ret: pd.Series):
    nav = (1 + port_ret).cumprod()
    ann_ret = nav.iloc[-1] ** (252 / len(nav)) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    cum = (1 + port_ret).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    mdd = dd.min()
    end = dd[dd == mdd].index[0]
    start = peak[:end][peak == peak[:end].max()].index[-1]
    recovery = len(dd[end:][dd[end:] == 0])
    monthly = port_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
    win_rate = (monthly > 0).mean()
    rd = ann_ret / abs(mdd) if mdd != 0 else np.nan
    return {
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'mdd': mdd,
        'recovery': recovery,
        'win_rate': win_rate,
        'ret_dd': rd,
    }
