import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

from signals import trend_signal, regime_signal


def cap_weights(w: pd.Series, cap: float = 0.5) -> pd.Series:
    w = w.clip(lower=0)
    w = w / w.sum()
    while (w > cap).any():
        excess = (w[w > cap] - cap).sum()
        w[w > cap] = cap
        w[w <= cap] += excess * (w[w <= cap] / w[w <= cap].sum())
    return w

def compute_cov(returns: pd.DataFrame, method: str = 'sample'):
    if method == 'ledoit':
        lw = LedoitWolf().fit(returns)
        return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    return returns.cov()


def risk_weights(cov: pd.DataFrame, method: str = 'erc'):
    n = len(cov)
    if method == 'erc':
        def obj(w):
            w = np.array(w)
            port_var = w @ cov.values @ w
            mrc = cov.values @ w
            rc = w * mrc
            return ((rc - rc.mean()) ** 2).sum()

        cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1},
                {'type': 'ineq', 'fun': lambda w: w}]
        x0 = np.ones(n) / n
        res = minimize(obj, x0, constraints=cons)
        w = res.x
    elif method == 'minvar':
        def obj(w):
            return w @ cov.values @ w
        cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1},
                {'type': 'ineq', 'fun': lambda w: w}]
        x0 = np.ones(n) / n
        res = minimize(obj, x0, constraints=cons)
        w = res.x
    elif method == 'vol':
        inv_vol = 1 / np.sqrt(np.diag(cov))
        w = inv_vol / inv_vol.sum()
    else:
        raise ValueError('unknown risk method')
    return pd.Series(w, index=cov.index)


def allocate_weights(prices: pd.DataFrame,
                     trend: str = 'mom',
                     risk: str = 'erc',
                     regime: str = 'turbulence',
                     cov_method: str = 'sample',
                     rebalance: str = 'M',
                     threshold: float | None = None,
                     target_vol: float = 0.10,
                     safe_assets: list[int] | None = None) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    trend_sig = trend_signal(prices, method=trend).reindex(returns.index)
    reg = regime_signal(returns, method=regime).reindex(returns.index)
    weights = pd.DataFrame(index=returns.index, columns=prices.columns)
    last_w = None
    if safe_assets is None:
        safe_assets = [1, 3]  # assume bond and gold
    for d in returns.resample(rebalance).last().index:
        date = returns.index[returns.index.get_indexer([d], method='pad')[0]]
        hist = returns.loc[:date]
        cov = compute_cov(hist, cov_method)
        w = risk_weights(cov, method=risk)
        w = w * trend_sig.loc[date]
        if w.sum() == 0:
            w = pd.Series(np.ones(len(w)), index=w.index) / len(w)
        w = cap_weights(w)
        if reg.loc[date, 'state'] < 0.5:
            risky = list(set(range(len(w))) - set(safe_assets))
            w.iloc[risky] = 0
            w.iloc[safe_assets] = w.iloc[safe_assets] / w.iloc[safe_assets].sum()
        port_vol = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
        if risk == 'vol':
            w *= target_vol / port_vol
        if threshold is not None and last_w is not None:
            if np.max(np.abs(w - last_w)) < threshold:
                w = last_w
        weights.loc[date] = w
        last_w = w
    weights = weights.ffill().fillna(0)
    return weights
