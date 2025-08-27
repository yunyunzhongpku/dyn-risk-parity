import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def _kalman_slope(series):
    level = 0.0
    slope = 0.0
    P = np.eye(2)
    F = np.array([[1, 1], [0, 1]])
    Q = np.eye(2) * 1e-5
    H = np.array([[1, 0]])
    R = np.array([[1e-2]])
    slopes = []
    for y in series.values:
        x_pred = F @ np.array([level, slope])
        P = F @ P @ F.T + Q
        y_pred = H @ x_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x_pred + (K @ (y - y_pred)).flatten()
        P = (np.eye(2) - K @ H) @ P
        level, slope = x
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def trend_signal(prices: pd.DataFrame, method: str = 'mom') -> pd.DataFrame:
    prices = prices.copy()
    if method == 'mom':
        r12 = prices.pct_change(252)
        r2 = prices.pct_change(42)
        signal = (r12 - r2) > 0
    elif method == 'ma':
        short = prices.rolling(50).mean()
        long = prices.rolling(200).mean()
        signal = short > long
    elif method == 'kalman':
        slopes = pd.DataFrame({c: _kalman_slope(prices[c]) for c in prices})
        signal = slopes > 0
    else:
        raise ValueError('unknown trend method')
    return signal.astype(float)


def turbulence_index(returns, window=252):
    means = returns.rolling(window).mean()
    covs = returns.rolling(window).cov().dropna()
    turb = []
    for i, date in enumerate(returns.index):
        if date not in means.index or date not in covs.index:
            turb.append(np.nan)
            continue
        r = returns.loc[date] - means.loc[date]
        cov = covs.xs(date)
        try:
            d = r.values @ np.linalg.inv(cov.values) @ r.values.T
        except np.linalg.LinAlgError:
            d = np.nan
        turb.append(d)
    return pd.Series(turb, index=returns.index)


def tvp_correlation(r1, r2, lam=0.94):
    cov = var1 = var2 = 0.0
    corrs = []
    for x, y in zip(r1, r2):
        cov = lam * cov + (1 - lam) * x * y
        var1 = lam * var1 + (1 - lam) * x ** 2
        var2 = lam * var2 + (1 - lam) * y ** 2
        if var1 > 0 and var2 > 0:
            corrs.append(cov / np.sqrt(var1 * var2))
        else:
            corrs.append(0.0)
    return pd.Series(corrs, index=r1.index)


def regime_signal(returns: pd.DataFrame, method: str = 'turbulence'):
    if method == 'turbulence':
        turb = turbulence_index(returns)
        th = turb.quantile(0.9)
        state = (turb < th).astype(float)
        return pd.DataFrame({'state': state, 'metric': turb})
    elif method == 'tvp':
        corr = tvp_correlation(returns.iloc[:, 0], returns.iloc[:, 1])
        state = (corr < 0.3).astype(float)
        return pd.DataFrame({'state': state, 'metric': corr})
    elif method == 'msm':
        mod = MarkovRegression(returns.iloc[:, 0], k_regimes=2, trend='c', switching_variance=True)
        res = mod.fit(disp=False)
        prob = res.smoothed_marginal_probabilities[0]
        state = (prob < 0.5).astype(float)
        return pd.DataFrame({'state': state, 'metric': prob})
    else:
        raise ValueError('unknown regime method')
