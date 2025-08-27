"""Dynamic Risk Parity mini framework.

Versions:
- v1: turbulence regime, ERC weights, sample covariance, monthly rebalance.
  Suitable for stable markets.
- v2: TVP correlation regime, MinVar with Ledoit-Wolf shrinkage, weekly rebalance.
  Suitable when correlation instability is key.
- v3: Markov switching regime, volatility targeting with threshold rebalance.
  For crisis-sensitive allocation.
"""
import os
from data import fetch_data
from portfolio import allocate_weights
from backtest import run_backtest, equal_weight, static_rp, performance_stats
from report import plot_nav, plot_weights, plot_metric, summary_table


def run_version(name, cfg, prices):
    weights = allocate_weights(prices, trend=cfg['trend'], risk=cfg['risk'],
                               regime=cfg['regime'], cov_method=cfg['cov'],
                               rebalance=cfg['rebalance'], threshold=cfg.get('threshold'))
    result = run_backtest(prices, weights)
    ew = run_backtest(prices, equal_weight(prices))
    rp = run_backtest(prices, static_rp(prices))
    stats = {
        'strategy': performance_stats(result['ret']),
        'equal_weight': performance_stats(ew['ret']),
        'static_rp': performance_stats(rp['ret'])
    }
    outdir = os.path.join('outputs', name)
    os.makedirs(outdir, exist_ok=True)
    plot_nav(result, {'EW': ew, 'SRP': rp}, os.path.join(outdir, 'nav.png'))
    plot_weights(weights, os.path.join(outdir, 'weights.png'))
    plot_metric(allocate_weights.__globals__['regime_signal'](prices.pct_change().dropna(), cfg['regime'])['metric'],
                os.path.join(outdir, 'metric.png'), name=cfg['regime'])
    summary_table(stats, os.path.join(outdir, 'stats.csv'))
    weights.to_csv(os.path.join(outdir, 'weights.csv'))
    result.to_csv(os.path.join(outdir, 'result.csv'))
    yearly = result['ret'].resample('A').apply(lambda x: (1 + x).prod() - 1)
    yearly.to_csv(os.path.join(outdir, 'yearly_returns.csv'))


def main():
    tickers = ['CSI300', 'CGB10Y', 'Cu', 'Au']
    prices = fetch_data(tickers, '2010-01-01', '2023-12-31')
    versions = {
        'v1': {'trend': 'mom', 'risk': 'erc', 'regime': 'turbulence', 'cov': 'sample', 'rebalance': 'M'},
        'v2': {'trend': 'ma', 'risk': 'minvar', 'regime': 'tvp', 'cov': 'ledoit', 'rebalance': 'W'},
        'v3': {'trend': 'kalman', 'risk': 'vol', 'regime': 'msm', 'cov': 'sample', 'rebalance': 'M', 'threshold': 0.05},
    }
    for name, cfg in versions.items():
        run_version(name, cfg, prices)


if __name__ == '__main__':
    main()
