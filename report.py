import pandas as pd
import matplotlib.pyplot as plt


def plot_nav(result, benchmarks, path):
    plt.figure(figsize=(10, 6))
    plt.semilogy(result.index, result['nav'], label='Strategy')
    for name, nav in benchmarks.items():
        plt.semilogy(nav.index, nav['nav'], label=name)
    plt.legend()
    plt.title('Net Value')
    plt.savefig(path)
    plt.close()


def plot_weights(weights, path):
    weights.plot.area(figsize=(10, 6))
    plt.title('Weights')
    plt.savefig(path)
    plt.close()


def plot_metric(metric, path, name='metric'):
    metric.plot(figsize=(10, 4))
    plt.title(name)
    plt.savefig(path)
    plt.close()


def summary_table(stats, path):
    df = pd.DataFrame(stats).T
    df.to_csv(path)
    return df
