import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

palette = sns.color_palette("RdBu_r", 7)


def _build_dataframe(matrix, xlabel, ylabel):
    data = []
    for rid, row in enumerate(matrix):
        for cid, ele in enumerate(row):
            data.append({xlabel: cid, ylabel: rid, "value": ele})
    data = pd.DataFrame(data)
    data = data.pivot(ylabel, xlabel, "value")
    return data


def draw_heatmap(matrix1, xlabel="x", ylabel="y", dpi=72, matrix2=None, title=None):
    data = _build_dataframe(matrix1, xlabel, ylabel)
    if matrix2 is None:
        f, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
        sns.heatmap(
            data, annot=False, linewidths=.0, square=False, center=0., ax=ax
        )
    else:
        fig = plt.figure(figsize=(8, 20), dpi=dpi)
        ax1 = fig.add_subplot(121)
        sns.heatmap(
            data,
            annot=False,
            linewidths=.0,
            square=True,
            ax=ax1,
            cmap="RdBu_r",
            center=0.
        )
        data2 = _build_dataframe(matrix2, xlabel, ylabel)
        ax2 = fig.add_subplot(122)
        sns.heatmap(
            data2,
            annot=False,
            linewidths=.0,
            square=True,
            ax=ax2,
            cmap="Blues_r",
            vmin=0.0,
            vmax=1.0
        )
    if title is not None:
        plt.title(title)
