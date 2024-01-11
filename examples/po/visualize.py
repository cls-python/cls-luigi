import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os.path import join as pjoin
from seaborn.objects import Text


def load_summaries_df(path="results_analysis/raw_summaries.csv"):
    return pd.read_csv(path)


def draw_regret_facet_grid(summaries_df, save_to_path="results_analysis", title=""):
    sns.set_style("whitegrid")

    summaries_df = summaries_df.rename(columns={
        "degree": "Polynomial Degree",
        "regressor": "Regressor",
        "regret": "Normalized Regret",
        "noise": "Noise Half−width",
        "training_size": "Training Set Size"
    })

    g = sns.FacetGrid(summaries_df, col="Noise Half−width", row="Training Set Size", aspect=1.4)
    g.map_dataframe(sns.boxplot, x="Polynomial Degree", y="Normalized Regret", hue="Regressor",
                    palette=["#78CEB5", "#FFBAC8"], fliersize=2, fill=True)
    g.add_legend()
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

    g.set(ylim=(-0.01, 0.26))
    g.fig.suptitle(title)
    g.tight_layout()
    g.savefig(pjoin(save_to_path, "facet_plot.png"), dpi=500)


if __name__ == "__main__":
    summaries_df = load_summaries_df()
    draw_regret_facet_grid(summaries_df, title="Regret for the Shortest Path Problems\n\n")
