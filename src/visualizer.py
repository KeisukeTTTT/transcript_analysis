import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Visualizer:
    def __init__(self):
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]

    def visualize_results(self, rouge_l_scores, view_counts, durations, rouge_scores, channel_scores, result_dir):
        self._plot_rouge_histogram_comparison(rouge_l_scores, result_dir)
        self._plot_rouge_histogram_all(rouge_l_scores, result_dir)
        self._plot_view_count_vs_rouge(view_counts, rouge_scores, result_dir)
        self._plot_duration_vs_rouge(durations, rouge_scores, result_dir)
        self._plot_rouge_by_channel(channel_scores, result_dir)

    def _plot_rouge_histogram_comparison(self, rouge_l_scores, result_dir):
        plt.figure(figsize=(12, 7))
        bins = np.linspace(0, 1, 50)
        self.sns.histplot(data=rouge_l_scores["regular"], bins=bins, kde=True, color="blue", alpha=0.5, label="Regular Videos")
        self.sns.histplot(data=rouge_l_scores["short"], bins=bins, kde=True, color="red", alpha=0.5, label="Short Videos")
        plt.xlabel("ROUGE-L Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of ROUGE-L Scores for Regular and Short Videos")
        plt.legend()
        plt.xlim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.savefig(os.path.join(result_dir, "rouge_l_histogram_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_rouge_histogram_all(self, rouge_l_scores, result_dir):
        plt.figure(figsize=(12, 7))
        bins = np.linspace(0, 1, 50)
        self.sns.histplot(data=rouge_l_scores["all"], bins=bins, kde=True, color="green", alpha=0.5, label="All Videos")
        plt.xlabel("ROUGE-L Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of ROUGE-L Scores for All Videos")
        plt.legend()
        plt.xlim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.savefig(os.path.join(result_dir, "rouge_l_histogram_all.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_view_count_vs_rouge(self, view_counts, rouge_scores, result_dir):
        plt.figure(figsize=(10, 6))
        plt.scatter(view_counts, rouge_scores, alpha=0.5)
        plt.xlabel("View Count")
        plt.ylabel("ROUGE-L Score")
        plt.title("View Count vs ROUGE-L Score")
        plt.xscale("log")
        plt.savefig(os.path.join(result_dir, "view_count_vs_rouge_score.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_duration_vs_rouge(self, durations, rouge_scores, result_dir):
        plt.figure(figsize=(10, 6))
        plt.scatter(durations, rouge_scores, alpha=0.5)
        plt.xlabel("Duration (seconds)")
        plt.ylabel("ROUGE-L Score")
        plt.title("Duration vs ROUGE-L Score")
        plt.savefig(os.path.join(result_dir, "duration_vs_rouge_score.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_rouge_by_channel(self, channel_scores, result_dir):
        sorted_channels = sorted(channel_scores.items(), key=lambda x: len(x[1]), reverse=True)
        frequent_channels = OrderedDict(sorted_channels[:10])

        if frequent_channels:
            fig, ax = plt.subplots(figsize=(15, 10))
            for channel, scores in frequent_channels.items():
                self.sns.histplot(scores, kde=True, label=channel, alpha=0.5)
            plt.xlabel("ROUGE-L Score")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of ROUGE-L Scores by Channel")
            plt.legend(title="Channel Name", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "rouge_l_histogram_by_channel.png"), dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(15, 10))
            self.sns.boxplot(data=[scores for scores in frequent_channels.values()], orient="h")
            plt.yticks(range(len(frequent_channels)), list(frequent_channels.keys()))
            plt.xlabel("ROUGE-L Score")
            plt.title(f"ROUGE-L Score Distribution by Channel")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "rouge_l_boxplot_by_channel.png"), dpi=300, bbox_inches="tight")
            plt.close()
