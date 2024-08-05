import json
import os
from collections import OrderedDict, defaultdict

import numpy as np
from rouge_score import rouge_scorer
from scipy import stats
from tqdm import tqdm

from video_info_manager import VideoInfoManager


class RougeScoreAnalyzer:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def calculate_and_visualize_rouge_scores(self, video_info_file, transcript_file, ocr_results_file, output_file, result_dir):
        video_info, transcript_data, ocr_data = self._load_data(video_info_file, transcript_file, ocr_results_file)
        results, rouge_l_scores, view_counts, durations, rouge_scores, channel_scores = self._calculate_rouge_scores(
            transcript_data, ocr_data, video_info
        )

        self._save_results(results, output_file)
        print(f"ROUGE scores, OCR results, transcripts, and video types saved to {output_file}")

        stats = self._calculate_statistics(rouge_l_scores, view_counts, rouge_scores, durations, channel_scores)
        self._print_statistics(stats)

        self._save_statistics(stats, os.path.join(result_dir, "rouge_l_stats.json"))
        print(f"\nROUGE-L statistics saved to {os.path.join(result_dir, 'rouge_l_stats.json')}")

        return rouge_l_scores, view_counts, durations, rouge_scores, channel_scores

    def _load_data(self, video_info_file, transcript_file, ocr_results_file):
        with open(video_info_file, "r") as f:
            video_info = json.load(f)
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        with open(ocr_results_file, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        return video_info, transcript_data, ocr_data

    def _calculate_rouge_scores(self, transcript_data, ocr_data, video_info):
        results = {}
        rouge_l_scores = {"short": [], "regular": [], "all": []}
        view_counts, durations, rouge_scores = [], [], []
        channel_scores = defaultdict(list)

        for video_id in tqdm(transcript_data.keys(), desc="Calculating ROUGE scores"):
            if video_id in ocr_data and video_id in video_info:
                transcript = transcript_data[video_id]["transcript"]
                ocr_results = ocr_data[video_id]["ocr_results"]
                is_short = video_info[video_id]["is_short"]
                video_type = "short" if is_short else "regular"

                max_rouge_score = 0
                best_height = None
                best_ocr_text = None

                for height, ocr_text in ocr_results.items():
                    scores = self.scorer.score(transcript, ocr_text)
                    rouge_l_score = scores["rougeL"].fmeasure
                    if rouge_l_score > max_rouge_score:
                        max_rouge_score = rouge_l_score
                        best_height = height
                        best_ocr_text = ocr_text

                if best_height:
                    results[video_id] = {
                        "transcript": transcript,
                        "ocr_results": {best_height: best_ocr_text},
                        "rouge_scores": {best_height: {"rougeL": max_rouge_score}},
                        "video_type": video_type,
                    }
                    rouge_l_scores[video_type].append(max_rouge_score)
                    rouge_l_scores["all"].append(max_rouge_score)

                video_info_manager = VideoInfoManager()
                video_details = video_info_manager.get_video_details(video_id)
                if video_details:
                    results[video_id].update(video_details)

            if video_id in results:
                view_counts.append(results[video_id].get("view_count", 0))
                durations.append(results[video_id].get("duration", 0))
                rouge_scores.append(max_rouge_score)

            if "rouge_scores" in results.get(video_id, {}) and "channel_name" in results.get(video_id, {}):
                rouge_score = max(score["rougeL"] for score in results[video_id]["rouge_scores"].values())
                channel_scores[results[video_id]["channel_name"]].append(rouge_score)

        return results, rouge_l_scores, view_counts, durations, rouge_scores, channel_scores

    def _calculate_statistics(self, rouge_l_scores, view_counts, rouge_scores, durations, channel_scores):
        stats_result = {
            "regular": self._calculate_basic_stats(rouge_l_scores["regular"]),
            "short": self._calculate_basic_stats(rouge_l_scores["short"]),
            "all": self._calculate_basic_stats(rouge_l_scores["all"]),
        }

        # Calculate Spearman's rank correlation
        view_count_correlation, view_count_p = stats.spearmanr(view_counts, rouge_scores)
        duration_correlation, duration_p = stats.spearmanr(durations, rouge_scores)
        duration_view_correlation, duration_view_p = stats.spearmanr(durations, view_counts)

        stats_result["correlations"] = {
            "view_count_rouge": {"coefficient": view_count_correlation, "p_value": view_count_p},
            "duration_rouge": {"coefficient": duration_correlation, "p_value": duration_p},
            "duration_view_count": {"coefficient": duration_view_correlation, "p_value": duration_view_p},
        }

        sorted_channels = sorted(channel_scores.items(), key=lambda x: len(x[1]), reverse=True)
        frequent_channels = OrderedDict(sorted_channels[:10])
        channel_stats = {}
        for channel, scores in frequent_channels.items():
            channel_stats[channel] = self._calculate_basic_stats(scores)
            channel_stats[channel]["count"] = len(scores)
        stats_result["channel_stats"] = channel_stats

        return stats_result

    def _calculate_basic_stats(self, data):
        return {
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
        }

    def _print_statistics(self, stats):
        print("\nROUGE-L Score Statistics:")
        for video_type in ["Regular Videos", "Short Videos", "All Videos"]:
            print(f"\n{video_type}:")
            for key, value in stats[video_type.lower().split()[0]].items():
                print(f"  {key}: {value:.4f}")

        print("\nCorrelations:")
        print("View Count vs ROUGE Score:")
        print(f"  Spearman's Correlation Coefficient: {stats['correlations']['view_count_rouge']['coefficient']:.4f}")
        print(f"  p-value: {stats['correlations']['view_count_rouge']['p_value']:.4f}")
        print("Duration vs ROUGE Score:")
        print(f"  Spearman's Correlation Coefficient: {stats['correlations']['duration_rouge']['coefficient']:.4f}")
        print(f"  p-value: {stats['correlations']['duration_rouge']['p_value']:.4f}")
        print("Duration vs View Count:")
        print(f"  Spearman's Correlation Coefficient: {stats['correlations']['duration_view_count']['coefficient']:.4f}")
        print(f"  p-value: {stats['correlations']['duration_view_count']['p_value']:.4f}")

    def _save_results(self, results, output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def _save_statistics(self, stats, output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
