import argparse
import json
import os
import re
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from functools import wraps

import cv2
import matplotlib.pyplot as plt

# import MeCab
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yt_dlp
from dotenv import load_dotenv
from google.cloud import vision
from google.oauth2 import service_account
from googleapiclient.discovery import build
from rouge_score import rouge_scorer
from scipy import stats
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()


class YouTubeAnalyzer:
    def __init__(self):
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]

    # ==========================================================================================
    # Video Information
    # ==========================================================================================
    def is_short_video(self, video_id):
        short_url = f"https://www.youtube.com/shorts/{video_id}"
        regular_url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            short_response = requests.head(short_url, allow_redirects=True)
            if short_response.url == short_url:
                return True
            regular_response = requests.head(regular_url, allow_redirects=True)
            if regular_response.url == regular_url:
                return False
            return False
        except Exception:
            return False

    def save_video_info(self, video_ids, output_file):
        results = {}
        for video_id in tqdm(video_ids, desc="Processing video info"):
            try:
                is_short = self.is_short_video(video_id)
                results[video_id] = {"is_short": is_short}
            except Exception as e:
                print(f"Error processing video info for {video_id}: {str(e)}")
                results[video_id] = {"is_short": None, "error": str(e)}

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Video info saved to {output_file}")

    def get_video_details(self, video_id):
        print(f"Getting video details for {video_id}")
        try:
            youtube = build("youtube", "v3", developerKey=os.environ["GOOGLE_API_KEY"])
            request = youtube.videos().list(part="snippet,statistics,contentDetails", id=video_id)
            response = request.execute()

            if "items" in response and len(response["items"]) > 0:
                item = response["items"][0]
                view_count = int(item["statistics"].get("viewCount", 0))
                duration = item["contentDetails"]["duration"]
                duration_seconds = self.parse_duration(duration)
                title = item["snippet"]["title"]
                channel_name = item["snippet"]["channelTitle"]

                return {"view_count": view_count, "duration": duration_seconds, "title": title, "channel_name": channel_name}
            else:
                return None
        except Exception as e:
            print(f"Error getting video details for {video_id}: {str(e)}")
            return None

    def parse_duration(self, duration):
        import re
        from datetime import timedelta

        regex = re.compile(r"P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?")
        match = regex.match(duration)
        if not match:
            return 0
        parts = match.groupdict()
        time_params = {}
        for name, param in parts.items():
            if param:
                time_params[name] = int(param)
        return int(timedelta(**time_params).total_seconds())

    # ==========================================================================================
    # Video Download
    # ==========================================================================================
    def download_videos(self, video_ids, video_dir):
        os.makedirs(video_dir, exist_ok=True)
        for video_id in tqdm(video_ids, desc="Downloading videos"):
            try:
                self.download_video(video_id, video_dir)
            except Exception as e:
                print(f"Error downloading video {video_id}: {str(e)}")
        print(f"Videos downloaded to {video_dir}")

    def download_video(self, video_id, video_dir):
        filename = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(filename):
            ydl_opts = {
                "format": "best",
                "outtmpl": filename,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(video_id, download=True)
        return filename

    # ==========================================================================================
    # Transcripts Processing
    # ==========================================================================================
    def get_transcript(self, video_id):
        try:
            transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=["ja", "en"])
            is_short = self.is_short_video(video_id)

            if is_short:
                filtered_transcripts = [tr for tr in transcripts if tr["start"] <= 60]
            else:
                filtered_transcripts = [tr for tr in transcripts if tr["start"] <= 300]

            transcript_text = " ".join([tr["text"] for tr in filtered_transcripts])
            return transcript_text, filtered_transcripts
        except Exception as e:
            print(f"Error getting transcript for video {video_id}: {str(e)}")
            return None, None

    def save_transcripts(self, video_ids, output_file):
        results = {}
        for video_id in tqdm(video_ids, desc="Getting transcripts"):
            try:
                transcript, _ = self.get_transcript(video_id)
                if transcript:
                    results[video_id] = {"transcript": transcript}
                else:
                    results[video_id] = {"transcript": None, "error": "No transcript available"}
            except Exception as e:
                print(f"Error getting transcript for video {video_id}: {str(e)}")
                results[video_id] = {"transcript": None, "error": str(e)}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Transcripts saved to {output_file}")

    # ==========================================================================================
    # OCR Processing
    # ==========================================================================================
    def detect_text(self, image, min_height=20):
        success, encoded_image = cv2.imencode(".jpg", image)
        content = encoded_image.tobytes()
        image = vision.Image(content=content)
        credentials = service_account.Credentials.from_service_account_file(os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"])
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        response = vision_client.text_detection(image=image)
        text_annotations = response.text_annotations
        text = ""
        for text_annotation in text_annotations[1:]:
            vertices = text_annotation.bounding_poly.vertices
            bounding_box_height = vertices[2].y - vertices[0].y
            if bounding_box_height >= min_height:
                text += text_annotation.description
        return text

    def get_frame_at_timestamp(self, video_path, timestamp):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps) + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def process_video_ocr(self, video_id, transcripts, video_dir, min_height=20, is_short=False):
        filename = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(filename):
            return None

        crop_ratio = 0.5 if is_short else 0.7
        max_duration = 60 if is_short else 300  # 60秒 or 5分

        full_text = ""
        start_time = time.time()
        for transcript in transcripts:
            timestamp = transcript["start"]
            if timestamp > max_duration:
                break

            frame = self.get_frame_at_timestamp(filename, timestamp)
            if frame is not None:
                height, width = frame.shape[:2]
                crop_height = int(height * crop_ratio)
                bottom_region = frame[crop_height:, :]
                text = self.detect_text(bottom_region, min_height)
                full_text += text + " "

            if time.time() - start_time > max_duration:
                break

        return full_text.strip()

    def perform_ocr(self, video_ids, video_info_file, transcript_file, output_file, video_dir):
        with open(video_info_file, "r") as f:
            video_info = json.load(f)

        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        results = {}
        for video_id in tqdm(video_ids, desc="Performing OCR"):
            try:
                if video_id in video_info and video_id in transcript_data:
                    is_short = video_info[video_id]["is_short"]
                    transcript = transcript_data[video_id]["transcript"]
                    try:
                        transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=["ja", "en"])
                    except Exception as e:
                        print(f"Error getting transcript for video {video_id}: {str(e)}")
                        continue

                    ocr_results = {}
                    for height in [20, 30, 40]:
                        result = self.process_video_ocr(video_id, transcripts, video_dir, min_height=height, is_short=is_short)
                        if result:
                            ocr_results[str(height)] = result

                    results[video_id] = {"ocr_results": ocr_results}
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"OCR results saved to {output_file}")

    # ==========================================================================================
    # ROUGE Score Calculation
    # ==========================================================================================
    def load_data(self, video_info_file, transcript_file, ocr_results_file):
        with open(video_info_file, "r") as f:
            video_info = json.load(f)
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        with open(ocr_results_file, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        return video_info, transcript_data, ocr_data

    def calculate_rouge_scores(self, scorer, transcript_data, ocr_data, video_info):
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
                    scores = scorer.score(transcript, ocr_text)
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

                video_details = self.get_video_details(video_id)
                if video_details:
                    if video_id not in results:
                        results[video_id] = {}

                    results[video_id].update(
                        {
                            "view_count": video_details["view_count"],
                            "duration": video_details["duration"],
                            "title": video_details["title"],
                            "channel_name": video_details["channel_name"],
                        }
                    )

            if video_id in results:
                view_counts.append(results[video_id].get("view_count", 0))
                durations.append(results[video_id].get("duration", 0))
                rouge_scores.append(max_rouge_score)

            for video_id, data in results.items():
                if "rouge_scores" in data and "channel_name" in data:
                    rouge_score = max(score["rougel"] for score in data["rouge_scores"].values())
                    channel_scores[data["channel_name"]].append(rouge_score)

        return results, rouge_l_scores, view_counts, durations, rouge_scores, channel_scores

    # ==========================================================================================
    # visualization and statistics
    # ==========================================================================================
    def visualize_results(self, rouge_l_scores, view_counts, durations, rouge_scores, channel_scores, result_dir):
        # ヒストグラムの描画（ショートとレギュラーの比較）
        plt.figure(figsize=(12, 7))
        bins = np.linspace(0, 1, 50)
        sns.histplot(data=rouge_l_scores["regular"], bins=bins, kde=true, color="blue", alpha=0.5, label="regular videos")
        sns.histplot(data=rouge_l_scores["short"], bins=bins, kde=true, color="red", alpha=0.5, label="short videos")
        plt.xlabel("rouge-l score")
        plt.ylabel("frequency")
        plt.title("distribution of rouge-l scores for regular and short videos")
        plt.legend()
        plt.xlim(0, 1)
        plt.grid(true, linestyle="--", alpha=0.7)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.savefig(os.path.join(result_dir, "rouge_l_histogram_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # ヒストグラムの描画（全てのビデオ）
        plt.figure(figsize=(12, 7))
        sns.histplot(data=rouge_l_scores["all"], bins=bins, kde=true, color="green", alpha=0.5, label="all videos")
        plt.xlabel("rouge-l score")
        plt.ylabel("frequency")
        plt.title("distribution of rouge-l scores for all videos")
        plt.legend()
        plt.xlim(0, 1)
        plt.grid(true, linestyle="--", alpha=0.7)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.savefig(os.path.join(result_dir, "rouge_l_histogram_all.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(view_counts, rouge_scores, alpha=0.5)
        plt.xlabel("view count")
        plt.ylabel("rouge-l score")
        plt.title("view count vs rouge-l score")
        plt.xscale("log")
        plt.savefig(os.path.join(result_dir, "view_count_vs_rouge_score.png"), dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(durations, rouge_scores, alpha=0.5)
        plt.xlabel("duration (seconds)")
        plt.ylabel("rouge-l score")
        plt.title("duration vs rouge-l score")
        plt.savefig(os.path.join(result_dir, "duration_vs_rouge_score.png"), dpi=300, bbox_inches="tight")
        plt.close()

        sorted_channels = sorted(channel_scores.items(), key=lambda x: len(x[1]), reverse=true)
        frequent_channels = ordereddict(sorted_channels[:10])

        if frequent_channels:
            fig, ax = plt.subplots(figsize=(15, 10))
            for channel, scores in frequent_channels.items():
                sns.histplot(scores, kde=true, label=channel, alpha=0.5)
            plt.xlabel("rouge-l score")
            plt.ylabel("frequency")
            plt.title(f"distribution of rouge-l scores by channel")
            plt.legend(title="channel name", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "rouge_l_histogram_by_channel.png"), dpi=300, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(15, 10))
            sns.boxplot(data=[scores for scores in frequent_channels.values()], orient="h")
            plt.yticks(range(len(frequent_channels)), list(frequent_channels.keys()))
            plt.xlabel("rouge-l score")
            plt.title(f"rouge-l score distribution by channel")
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, "rouge_l_boxplot_by_channel.png"), dpi=300, bbox_inches="tight")
            plt.close()

    def calculate_statistics(self, rouge_l_scores, view_counts, rouge_scores, durations, channel_scores):
        stats_result = {
            "regular": {
                "mean": np.mean(rouge_l_scores["regular"]),
                "median": np.median(rouge_l_scores["regular"]),
                "std": np.std(rouge_l_scores["regular"]),
                "min": np.min(rouge_l_scores["regular"]),
                "max": np.max(rouge_l_scores["regular"]),
            },
            "short": {
                "mean": np.mean(rouge_l_scores["short"]),
                "median": np.median(rouge_l_scores["short"]),
                "std": np.std(rouge_l_scores["short"]),
                "min": np.min(rouge_l_scores["short"]),
                "max": np.max(rouge_l_scores["short"]),
            },
            "all": {
                "mean": np.mean(rouge_l_scores["all"]),
                "median": np.median(rouge_l_scores["all"]),
                "std": np.std(rouge_l_scores["all"]),
                "min": np.min(rouge_l_scores["all"]),
                "max": np.max(rouge_l_scores["all"]),
            },
        }

        # スピアマンの順位相関係数を計算
        view_count_correlation, view_count_p = stats.spearmanr(view_counts, rouge_scores)
        duration_correlation, duration_p = stats.spearmanr(durations, rouge_scores)
        duration_view_correlation, duration_view_p = stats.spearmanr(durations, view_counts)

        stats_result["correlations"] = {
            "view_count_rouge": {"coefficient": view_count_correlation, "p_value": view_count_p},
            "duration_rouge": {"coefficient": duration_correlation, "p_value": duration_p},
            "duration_view_count": {"coefficient": duration_view_correlation, "p_value": duration_view_p},
        }

        sorted_channels = sorted(channel_scores.items(), key=lambda x: len(x[1]), reverse=true)
        frequent_channels = ordereddict(sorted_channels[:10])
        channel_stats = {}
        for channel, scores in frequent_channels.items():
            channel_stats[channel] = {
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "count": len(scores),
            }
        stats_result["channel_stats"] = channel_stats

        return stats_result

    def calculate_and_visualize_rouge_scores(self, video_info_file, transcript_file, ocr_results_file, output_file, result_dir):
        video_info, transcript_data, ocr_data = self.load_data(video_info_file, transcript_file, ocr_results_file)
        scorer = rouge_scorer.rougescorer(["rougel"], use_stemmer=True)
        results, rouge_l_scores, view_counts, durations, rouge_scores, channel_scores = self.calculate_rouge_scores(
            scorer, transcript_data, ocr_data, video_info
        )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"rouge scores, ocr results, transcripts, and video types saved to {output_file}")

        self.visualize_results(rouge_l_scores, view_counts, durations, rouge_scores, channel_scores, result_dir)

        print(f"visualizations saved in {result_dir}")

        stats = self.calculate_statistics(rouge_l_scores, view_counts, rouge_scores, durations, channel_scores)

        print("\nrouge-l score statistics:")
        for video_type in ["regular videos", "short videos", "all videos"]:
            print(f"\n{video_type}:")
            for key, value in stats[video_type.lower().split()[0]].items():
                print(f"  {key}: {value:.4f}")

        print("\ncorrelations:")
        print("view count vs rouge score:")
        print(f"  spearman's correlation coefficient: {stats['correlations']['view_count_rouge']['coefficient']:.4f}")
        print(f"  p-value: {stats['correlations']['view_count_rouge']['p_value']:.4f}")
        print("duration vs rouge score:")
        print(f"  spearman's correlation coefficient: {stats['correlations']['duration_rouge']['coefficient']:.4f}")
        print(f"  p-value: {stats['correlations']['duration_rouge']['p_value']:.4f}")
        print("duration vs view count:")
        print(f"  spearman's correlation coefficient: {stats['correlations']['duration_view_count']['coefficient']:.4f}")
        print(f"  p-value: {stats['correlations']['duration_view_count']['p_value']:.4f}")

        with open(os.path.join(result_dir, "rouge_l_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

        print(f"\nrouge-l statistics saved to {os.path.join(result_dir, 'rouge_l_stats.json')}")


# ==========================================================================================
# Main Execution
# ==========================================================================================
def main(args):
    analyzer = YouTubeAnalyzer()

    if args.save_video_info:
        if not args.video_list:
            print("Please specify a video list file for saving video info.")
            return
        video_ids = pd.read_csv(args.video_list)["videoId"].unique()
        analyzer.save_video_info(video_ids, os.path.join(args.result_dir, args.video_info_file))

    if args.download_videos:
        with open(os.path.join(args.result_dir, args.video_info_file), "r") as f:
            video_info = json.load(f)
        video_ids = list(video_info.keys())
        analyzer.download_videos(video_ids, args.video_dir)

    if args.save_transcripts:
        with open(os.path.join(args.result_dir, args.video_info_file), "r") as f:
            video_info = json.load(f)
        video_ids = list(video_info.keys())
        analyzer.save_transcripts(video_ids, os.path.join(args.result_dir, args.transcript_file))

    if args.perform_ocr:
        with open(os.path.join(args.result_dir, args.video_info_file), "r") as f:
            video_info = json.load(f)
        video_ids = list(video_info.keys())
        analyzer.perform_ocr(
            video_ids,
            os.path.join(args.result_dir, args.video_info_file),
            os.path.join(args.result_dir, args.transcript_file),
            os.path.join(args.result_dir, args.ocr_results_file),
            args.video_dir,
        )

    if args.calculate_and_visualize_rouge:
        analyzer.calculate_and_visualize_rouge_scores(
            os.path.join(args.result_dir, args.video_info_file),
            os.path.join(args.result_dir, args.transcript_file),
            os.path.join(args.result_dir, args.ocr_results_file),
            os.path.join(args.result_dir, args.rouge_scores_file),
            args.result_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program to compare and analyze YouTube video captions and OCR results")
    parser.add_argument("--video_list", help="Path to CSV file containing list of video IDs to process")
    parser.add_argument("--video_dir", default="youtube-videos", help="Directory to save video files")
    parser.add_argument("--result_dir", default="results", help="Directory to save result files")
    parser.add_argument("--save_video_info", action="store_true", help="Save video info (is_short)")
    parser.add_argument("--download_videos", action="store_true", help="Download videos")
    parser.add_argument("--save_transcripts", action="store_true", help="Save video transcripts")
    parser.add_argument("--perform_ocr", action="store_true", help="Perform OCR on videos")
    parser.add_argument("--video_info_file", default="video_info.json", help="Filename for video info results")
    parser.add_argument("--transcript_file", default="transcripts.json", help="Filename for transcript results")
    parser.add_argument("--ocr_results_file", default="ocr_results.json", help="Filename for OCR results")
    parser.add_argument("--calculate_and_visualize_rouge", action="store_true", help="Calculate and visualize ROUGE scores")
    parser.add_argument("--rouge_scores_file", default="rouge_scores.json", help="Filename for ROUGE scores results")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    main(args)
