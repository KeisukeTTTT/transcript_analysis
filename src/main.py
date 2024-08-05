import argparse
import os

import pandas as pd

from ocr_manager import OCRManager
from rouge_score_analyzer import RougeScoreAnalyzer
from transcript_manager import TranscriptManager
from video_info_manager import VideoInfoManager
from visualizer import Visualizer


class YouTubeAnalyzer:
    def __init__(self):
        self.video_info_manager = VideoInfoManager()
        self.transcript_manager = TranscriptManager()
        self.ocr_manager = OCRManager()
        self.rouge_score_analyzer = RougeScoreAnalyzer()
        self.visualizer = Visualizer()

    def process_videos(self, video_ids, args):
        import pdb

        pdb.set_trace()
        if args.save_video_info:
            print("Saving video info...")
            self.video_info_manager.save_video_info(video_ids, args.output_file)

        if args.download_videos:
            print("Downloading videos...")
            self.video_info_manager.download_videos(video_ids, args.video_dir)

        if args.save_transcripts:
            print("Saving video transcripts...")
            self.transcript_manager.save_transcripts(video_ids, args.output_file)

        if args.perform_ocr:
            print("Performing OCR on videos...")
            self.ocr_manager.perform_ocr(video_ids, args.video_info_file, args.transcript_file, args.output_file, args.video_dir)

        if args.calculate_and_visualize_rouge:
            print("Calculating and visualizing ROUGE scores...")
            self.rouge_score_analyzer.calculate_and_visualize_rouge_scores(
                args.video_info_file, args.transcript_file, args.ocr_results_file, args.output_file, args.result_dir
            )


def main():
    parser = argparse.ArgumentParser(description="YouTube Video Analysis Tool")
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
    parser.add_argument("--output_file", default="output.json", help="Filename for general output results")

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # Update file paths to include result directory
    args.video_info_file = os.path.join(args.result_dir, args.video_info_file)
    args.transcript_file = os.path.join(args.result_dir, args.transcript_file)
    args.ocr_results_file = os.path.join(args.result_dir, args.ocr_results_file)
    args.rouge_scores_file = os.path.join(args.result_dir, args.rouge_scores_file)
    args.output_file = os.path.join(args.result_dir, args.output_file)

    # Load video IDs from CSV file
    if args.video_list:
        video_ids = pd.read_csv(args.video_list)["videoId"].unique()
    else:
        print("Error: Please specify a video list file.")
        return

    analyzer = YouTubeAnalyzer()
    analyzer.process_videos(video_ids, args)

    print("Video analysis complete.")


if __name__ == "__main__":
    main()
