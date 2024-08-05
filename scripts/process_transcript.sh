#!/bin/bash

VIDEO_LIST="./csv/testid.csv"
VIDEO_DIR="youtube-videos"
RESULT_DIR="results"
VIDEO_INFO_FILE="video_info.json"
TRANSCRIPT_FILE="transcripts.json"
OCR_RESULTS_FILE="ocr_results.json"
ROUGE_SCORES_FILE="rouge_scores.json"
OUTPUT_FILE="output.json"
# ACTIONS="--save_video_info --download_videos --save_transcripts --perform_ocr --calculate_and_visualize_rouge"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video_list) VIDEO_LIST="$2"; shift ;;
        --video_dir) VIDEO_DIR="$2"; shift ;;
        --result_dir) RESULT_DIR="$2"; shift ;;
        --video_info_file) VIDEO_INFO_FILE="$2"; shift ;;
        --transcript_file) TRANSCRIPT_FILE="$2"; shift ;;
        --ocr_results_file) OCR_RESULTS_FILE="$2"; shift ;;
        --rouge_scores_file) ROUGE_SCORES_FILE="$2"; shift ;;
        --output_file) ROUGE_SCORES_FILE="$2"; shift ;;
        --save_video_info|--download_videos|--save_transcripts|--perform_ocr|--calculate_and_visualize_rouge)
            ACTIONS="$ACTIONS $1" ;;
        *) echo "Unexpected argument: $1" >&2; exit 1 ;;
    esac
    shift
done

echo "Starting YouTube video analysis"
echo "================================"
echo "Video List: $VIDEO_LIST"
echo "Video Directory: $VIDEO_DIR"
echo "Result Directory: $RESULT_DIR"
echo "Video Info File: $VIDEO_INFO_FILE"
echo "Transcript File: $TRANSCRIPT_FILE"
echo "OCR Results File: $OCR_RESULTS_FILE"
echo "ROUGE Scores File: $ROUGE_SCORES_FILE"
echo "Output File: $OUTPUT_FILE"
echo "Actions to perform: $ACTIONS"
echo "================================"

if [ -z "$VIDEO_LIST" ]; then
    echo "Error: Video list is required."
    exit 1
fi

COMMAND="python main.py \
    --video_list "$VIDEO_LIST" \
    --video_dir "$VIDEO_DIR" \
    --result_dir "$RESULT_DIR" \
    --video_info_file "$VIDEO_INFO_FILE" \
    --transcript_file "$TRANSCRIPT_FILE" \
    --ocr_results_file "$OCR_RESULTS_FILE" \
    --rouge_scores_file "$ROUGE_SCORES_FILE" \
    --output_file "$OUTPUT_FILE" \
    $ACTIONS"

echo "Executing command: $COMMAND"
eval $COMMAND

echo "All specified actions completed."