PYTHON = python
SCRIPT = main.py
VIDEO_LIST = videos.csv
VIDEO_DIR = youtube-videos
RESULT_DIR = results
JSON_DIR = $(RESULT_DIR)/json
IMG_DIR = $(RESULT_DIR)/img
VIDEO_INFO_FILE = $(JSON_DIR)/video_info.json
TRANSCRIPT_FILE = $(JSON_DIR)/transcripts.json
OCR_RESULTS_FILE = $(JSON_DIR)/ocr_results.json
ROUGE_SCORES_FILE = $(JSON_DIR)/rouge_scores.json
OUTPUT_FILE = $(JSON_DIR)/output.json

all: video_info transcripts ocr rouge

$(RESULT_DIR) $(JSON_DIR) $(IMG_DIR):
	mkdir -p $@

video_info: $(JSON_DIR)
	$(PYTHON) $(SCRIPT) --video_list $(VIDEO_LIST) --result_dir $(RESULT_DIR) --video_info_file $(VIDEO_INFO_FILE) --save_video_info

transcripts: $(JSON_DIR) video_info
	$(PYTHON) $(SCRIPT) --video_list $(VIDEO_LIST) --result_dir $(RESULT_DIR) --video_info_file $(VIDEO_INFO_FILE) --transcript_file $(TRANSCRIPT_FILE) --save_transcripts

ocr: $(JSON_DIR) video_info transcripts
	$(PYTHON) $(SCRIPT) --video_list $(VIDEO_LIST) --video_dir $(VIDEO_DIR) --result_dir $(RESULT_DIR) --video_info_file $(VIDEO_INFO_FILE) --transcript_file $(TRANSCRIPT_FILE) --ocr_results_file $(OCR_RESULTS_FILE) --perform_ocr

rouge: $(JSON_DIR) $(IMG_DIR) 
	$(PYTHON) $(SCRIPT) --video_list $(VIDEO_LIST) --result_dir $(RESULT_DIR) --video_info_file $(VIDEO_INFO_FILE) --transcript_file $(TRANSCRIPT_FILE) --ocr_results_file $(OCR_RESULTS_FILE) --rouge_scores_file $(ROUGE_SCORES_FILE) --img_dir $(IMG_DIR) --calculate_and_visualize_rouge

download_videos: $(JSON_DIR) video_info
	$(PYTHON) $(SCRIPT) --video_list $(VIDEO_LIST) --video_dir $(VIDEO_DIR) --result_dir $(RESULT_DIR) --video_info_file $(VIDEO_INFO_FILE) --download_videos

clean:
	rm -rf $(RESULT_DIR)

.PHONY: all video_info transcripts ocr rouge download_videos clean
