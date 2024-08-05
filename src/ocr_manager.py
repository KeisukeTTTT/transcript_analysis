import os
import time

import cv2
from google.cloud import vision
from google.oauth2 import service_account
from tqdm import tqdm

import utils


class OCRManager:
    def perform_ocr(self, video_ids, video_info_file, transcript_file, output_file, video_dir):
        video_info = utils.load_json(video_info_file)
        transcript_data = utils.load_json(transcript_file)

        results = {}
        for video_id in tqdm(video_ids, desc="Performing OCR"):
            try:
                if video_id in video_info and video_id in transcript_data:
                    is_short = video_info[video_id]["is_short"]
                    transcript = transcript_data[video_id]["transcript"]

                    ocr_results = {}
                    for height in [20, 30, 40]:
                        result = self._process_video_ocr(video_id, transcript, video_dir, min_height=height, is_short=is_short)
                        if result:
                            ocr_results[str(height)] = result

                    results[video_id] = {"ocr_results": ocr_results}
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")

        utils.save_json(results, output_file)
        print(f"OCR results saved to {output_file}")

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
