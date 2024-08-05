import json
import os

import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

import utils


class VideoInfoManager:
    def __init__(self):
        pass

    def save_video_info(self, video_ids, output_file):
        results = {}
        for video_id in tqdm(video_ids, desc="Processing video info"):
            try:
                is_short = utils.is_short_video(video_id)
                video_details = self.get_video_details(video_id)
                results[video_id] = {"is_short": is_short, **video_details}
            except Exception as e:
                print(f"Error processing video info for {video_id}: {str(e)}")
                results[video_id] = {"is_short": None, "error": str(e)}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Video info saved to {output_file}")

    def get_video_details(self, video_id):
        try:
            youtube = build("youtube", "v3", developerKey=os.environ["GOOGLE_API_KEY"])
            request = youtube.videos().list(part="snippet,statistics,contentDetails", id=video_id)
            response = request.execute()

            if "items" in response and len(response["items"]) > 0:
                item = response["items"][0]
                view_count = int(item["statistics"].get("viewCount", 0))
                duration = item["contentDetails"]["duration"]
                duration_seconds = utils.parse_duration(duration)
                title = item["snippet"]["title"]
                channel_name = item["snippet"]["channelTitle"]

                return {"view_count": view_count, "duration": duration_seconds, "title": title, "channel_name": channel_name}
            else:
                return None
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred: {e.content}")
            return None

    def download_videos(self, video_ids, video_dir):
        os.makedirs(video_dir, exist_ok=True)
        for video_id in tqdm(video_ids, desc="Downloading videos"):
            try:
                self._download_video(video_id, video_dir)
            except Exception as e:
                print(f"Error downloading video {video_id}: {str(e)}")
        print(f"Videos downloaded to {video_dir}")

    def _download_video(self, video_id, video_dir):
        filename = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(filename):
            ydl_opts = {
                "format": "best",
                "outtmpl": filename,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        return filename

    def load_video_info(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # def update_video_info(self, video_ids, input_file, output_file):
    #     existing_info = self.load_video_info(input_file)
    #     updated_info = existing_info.copy()

    #     for video_id in tqdm(video_ids, desc="Updating video info"):
    #         if video_id not in existing_info or "error" in existing_info[video_id]:
    #             try:
    #                 is_short = utils.is_short_video(video_id)
    #                 video_details = self.get_video_details(video_id)
    #                 updated_info[video_id] = {"is_short": is_short, **video_details}
    #             except Exception as e:
    #                 print(f"Error updating video info for {video_id}: {str(e)}")
    #                 updated_info[video_id] = {"is_short": None, "error": str(e)}

    #     with open(output_file, "w", encoding="utf-8") as f:
    #         json.dump(updated_info, f, ensure_ascii=False, indent=4)

    #     print(f"Updated video info saved to {output_file}")

    # def get_channel_videos(self, channel_id, max_results=50):
    #     try:
    #         request = self.youtube.search().list(part="id,snippet", channelId=channel_id, type="video", order="date", maxResults=max_results)
    #         response = request.execute()

    #         videos = []
    #         for item in response.get("items", []):
    #             video_id = item["id"]["videoId"]
    #             title = item["snippet"]["title"]
    #             videos.append({"id": video_id, "title": title})

    #         return videos
    #     except HttpError as e:
    #         print(f"An HTTP error {e.resp.status} occurred: {e.content}")
    #         return None

    # def get_video_comments(self, video_id, max_results=100):
    #     try:
    #         request = self.youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText", maxResults=max_results)
    #         response = request.execute()

    #         comments = []
    #         for item in response.get("items", []):
    #             comment = item["snippet"]["topLevelComment"]["snippet"]
    #             comments.append(
    #                 {
    #                     "author": comment["authorDisplayName"],
    #                     "text": comment["textDisplay"],
    #                     "likes": comment["likeCount"],
    #                     "published_at": comment["publishedAt"],
    #                 }
    #             )

    #         return comments
    #     except HttpError as e:
    #         print(f"An HTTP error {e.resp.status} occurred: {e.content}")
    #         return None
