import json

from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

import utils


class TranscriptManager:
    def __init__(self):
        self.languages = ["ja"]

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

    def get_transcript(self, video_id):
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

            for lang in self.languages:
                try:
                    transcript = transcripts.find_transcript([lang])
                    return self._process_transcript(transcript.fetch(), utils.is_short_video(video_id))
                except NoTranscriptFound:
                    continue

            transcript = transcripts.find_transcript([])
            return self._process_transcript(transcript.fetch(), utils.is_short_video(video_id))

        except TranscriptsDisabled:
            print(f"Transcripts are disabled for video {video_id}")
            return None, None
        except Exception as e:
            print(f"Error getting transcript for video {video_id}: {str(e)}")
            return None, None

    def _process_transcript(self, transcript, is_short):
        max_duration = 60 if is_short else 300  # 60秒 or 5分

        filtered_transcripts = [tr for tr in transcript if tr["start"] <= max_duration]
        transcript_text = " ".join([tr["text"] for tr in filtered_transcripts])
        return transcript_text, filtered_transcripts

    def load_transcripts(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_transcripts(self, video_ids, input_file, output_file):
        existing_transcripts = self.load_transcripts(input_file)
        updated_transcripts = existing_transcripts.copy()

        for video_id in tqdm(video_ids, desc="Updating transcripts"):
            if video_id not in existing_transcripts or "error" in existing_transcripts[video_id]:
                try:
                    transcript, _ = self.get_transcript(video_id)
                    if transcript:
                        updated_transcripts[video_id] = {"transcript": transcript}
                    else:
                        updated_transcripts[video_id] = {"transcript": None, "error": "No transcript available"}
                except Exception as e:
                    print(f"Error updating transcript for video {video_id}: {str(e)}")
                    updated_transcripts[video_id] = {"transcript": None, "error": str(e)}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(updated_transcripts, f, ensure_ascii=False, indent=4)

        print(f"Updated transcripts saved to {output_file}")

    def get_transcript_statistics(self, transcripts):
        stats = {
            "total_videos": len(transcripts),
            "videos_with_transcript": 0,
            "videos_without_transcript": 0,
            "average_transcript_length": 0,
            "longest_transcript": 0,
            "shortest_transcript": float("inf"),
        }

        total_length = 0
        for video_id, data in transcripts.items():
            if data.get("transcript"):
                stats["videos_with_transcript"] += 1
                transcript_length = len(data["transcript"].split())
                total_length += transcript_length
                stats["longest_transcript"] = max(stats["longest_transcript"], transcript_length)
                stats["shortest_transcript"] = min(stats["shortest_transcript"], transcript_length)
            else:
                stats["videos_without_transcript"] += 1

        if stats["videos_with_transcript"] > 0:
            stats["average_transcript_length"] = total_length / stats["videos_with_transcript"]

        stats["shortest_transcript"] = stats["shortest_transcript"] if stats["shortest_transcript"] != float("inf") else 0

        return stats

    def print_transcript_statistics(self, stats):
        print("\nTranscript Statistics:")
        print(f"Total videos: {stats['total_videos']}")
        print(f"Videos with transcript: {stats['videos_with_transcript']}")
        print(f"Videos without transcript: {stats['videos_without_transcript']}")
        print(f"Average transcript length: {stats['average_transcript_length']:.2f} words")
        print(f"Longest transcript: {stats['longest_transcript']} words")
        print(f"Shortest transcript: {stats['shortest_transcript']} words")
