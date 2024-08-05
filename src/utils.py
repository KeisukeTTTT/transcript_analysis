# utils.py

import os
import json
import requests
from datetime import timedelta
import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def is_short_video(video_id):
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

def parse_duration(duration):
    import re
    regex = re.compile(r'P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?')
    match = regex.match(duration)
    if not match:
        return 0
    parts = match.groupdict()
    time_params = {}
    for name, param in parts.items():
        if param:
            time_params[name] = int(param)
    return int(timedelta(**time_params).total_seconds())

def setup_matplotlib():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]