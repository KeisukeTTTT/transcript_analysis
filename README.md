# YouTube字幕分析ツール

## Setup

1. Clone & Install:

```bash
git clone https://github.com/KeisukeTTTT/transcript_analysis.git
cd transcript_analysis
pip install -r requirements.txt
```

2. 環境変数の設定:

Google API キー, Google service account file を環境変数に設定

```bash
export GOOGLE_API_KEY="your_api_key"
export GOOGLE_SERVICE_ACCOUNT_FILE="path/to/file"
```

## 実行方法
1. `videos.csv`に分析したいYouTube動画の動画ID一覧を記載
2. 個別処理
- YouTube 動画の情報取得: `make video_info`
- Youtube動画のダウンロード: `make download_videos`
- トランスクリプト取得: `make transcripts`
- OCR 処理: `make ocr`
- ROUGE スコアの計算と可視化: `make rouge`
