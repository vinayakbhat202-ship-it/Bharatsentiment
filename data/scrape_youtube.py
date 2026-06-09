# data/scrape_youtube.py
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ── Brands + their YouTube video URLs ──────────────────────
# Add any Indian brand videos you want to track
BRAND_VIDEOS = {
    "Jio": [
        "https://www.youtube.com/watch?v=xWYROuB-pRs",
    ],
    "Zomato": [
        "https://www.youtube.com/watch?v=dummy1",  # replace with real URLs
    ],
    "CRED": [
        "https://www.youtube.com/watch?v=dummy2",
    ],
}

MAX_COMMENTS_PER_VIDEO = 200

def scrape_video(url, brand):
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        generator = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)
        for i, comment in enumerate(generator):
            if i >= MAX_COMMENTS_PER_VIDEO:
                break
            comments.append({
                "brand":      brand,
                "source":     "youtube",
                "text":       comment["text"],
                "likes":      comment.get("votes", 0),
                "scraped_at": datetime.utcnow().isoformat(),
                "video_url":  url
            })
        print(f"  ✅ {len(comments)} comments from {url}")
    except Exception as e:
        print(f"  ❌ Failed {url}: {e}")
    return comments

def main():
    all_comments = []
    for brand, urls in BRAND_VIDEOS.items():
        print(f"\n🔍 Scraping: {brand}")
        for url in urls:
            all_comments.extend(scrape_video(url, brand))

    df = pd.DataFrame(all_comments)
    out = "data/raw/youtube_comments.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Saved {len(df)} total comments → {out}")
    print(df["brand"].value_counts())

if __name__ == "__main__":
    main()