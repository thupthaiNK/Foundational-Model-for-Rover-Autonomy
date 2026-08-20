"""
Purpose: Download real Mars rover images from NASA API for CLIP testing
Inputs:  NASA API key (free at https://api.nasa.gov/)
Outputs: ~100 Mars terrain images saved to data/mars_images/
How to run:
    python3 download_mars_images.py --api-key YOUR_KEY
    python3 download_mars_images.py --api-key YOUR_KEY --target 200
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import time
import urllib.request
import json
import argparse

SAVE_DIR = os.path.join(os.path.dirname(__file__), "../data/mars_images")
TARGET = 100

# sol = Martian day; these sols have diverse terrain in Curiosity's traverse
SOLS = [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3200]


def fetch_photo_urls(sol, api_key, camera="NAVCAM"):
    url = (
        f"https://api.nasa.gov/mars-photos/api/v1/rovers/curiosity/photos"
        f"?sol={sol}&camera={camera}&api_key={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        return [p["img_src"] for p in data.get("photos", [])]
    except Exception as e:
        print(f"  [skip sol {sol}] {e}")
        return []


def download_image(img_url, save_path):
    try:
        urllib.request.urlretrieve(img_url, save_path)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="NASA API key from https://api.nasa.gov/")
    parser.add_argument("--target", type=int, default=TARGET, help="Number of images to download")
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving images to: {SAVE_DIR}")
    print(f"Target: {args.target} images\n")
    NASA_API_KEY = args.api_key
    target = args.target

    downloaded = 0
    for sol in SOLS:
        if downloaded >= target:
            break
        print(f"Fetching URLs from sol {sol}...")
        urls = fetch_photo_urls(sol, NASA_API_KEY)
        print(f"  Found {len(urls)} photos")

        for url in urls:
            if downloaded >= target:
                break
            fname = f"curiosity_sol{sol}_{downloaded:04d}.jpg"
            save_path = os.path.join(SAVE_DIR, fname)
            if os.path.exists(save_path):
                downloaded += 1
                continue
            ok = download_image(url, save_path)
            if ok:
                downloaded += 1
                print(f"  [{downloaded:3d}/{TARGET}] {fname}")
            time.sleep(0.3)  # polite rate limit

    print(f"\nDone. {downloaded} images saved to {SAVE_DIR}")
    print("Next: python3 clip_terrain_test.py --folder ../data/mars_images/")


if __name__ == "__main__":
    main()
