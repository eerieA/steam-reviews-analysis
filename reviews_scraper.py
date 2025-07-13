from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import requests

STEAM_API_URL = "https://store.steampowered.com/appreviews/"
RATE_LIMIT_PER_MINUTE = 30
COOLDOWN_SECONDS = 60


class SteamReviewScraper:
    """
    A class to scrape all reviews for a given Steam App ID.
    """

    def __init__(self, app_id: int):
        if not isinstance(app_id, int) or app_id <= 0:
            raise ValueError("Invalid Steam App ID provided.")
        self.app_id = app_id
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Steam Review Scraper"})

    def _make_request(
        self, cursor: str = "*", params: Optional[dict] = None
    ) -> Optional[dict]:
        """
        Internal method to make a single request to the Steam API.

        Args:
            cursor (str): The cursor for pagination. '*' for the first page.
            params (dict): A dictionary of request parameters.

        Returns:
            Optional[dict]: JSON response dictionary, or None if the request fails.
        """
        default_params = {
            "json": "1",
            "language": "all",
            "filter": "all",
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": "100",
        }
        if params:
            default_params.update(params)

        default_params["cursor"] = cursor

        try:
            response = self.session.get(
                f"{STEAM_API_URL}{self.app_id}", params=default_params
            )

            if response.status_code == HTTPStatus.BAD_GATEWAY:
                print("Received 502 Bad Gateway. Waiting 10 seconds and retrying...")
                time.sleep(10)
                response = self.session.get(
                    f"{STEAM_API_URL}{self.app_id}", params=default_params
                )

            response.raise_for_status()

            if "application/json" not in response.headers.get("Content-Type", ""):
                print("Unexpected content type. Expected JSON.")
                return None

            return response.json()

        except (requests.RequestException, requests.HTTPError) as e:
            print(f"An error occurred during the request: {e}")
            return None
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from response. Content: {response.text}")
            return None

    def fetch_reviews(self, params_override: Optional[dict] = None) -> list:
        """
        Fetches all reviews for the configured App ID.

        Args:
            params_override (dict, optional): Overrides default request parameters.

        Returns:
            list: A list of review dictionaries.
        """
        all_reviews = []
        cursor = "*"
        query_count = 0
        total_expected_reviews = None

        print(f"Starting review scrape for App ID: {self.app_id}")

        while True:
            if query_count > 0 and query_count % RATE_LIMIT_PER_MINUTE == 0:
                print(
                    f"Rate limit hit ({query_count} requests). Cooling down for {COOLDOWN_SECONDS} seconds..."
                )
                time.sleep(COOLDOWN_SECONDS)

            data = self._make_request(cursor, params=params_override)
            query_count += 1

            if not data or data.get("success") != 1:
                print("API request failed or indicated failure. Stopping.")
                break

            reviews = data.get("reviews", [])
            if not reviews:
                print("No more reviews found. Scrape complete.")
                break

            all_reviews.extend(reviews)
            print(f"  > Fetched {len(reviews)} reviews on this page.")

            if total_expected_reviews is None and "query_summary" in data:
                total_expected_reviews = data["query_summary"].get("total_reviews", 0)
                print(f"Discovered a total of {total_expected_reviews} reviews.")

            print(
                f"  > Downloaded {len(all_reviews)} / {total_expected_reviews or '?'} reviews..."
            )

            next_cursor = data.get("cursor")
            if not next_cursor or next_cursor == cursor:
                print("No new cursor returned. Assuming all reviews have been fetched.")
                break

            cursor = next_cursor

        print(f"\nScraping finished. Total reviews downloaded: {len(all_reviews)}")
        return all_reviews


def save_reviews_to_json(reviews: list, app_id: int, filename: Optional[str] = None):
    """
    Saves the list of reviews to a JSON file.

    Args:
        reviews (list): The list of review dictionaries.
        app_id (int): The Steam App ID.
        filename (str, optional): Output filename. Defaults to steam_reviews_{app_id}.json.
    """
    if not filename:
        filename = f"steam_reviews_{app_id}.json"

    output_path = Path(filename)
    print(f"Saving {len(reviews)} reviews to {output_path.resolve()}...")

    output_data = {
        "app_id": app_id,
        "total_reviews_scraped": len(reviews),
        "reviews": reviews,
    }

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print("Save successful!")
    except IOError as e:
        print(f"Error saving file: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape all Steam reviews for a given App ID.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "app_id", type=int, help="The Steam App ID of the game to scrape."
    )
    parser.add_argument("-o", "--output", type=str, help="Output JSON file path.")
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="all",
        help="Review language (e.g., 'english').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        scraper = SteamReviewScraper(args.app_id)
        api_params = {"language": args.language}
        reviews_data = scraper.fetch_reviews(params_override=api_params)

        if reviews_data:
            save_reviews_to_json(reviews_data, args.app_id, args.output)
        else:
            print("No reviews were downloaded.")
    except ValueError as e:
        print(f"Error: {e}")
