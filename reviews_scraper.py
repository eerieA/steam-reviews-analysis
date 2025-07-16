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
            "filter": "recent",  # "recent" is usually better than "all"
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

    def fetch_reviews_with_filter(
        self, filter_type: str, params_override: Optional[dict] = None
    ) -> list:
        """
        Fetches reviews using a specific filter type.

        Args:
            filter_type (str): The filter type to use ("recent", "updated", "all")
            params_override (dict, optional): Overrides default request parameters.

        Returns:
            list: A list of review dictionaries.
        """
        all_reviews = []
        seen_ids = set()
        cursor = "*"
        query_count = 0
        seen_cursors = set()
        pages_without_new_reviews = 0
        max_pages_without_new = 3

        # Set up parameters with the specified filter
        default_params = {
            "json": "1",
            "language": "all",
            "filter": filter_type,
            "review_type": "all",
            "purchase_type": "all",
            "num_per_page": "100",
        }
        if params_override:
            default_params.update(params_override)

        print(f"Fetching reviews with filter: {filter_type}")

        while True:
            if cursor in seen_cursors:
                print(
                    f"Cursor already used with filter {filter_type}. Stopping pagination."
                )
                break
            seen_cursors.add(cursor)

            if query_count > 0 and query_count % RATE_LIMIT_PER_MINUTE == 0:
                print(
                    f"Rate limit hit ({query_count} requests). Cooling down for {COOLDOWN_SECONDS} seconds..."
                )
                time.sleep(COOLDOWN_SECONDS)

            # Override the cursor in params
            current_params = default_params.copy()
            current_params["cursor"] = cursor

            data = self._make_request(cursor, params=current_params)
            query_count += 1

            if not data or data.get("success") != 1:
                print(f"API request failed for filter {filter_type}. Stopping.")
                break

            reviews = data.get("reviews", [])
            if not reviews:
                print(f"No more reviews found for filter {filter_type}.")
                break

            # Process reviews and only count new ones based on seen rec ids
            new_reviews = []
            for review in reviews:
                rec_id = review.get("recommendationid")
                if rec_id not in seen_ids:
                    seen_ids.add(rec_id)
                    new_reviews.append(review)

            all_reviews.extend(new_reviews)

            print(
                f"  > Filter {filter_type}: {len(new_reviews)} new reviews from {len(reviews)} total"
            )

            # Check if we're getting no new reviews
            if len(new_reviews) == 0:
                pages_without_new_reviews += 1
                if pages_without_new_reviews >= max_pages_without_new:
                    print(
                        f"  > No new reviews for {pages_without_new_reviews} pages with filter {filter_type}. Stopping."
                    )
                    break
            else:
                pages_without_new_reviews = 0

            # Get next cursor
            next_cursor = data.get("cursor")
            if not next_cursor or next_cursor in seen_cursors:
                print(f"No valid next cursor for filter {filter_type}. Stopping.")
                break

            cursor = next_cursor

        return all_reviews

    def fetch_reviews(self, params_override: Optional[dict] = None) -> list:
        """
        Fetches all reviews using multiple filter strategies.

        Args:
            params_override (dict, optional): Overrides default request parameters.

        Returns:
            list: A list of review dictionaries.
        """
        print(f"Starting comprehensive review scrape for App ID: {self.app_id}")

        all_reviews = []
        seen_ids = set()

        # Try different filter types
        filters_to_try = ["recent", "updated"]

        for filter_type in filters_to_try:
            print(f"\n--- Trying filter: {filter_type} ---")

            filter_reviews = self.fetch_reviews_with_filter(
                filter_type, params_override
            )

            # Add only unique reviews
            new_unique_reviews = []
            for review in filter_reviews:
                rec_id = review.get("recommendationid")
                if rec_id not in seen_ids:
                    seen_ids.add(rec_id)
                    new_unique_reviews.append(review)

            all_reviews.extend(new_unique_reviews)
            print(
                f"Added {len(new_unique_reviews)} unique reviews from filter {filter_type}"
            )
            print(f"Total unique reviews so far: {len(all_reviews)}")

        # If we still haven't gotten many reviews, try the "all" filter as a last resort
        if len(all_reviews) < 10000:  # Arbitrary threshold
            print("\n--- Trying filter: all (last resort) ---")
            all_filter_reviews = self.fetch_reviews_with_filter("all", params_override)

            new_unique_reviews = []
            for review in all_filter_reviews:
                rec_id = review.get("recommendationid")
                if rec_id not in seen_ids:
                    seen_ids.add(rec_id)
                    new_unique_reviews.append(review)

            all_reviews.extend(new_unique_reviews)
            print(f"Added {len(new_unique_reviews)} unique reviews from filter 'all'")
            print(f"Total unique reviews so far: {len(all_reviews)}")

        print(
            f"\nScraping finished. Total unique reviews downloaded: {len(all_reviews)}"
        )
        return all_reviews

    def fetch_reviews_allow_duplicates(
        self, params_override: Optional[dict] = None
    ) -> list:
        """
        Alternative method: Fetch reviews allowing duplicates for later cleanup.
        This method is more aggressive and will make more API calls.

        Args:
            params_override (dict, optional): Overrides default request parameters.

        Returns:
            list: A list of review dictionaries (may contain duplicates).
        """
        print(
            f"Starting aggressive review scrape (allowing duplicates) for App ID: {self.app_id}"
        )

        all_reviews = []
        filters_to_try = ["recent", "updated", "all"]

        for filter_type in filters_to_try:
            print(f"\n--- Fetching with filter: {filter_type} ---")

            cursor = "*"
            query_count = 0
            seen_cursors = set()
            consecutive_empty_pages = 0
            max_consecutive_empty = 5

            while True:
                if cursor in seen_cursors:
                    print(
                        f"Cursor already used with filter {filter_type}. Moving to next filter."
                    )
                    break
                seen_cursors.add(cursor)

                if query_count > 0 and query_count % RATE_LIMIT_PER_MINUTE == 0:
                    print(f"Rate limit hit ({query_count} requests). Cooling down...")
                    time.sleep(COOLDOWN_SECONDS)

                # Set up parameters
                current_params = {
                    "json": "1",
                    "language": "all",
                    "filter": filter_type,
                    "review_type": "all",
                    "purchase_type": "all",
                    "num_per_page": "100",
                    "cursor": cursor,
                }
                if params_override:
                    current_params.update(params_override)

                data = self._make_request(cursor, params=current_params)
                query_count += 1

                if not data or data.get("success") != 1:
                    print(
                        f"API request failed for filter {filter_type}. Moving to next filter."
                    )
                    break

                reviews = data.get("reviews", [])
                if not reviews:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= max_consecutive_empty:
                        print(
                            f"No reviews for {consecutive_empty_pages} consecutive pages. Moving to next filter."
                        )
                        break
                    continue
                else:
                    consecutive_empty_pages = 0

                all_reviews.extend(reviews)
                print(
                    f"  > Filter {filter_type}: Fetched {len(reviews)} reviews (total so far: {len(all_reviews)})"
                )

                # Get next cursor
                next_cursor = data.get("cursor")
                if not next_cursor or next_cursor in seen_cursors:
                    print(
                        f"No valid next cursor for filter {filter_type}. Moving to next filter."
                    )
                    break

                cursor = next_cursor

        print(
            f"\nAggressive scraping finished. Total reviews downloaded: {len(all_reviews)} (may contain duplicates)"
        )
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


def deduplicate_reviews(reviews: list) -> list:
    """
    Remove duplicate reviews based on recommendation ID.

    Args:
        reviews (list): List of review dictionaries

    Returns:
        list: Deduplicated list of reviews
    """
    seen_ids = set()
    unique_reviews = []

    for review in reviews:
        rec_id = review.get("recommendationid")
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            unique_reviews.append(review)

    print(f"Deduplication: {len(reviews)} -> {len(unique_reviews)} reviews")
    return unique_reviews


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
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Use aggressive scraping (allows duplicates, more API calls)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        scraper = SteamReviewScraper(args.app_id)
        api_params = {"language": args.language}

        if args.aggressive:
            print("Using aggressive scraping method...")
            reviews_data = scraper.fetch_reviews_allow_duplicates(
                params_override=api_params
            )
            
            reviews_data = deduplicate_reviews(reviews_data)
        else:
            print("Using standard scraping method...")
            reviews_data = scraper.fetch_reviews(params_override=api_params)

        if reviews_data:
            save_reviews_to_json(reviews_data, args.app_id, args.output)
        else:
            print("No reviews were downloaded.")
    except ValueError as e:
        print(f"Error: {e}")
