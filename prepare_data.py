"""Rebuild the cleaned datasets from the raw scrape, without plots.

Usage (from the repository root):
    python prepare_data.py

Results go to outputs/ so the committed files in data/ are left untouched.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")

from src.steam_data_analyzer import SteamDataAnalyzer
from src.steam_review_cleaner import SteamReviewCleaner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/steam_ml_dataset_new_english.csv")
    parser.add_argument(
        "--games",
        default=None,
        help="Existing cleaned games CSV to take reviews from. Defaults to the one built from --raw.",
    )
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    games_path = args.games

    if games_path is None:
        analyzer = SteamDataAnalyzer(args.raw)
        analyzer.clean_data()
        games_path = os.path.join(args.out, "steam_cleaned_data.csv")
        analyzer.save_cleaned_data(games_path)

    cleaner = SteamReviewCleaner(games_path)
    cleaner.extract_reviews()
    cleaner.clean_all_reviews(min_length=10, max_length=1000)
    cleaner.balance_dataset(method="undersample")
    cleaner.export_cleaned_data(os.path.join(args.out, "steam_reviews_cleaned_final.csv"))


if __name__ == "__main__":
    main()
