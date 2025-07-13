# Steam reviews analysis

Only support downloading reviews for one game, because this is intended to be a module for a slightly larger program.

## Usage examples

```
python reviews_scraper.py <app_id>
```

This will scrape reviews and save them in a json file in the same folder as the script, with default name `steam_reviews_<app_id>.json`.

```
python reviews_scraper.py <app_id> -o subfolder/filename.json
```

This will scrape reviews and save them in a json file in the specified relative path `./subfolder/`, with specified name `filename.json`.