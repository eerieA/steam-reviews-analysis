# Steam reviews analysis

Only support downloading reviews for one game, because this is intended to be a module for a slightly larger program.

<!-- TOC -->

- [Steam reviews analysis](#steam-reviews-analysis)
    - [Usage examples](#usage-examples)
        - [Quick start (no arguments)](#quick-start-no-arguments)
        - [With arguments](#with-arguments)
    - [Dependencies](#dependencies)
        - [List of dependencies](#list-of-dependencies)
        - [File sizes](#file-sizes)

<!-- /TOC -->

## Usage examples

### Quick start (no arguments)

```
python reviews_scraper.py <app_id>
```

This will scrape reviews and save them in a json file in the same folder as the script, with default name `steam_reviews_<app_id>.json`.

### With arguments

```
python reviews_scraper.py <app_id> -o subfolder/filename.json
```

This will scrape reviews and save them in a json file in the specified relative path `./subfolder/`, with specified name `filename.json`.

```
python reviews_scraper.py <app_id> -l english -o subfolder/filename.json
```

This will scrape `english` (English) language reviews of the game and save them in a json file in the specified relative path.

## Dependencies

### List of dependencies

See requirements.txt.

### File sizes

The largest one would be PyTorch with CUDA 118. Check `venv/Lib/site-packages/torch/lib` and you will see several large files there. For me the total is about 5 gb.
