# Steam reviews analysis

Only support processing reviews for **one game**, because this is mostly for personal use.

⚠️Temporarily only supports reviews in English.

<!-- TOC -->

- [Steam reviews analysis](#steam-reviews-analysis)
    - [Usage examples](#usage-examples)
        - [Quick start](#quick-start)
    - [Dependencies](#dependencies)
        - [List of dependencies](#list-of-dependencies)
        - [Package sizes](#package-sizes)

<!-- /TOC -->

## Usage examples

### Quick start

Do the following 3 steps sequentially.

```bash
python reviews_scraper.py <app_id> -l english -o subfolder/filename.json
```

This will scrape `english` (English) language reviews of the game with id `<app_id>` and save them in a json file in the specified relative path.

```bash
python analyze_sentiments.py --filename subfolder/filename.json
```

This will extract frequencies of sentiment labels from all the reviews in the JSON file you just saved, then plot a graph saved as `emotion_distribution_top5.png`

> FYI it uses an emotion classification transformer model to extract the labels, and only collects top 5 labels from each review be default.

```bash
python create_word_cloud.py --filename subfolder/filename.json
```

This will produce a word cloud graph from all the reviews in the JSON file you just saved, saved as `wordcloud_reviews.png`.

## Dependencies

### List of dependencies

See requirements.txt.

Also a transformer model's files put in a child folder under `./models/`. This program only can use the ones with typical files, like this:

```
./models/some_emotion_classifier/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── vocab.txt (or vocab.json, or none)
```

### Package sizes

The largest one would be PyTorch with CUDA 118. Check `venv/Lib/site-packages/torch/lib` and you will see several large files there. For me the total was about 5 gb.

The second largest probably will be the transformer model, which can be from ~300 mb to ~1.2 gb or more.
