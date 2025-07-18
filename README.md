# Steam reviews analysis

Only support processing reviews for **one game**, because this is mostly for personal use. Needs a GPU with CUDA capability.

⚠️Temporarily only supports reviews in English and SChinese.

<!-- TOC -->

- [Steam reviews analysis](#steam-reviews-analysis)
    - [Usage examples](#usage-examples)
        - [Quick start](#quick-start)
        - [With customized constants](#with-customized-constants)
    - [Dependencies](#dependencies)
        - [List of dependencies](#list-of-dependencies)
        - [Package sizes](#package-sizes)

<!-- /TOC -->

## Usage examples

### Quick start

🚨First ensure the dependencies mentioned in [List of dependencies](#list-of-dependencies). Then if you can run the `tools\test_trfm_gpu.py` script without errors, it is probably good to go.

After that, do the following 3 steps sequentially to get results.

1.
    ```bash
    python reviews_scraper.py <app_id> -l english -o subfolder/filename.json
    ```

    This will scrape `english` (English) language reviews of the game with id `<app_id>` and save them in a json file in the specified relative path.

2.
    ```bash
    python analyze_sentiments.py --filename subfolder/filename.json -l english --appid <app_id>
    ```

    This will extract frequencies of sentiment labels from all the `english` reviews in the JSON file you just saved, then plot a graph saved as `<app_id>_emo_distrib_english.png`, which will be under `./output` by default.

    > FYI it uses an emotion classification transformer model to extract the labels, and only collects top 5 labels from each review be default.

3.
    ```bash
    python create_word_cloud.py --filename subfolder/filename.json -l english --appid <app_id>
    ```

    This will produce a word cloud graph from all the `english` in the JSON file you just saved, and save the graph as `<app_id>_wordcloud_english.png`, which will be under `./output` by default..

### With customized constants

There are a few constants used in the scripts. They are stored in `config.json`.

| Constant name | Description |
|-----------|-------------|
| `model_subfolder_eng` | The subfolder name for the sentiment classification transformer model files for English |
| `model_subfolder_sch` | The subfolder name for the sentiment classification transformer model files for SChinese |
| `batch_size` | The batch size used when running sentiment analysis. The higher it is, the more VRAM the computation (inference) requires |
| `top_k_labels` | The number of top labels to count for each review in sentiment analysis |
| `output_dir` | The subfolder where output images will be saved |

Customize these according to your needs. For example increase batch size from `8` to `32` if you have good VRAM, and run the sentiment analysis scripts again.

## Dependencies

### List of dependencies

**Python and packages**

Python `3.13.1`. See requirements.txt for packages. Note that torch+cu118 needs a GPU with CUDA.

**Transformer model**

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

The recommended and default models are

- `cirimus-modernbert-base-go-emo` (corresponds to [cirimus/modernbert-base-go-emotions](https://huggingface.co/cirimus/modernbert-base-go-emotions))
- `schuylerh-bert-multi-go-emo` (corresponds to [SchuylerH/bert-multilingual-go-emtions](https://huggingface.co/SchuylerH/bert-multilingual-go-emtions))

so you need to download their files manually and put them under `./models/`, with exactly the subfolder names above, for example `./models/schuylerh-bert-multi-go-emo`.

> You can change these subfolder names and tell the script about them through ./config.json, even if you use the default models.

But as described above, other models can be used, and if that is the case, please also change the values in `config.json` to match your model file subfolder names.

### Package sizes

The largest one would be PyTorch with CUDA 118. I checked my `./venv/Lib/site-packages/torch/lib` and saw several large files there. For me the total was about 5 gb.

The second largest probably will be the transformer models, each of which can be from ~300 mb to ~1.2 gb or more. The default ones are about 600 mb.
