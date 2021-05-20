# Clickbait Detector

This is a complete set of utilities to download data, visualize it, train a neural network model and then use it with Discord bot, which uses the API server.


### Setup
```
python3 pip install requirements.txt
```

### Directory setup
You need to create directories, which will hold datasets, models and all the other data. In my case:
```
datasets/
raw_datasets/
raw_json_datasets/
wandb/
youtube/
images/
plots/
models/
packs/
tables/
```

## Data sources

If you have any CSV datasets that you want to use for the purpose of training the neural network model you will need to put them into `/datasets` and there are some requirements:
- Your files must have a `title` column
- To specify if titles from given dataset should be classified as clickbait or nonclickbait prefix their names with `clickbait_` or `nonclickbait_` respectively.
- Mixed datasets must contain an additional column `clickbait` which holds value 0 for a nonclickbait title and 1 for clickbait title. Prefix those with `mix_`

If you have any JSON datasets:
- Your files must have a `title` field
- To specify if titles from given dataset should be classified as clickbait or nonclickbait prefix their names with `clickbait_` or `nonclickbait_` respectively.
- Support for mixed datasets isn't implemented.


Clickbait Detector is capable of downloading data from:

#### YouTube

To set it up create text files with a list of links to clickbait and nonclickbait YouTube channels (separated by new line).
```
youtube/yt_clickbait_channels.txt
youtube/yt_nonclickbait_channels.txt
```
You will also need to create `youtube/yt_api.txt` file and paste your API key in there.

To download video titles from specified channels run `create_dataset()` function from `youtube.py` file. This will create CSV datasets (later used for training the neural network) and two files responsible for tracking which channels have already been checked, kinda a finicky solution, you have to clear those files to re-download video titles:
```
youtube/checked_nonclickbait_channels.txt
youtube/checked_clickbait_channels.txt
```

#### Wikipedia

Running a `create_dataset(100000000000000000000)` from `wikipedia.py` will download as many news titles as possible from wikinews, and automatically consider them nonclickbait by putting them into `raw_datasets/nonclickbait_wikinews_titles.csv`.

#### Reddit

A small amount of titles in given subreddits can be downloaded by running `create_dataset()` function from `reddit.py`. List of subreddits to check is hardcoded inside that file.



All datasets created by fetching data from those three sources will be put in `/raw_datasets` directory.



## Merging datasets

`dataset_maker.py` is responsible for merging your dataset into one big training dataset. List of datasets that should be included is hardcoded inside that file.
The resulting dataset is `datasets/mixed_dataset.csv`


## Visualization

Once you merged your datasets you can visualize certain features of it. `visualization.py` will read `datasets/mixed_dataset.csv` file and output multiple images of the resulting analysis. Examples:


| Most common capital letters in clickbait titles | Most common words in nonclickbait titles |
| ------------------- | ---------------|
|<img src="https://i.imgur.com/sfLOZAh.png" width=250px height=250px> | <img src="https://i.imgur.com/01Z4qj0.png" width=250px height=250px> |




| Number of capital letters | Number of words | 
| ------------------- | ---------------|
| <img src="https://i.imgur.com/iXeTBEv.png" width=250px height=250px> | <img src="https://i.imgur.com/jGcSG94.png" width=250px height=250px> |


### Wandb integration

Log in to wandb before you start training to log your progress.


### Run

In order to start a full process of downloading the data, visualizing it and then training the network run `main.py` file.


## Results

With training data I used the neural network achieved 95% accuracy.

### Example usage (Discord integration)

![](https://i.imgur.com/uyyK9jf.png)

![](https://i.imgur.com/CbVcOiq.png)

![](https://i.imgur.com/goWVDjQ.png)


