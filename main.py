from visualisation import visualize
import youtube
import reddit
import jsontocsv
import wikipedia as wiki
from dataset_maker import merge_datasets
from nn import train
from bot import start_bot


def run():
    # wiki.create_dataset(1000000000000000)
    # youtube.create_dataset()
    # reddit.create_dataset()
    # jsontocsv.create_dataset()
    merge_datasets()
    #visualize()
    train()
    start_bot()


if __name__ == '__main__':
    run()
