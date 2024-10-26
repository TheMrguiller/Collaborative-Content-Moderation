import os
from utils import download_kaggle_competition_data,does_path_exist,download_kaggle_dataset_data
from datasets import load_dataset


if __name__ == '__main__':
    # Create a directory to store the raw data
    directory = 'src/data/raw'  
    directory_jigsaw_unintended_bias = 'src/data/raw/jigsaw-unintended-bias-in-toxicity-classification'
    directory_specialized_rater_pools_dataset = 'src/data/raw/specialized-rater-pools-dataset'
    
    if not does_path_exist(directory):
        os.makedirs(directory)
    if not does_path_exist(directory_jigsaw_unintended_bias):
        os.makedirs(directory_jigsaw_unintended_bias)
        download_kaggle_competition_data(competition_name='jigsaw-unintended-bias-in-toxicity-classification', download_path=directory_jigsaw_unintended_bias)
    if not does_path_exist(directory_specialized_rater_pools_dataset):
        os.makedirs(directory_specialized_rater_pools_dataset)
        download_kaggle_dataset_data(dataset_name='google/jigsaw-specialized-rater-pools-dataset', download_path=directory_specialized_rater_pools_dataset)
