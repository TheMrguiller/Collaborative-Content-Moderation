import subprocess
import zipfile
import os
import requests
import pyzipper
import torch
import numpy as np
import random

def download_kaggle_competition_data(competition_name, download_path='.'):
    """
    Downloads competition data from Kaggle.
    
    Parameters:
    - competition_name: str : The name of the competition on Kaggle.
    - download_path: str : The path to the directory where the data should be downloaded.
    """
    try:
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', competition_name, '-p', download_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Downloaded competition data to {download_path}")
        zip_file_path = os.path.join(download_path, f"{competition_name.split('/')[-1]}.zip")
        
        # Unzip the dataset
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print(f"Unzipped dataset to {download_path}")
        
        # Remove the zip file
        os.remove(zip_file_path)
        print(f"Removed zip file {zip_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading competition data: {e.stderr.decode('utf-8')}")

def download_kaggle_dataset_data(dataset_name, download_path='.'):
    """
    Downloads and unzips a dataset from Kaggle.
    
    Parameters:
    - dataset_name: str : The name of the dataset on Kaggle (e.g., "zillow/zecon").
    - download_path: str : The path to the directory where the data should be downloaded and unzipped.
    """
    try:
        # Download the dataset
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', download_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Downloaded dataset to {download_path}")
        
        # Find the downloaded zip file
        zip_file_path = os.path.join(download_path, f"{dataset_name.split('/')[-1]}.zip")
        
        # Unzip the dataset
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print(f"Unzipped dataset to {download_path}")
        
        # Remove the zip file
        os.remove(zip_file_path)
        print(f"Removed zip file {zip_file_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e.stderr.decode('utf-8')}")
    except zipfile.BadZipFile as e:
        print(f"Error unzipping file: {str(e)}")

        
def does_path_exist(path):
    """
    Check if a path exists.
    
    Parameters:
    - path : str : The path to check.
    
    Returns:
    - bool : True if the path exists, False otherwise.
    """
    return os.path.exists(path)


def get_complete_path_of_file(filename):
    """Join the path of the current directory with the input filename."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, filename)


def read_wordlist(filename: str):
    """Return words from a wordlist file."""
    with open(filename, encoding="utf-8") as wordlist_file:
        for row in iter(wordlist_file):
            row = row.strip()
            if row != "":
                yield row





def any_next_words_form_swear_word(cur_word, words_indices, censor_words):
    """
    Return True, and the end index of the word in the text,
    if any word formed in words_indices is in `CENSOR_WORDSET`.
    """
    # print("cur_word", cur_word)
    full_word = cur_word.lower()
    full_word_with_separators = cur_word.lower()

    # Check both words in the pairs
    for index in iter(range(0, len(words_indices), 2)):
        single_word, end_index = words_indices[index]
        word_with_separators, _ = words_indices[index + 1]
        if single_word == "":
            continue

        full_word = "%s%s" % (full_word, single_word.lower())
        full_word_with_separators = "%s%s" % (
            full_word_with_separators,
            word_with_separators.lower(),
        )
        # print("full_word", full_word)
        # print(censor_words)
        if full_word in censor_words or full_word_with_separators in censor_words:
            try:
                index_= censor_words.index(full_word)
                # print(index_)
                original_word = str(censor_words[index_])
                return True, end_index,original_word
            except ValueError:
                return False, -1,None
    return False, -1, None

def determine_soft_labels_need(experiment_name):
    
    if "soft" in experiment_name:
        return True
    return False

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
