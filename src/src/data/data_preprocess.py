import pandas as pd
import random
import numpy as np
from pathlib import Path
import os
from src.data.data_cleaner import TextToxicityCleaner
from transformers import BertTokenizerFast,RobertaTokenizerFast,AlbertTokenizerFast,DebertaTokenizerFast
import ast

project_path=os.path.abspath(__file__).split('src')[0] 
data_cleaner=TextToxicityCleaner()
bert_tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")
roberta_tokenizer=RobertaTokenizerFast.from_pretrained("roberta-base")
albert_tokenizer=AlbertTokenizerFast.from_pretrained("albert-base-v2")
deberta_tokenizer=DebertaTokenizerFast.from_pretrained("microsoft/deberta-base")

def generate_random_distribution(num_annotators, prop_toxicity):
    """
    Generate a list with a random distribution of 0s and 1s based on a given proportion.

    Args:
    num_annotators (int): Total number of annotators.
    prop_toxicity (float): Proportion of 1s (toxicity values).

    Returns:
    list: List with the distribution of 0s and 1s.
    """
    num_toxic = int(num_annotators * prop_toxicity)
    distribution = [1] * num_toxic + [0] * (num_annotators - num_toxic)
    random.shuffle(distribution)
    return distribution

"""
In this case, in the annotations there is the possibility to mark if a comment is toxic, not toxic or we are unsure. 
The 0.0 value marsk that possibility,unsure, which is rather interesting. 
In this case all unsure values are going to be represented by a toxic and not toxic values. 
To balanced the unsureness for each 0.0 we will add a 1.0 (toxic) and a 0.0 (not toxic) to the list. 
As in the other datasets i dont have that information, i need to quantify it as if two different viewpoints where given.
"""
def get_toxicity_score(value):
    if value == 0.0:
        return [1,0]
    elif value < 0.0 :
        return 1
    elif value >0.0:
        return 0

def expand_list(value_list):
    new_value_list=[]
    for value in value_list:
        if type(value)==list:
            new_value_list.extend(value)
        else:
            new_value_list.append(value)
    return new_value_list

def merge_jigsaw_unintended_data():
    if os.path.exists(project_path+"/src/data/raw/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv") and os.path.exists(project_path+"src/data/raw/jigsaw-unintended-bias-in-toxicity-classification/toxicity_individual_annotations.csv") and os.path.exists(project_path+"src/data/raw/specialized-rater-pools-dataset/specialized_rater_pools_data.csv"):
        df = pd.read_csv(project_path+"/src/data/raw/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv")
        df_annotations=pd.read_csv(project_path+"src/data/raw/jigsaw-unintended-bias-in-toxicity-classification/toxicity_individual_annotations.csv")
        df.drop(columns=['created_date', 'publication_id',
        'parent_id', 'article_id', 'rating', 'funny', 'wow', 'sad', 'likes',
        'disagree','male', 'female', 'transgender',
        'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
        'other_sexual_orientation', 'christian', 'jewish', 'muslim', 'hindu',
        'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian',
        'latino', 'other_race_or_ethnicity', 'physical_disability',
        'intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
        'other_disability', 'identity_annotator_count','severe_toxicity', 'obscene', 'sexual_explicit',
        'identity_attack', 'insult', 'threat'],inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df_comment_annotation=df_annotations.groupby('id')['toxic'].apply(list).reset_index(name='toxic_values')
        df_final = df.merge(df_comment_annotation, on='id', how='left')
        df_final['toxic_values'] = df_final.apply(lambda x: generate_random_distribution(x["toxicity_annotator_count"],x["toxicity"]) if type(x["toxic_values"])==float else x["toxic_values"], axis=1)
        df_final.dropna(inplace=True,subset=['toxic_values'])
        df_final.reset_index(drop=True, inplace=True)
        del df_annotations
        del df_comment_annotation
        del df
        df_specialized_rater= pd.read_csv(project_path+"src/data/raw/specialized-rater-pools-dataset/specialized_rater_pools_data.csv")
        df_specialized_rater.dropna(inplace=True)
        df_specialized_rater["toxic_score"]= df_specialized_rater["toxic_score"].apply(lambda x: get_toxicity_score(x))
        df_comment_annotation=df_specialized_rater.groupby('id')['toxic_score'].apply(list).reset_index(name='toxic_values')
        df_comment_annotation["toxic_values"]=df_comment_annotation["toxic_values"].apply(lambda x: expand_list(x))
        merged_df = df_final.merge(df_comment_annotation, on='id', how='left')
        merged_df["toxic_values_y"]= merged_df["toxic_values_y"].apply(lambda x: [] if type(x)==float else x)
        merged_df["toxic_values"]= merged_df["toxic_values_x"]+merged_df["toxic_values_y"]
        merged_df["toxicity_annotator_count"]=merged_df["toxic_values"].apply(lambda x: len(x))
        merged_df["toxicity"]=merged_df["toxic_values"].apply(lambda x: np.mean(x))
        merged_df.drop(columns=['toxic_values_x','toxic_values_y'],inplace=True)
        del df_comment_annotation
        del df_specialized_rater
        df_final=merged_df
        if not os.path.exists(project_path+"src/data/processed"):
            os.makedirs(project_path+"src/data/processed")
        df_final.to_csv(project_path+"src/data/processed/jigsaw-unintended-bias-in-toxicity-classification_merged_data.csv",index=False)
        del merged_df
        return df_final
    else:
        raise FileNotFoundError("Files not found")

def jigsaw_unintended_data_split(df):

    df_test = df[df["split"] == "test"]
    df_train = df[df["split"] == "train"]
    df_test_additional_data=df_train[(df_train["toxicity_annotator_count"]>=15) & (df_train["toxicity_annotator_count"]<=49)].sample(frac=df_test.shape[0]/df_train.shape[0], random_state=42)
    df_train = df_train.drop(df_test_additional_data.index)
    df_train.reset_index(drop=True, inplace=True)
    df_test = pd.concat([df_test, df_test_additional_data], axis=0)
    df_test.reset_index(drop=True, inplace=True)
    grouped = df_train.groupby(['toxicity', 'toxicity_annotator_count'])
    # List to store sampled DataFrames
    sampled_dfs = []

    # Percentage of validation data to sample
    validation_frac = 0.1  # Adjust as needed

    # Sampling from each group
    for _, group_df in grouped:
        # Calculate number of samples to take from this group based on fraction
        n_samples = int(len(group_df) * validation_frac)
        
        # Sampling from the group
        sampled_group = group_df.sample(n=n_samples, random_state=42)  # Use a fixed random state for reproducibility
        
        # Adding sampled group to the list
        sampled_dfs.append(sampled_group)

    # Concatenating sampled groups into a validation set
    df_valid = pd.concat(sampled_dfs)
    df_train = df_train.drop(df_valid.index)
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    return df_train, df_valid, df_test

def clean_data(df,text_column="comment_text",sleeptime=8):
    return data_cleaner.clean(df,text_column,sleeptime)

def tokenize_text(text, tokenizer):
    '''
    Function to tokenize the given text.
    
    Parameter:
    ---------
    text: str
        Text which has to be tokenized.
    tokenizer: BertTokenizerFast
        Tokenizer to be used for tokenization.
    '''
    # Tokenize the text
    tokenized_text = tokenizer.tokenize(text)
    return len(tokenized_text)

def tokenize_inputs(text):
    '''
    Function to tokenize the given text using different tokenizers.
    
    Parameter:
    ---------
    text: str
        Text which has to be tokenized.
    '''
    bert_token_count = tokenize_text(text, bert_tokenizer)
    roberta_token_count = tokenize_text(text, roberta_tokenizer)
    albert_token_count = tokenize_text(text, albert_tokenizer)
    deberta_token_count = tokenize_text(text, deberta_tokenizer)
    
    return (bert_token_count, roberta_token_count, albert_token_count, deberta_token_count)

if __name__ == '__main__':
    
    df_final=merge_jigsaw_unintended_data()
    df_train,df_valid,df_test = jigsaw_unintended_data_split(df_final)
    df_train = clean_data(df_train,)
    df_valid = clean_data(df_valid)
    df_test = clean_data(df_test)
    df_train.to_csv(project_path+"src/data/processed/jigsaw-unintended-bias-in-toxicity-classification_train_data_clean.csv",index=False)
    df_valid.to_csv(project_path+"src/data/processed/jigsaw-unintended-bias-in-toxicity-classification_valid_data_clean.csv",index=False)
    df_test.to_csv(project_path+"src/data/processed/jigsaw-unintended-bias-in-toxicity-classification_test_data_clean.csv",index=False)

