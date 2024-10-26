
from typing import Any, Dict, Optional, Union
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
import pandas as pd
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from src.utils.compute_disagreement import get_variance_labels,get_distance_labels,get_entropy_labels
import os
from torch.utils.data import SequentialSampler,DataLoader, BatchSampler
import ast
from memory_profiler import profile
from src.data.utils import seed_everything,seed_worker
import random
import numpy as np


seed=42
seed_everything(seed)

class JigsawUnintendedDataset(Dataset):
    def __init__(self,
                tokenizer: PreTrainedTokenizerBase,
                file_path: Optional[str] = None,
                batch_size: int = 32,
                padding: Union[str, bool] = "max_length",
                truncation: str = "only_first",
                max_length: int = 128,
                task:str="classification",
                disagreement:str="distance",
                data: Optional[pd.DataFrame] = None,
                preprocessed_data_path:Optional[str]=None):
        """
        This class is used to load the Jigsaw Unintended Bias in Toxicity Classification dataset.
            file_path: str: Path to the dataset file.
            batch_size: int: Batch size for the DataLoader.
            num_workers: int: Number of workers for the DataLoader.
            padding: str: Padding method for the tokenizer.
            truncation: str: Truncation method for the tokenizer.
            max_length: int: Maximum length for the tokenizer.
            task: str: Task to perform. Either classification or multitask.
            disagreement: str: Disagreement method for the multitask. It can be variance, distance or entropy.
            tokenizer: PreTrainedTokenizerBase: Tokenizer to use.
            data: Optional[pd.DataFrame]: Data to use.
            
            preprocessed_data_path: Optional[str]: Path to save the preprocessed data.
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.batch_size = batch_size
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.task = task
        self.disagreement = disagreement
        
        self.preprocessed_data_path=preprocessed_data_path
        self.set_seed(seed)
        if preprocessed_data_path and os.path.exists(preprocessed_data_path):
            # Load preprocessed data if available
            dataset = torch.load(preprocessed_data_path)
            self.toxicity_scores = [data["toxicity"] for data in dataset]
            self.toxic_values = [data["toxic_values"] for data in dataset]
            self.input_ids = [data["input_ids"] for data in dataset]
            self.attention_mask = [data["attention_mask"] for data in dataset]
            self.labels = [data["labels"] for data in dataset]
            self.disagreement_weights = self.calculate_disagreement_weights(self.toxic_values)
            self.toxicity_scores_weights = self.calculate_toxicity_score_weights(self.toxicity_scores)
            self.update_labels(task=task,disagreement=disagreement)
        else:
            if data is None:
                try:
                    data = pd.read_csv(self.file_path)
                except FileNotFoundError:
                    raise FileNotFoundError("File not found")

            dataset = data[["toxicity", "toxic_values", "comment_text"]]
            self.toxicity_scores = dataset["toxicity"].tolist()
            dataset["toxic_values"] = dataset["toxic_values"].apply(ast.literal_eval)
            self.toxic_values = dataset["toxic_values"].tolist()
            comment_texts = dataset["comment_text"].tolist()
            del dataset
            self.disagreement_weights = self.calculate_disagreement_weights(self.toxic_values)
            self.toxicity_scores_weights = self.calculate_toxicity_score_weights(self.toxicity_scores)
            process_data = self._prepare_data(comment_texts=comment_texts)
            del comment_texts
            self.input_ids = [data["input_ids"] for data in process_data]
            self.attention_mask = [data["attention_mask"] for data in process_data]
            self.labels = [data["labels"] for data in process_data]
            
            
            if preprocessed_data_path:
                # Save preprocessed data for future use
                torch.save(process_data, preprocessed_data_path)

    def _prepare_data(self,comment_texts):
        # Generate batch of data
        processed_data = []
        for i in tqdm(range(0, len(comment_texts), self.batch_size), desc="Preprocessing data"):
            
            batch_comment_texts = comment_texts[i:i+self.batch_size]
            # print(batch_comment_texts)
            batch_toxicity_scores = self.toxicity_scores[i:i+self.batch_size]
            batch_toxic_values = self.toxic_values[i:i+self.batch_size]
            encoded_batch = self.tokenizer.batch_encode_plus(
                batch_comment_texts,
                truncation=self.truncation,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="pt"
            )

            encoded_labels = self.obtain_labels(batch_toxicity_scores, batch_toxic_values)
            
            for j in range(len(batch_comment_texts)):
                processed_data.append({
                    "input_ids": encoded_batch["input_ids"][j],
                    "attention_mask": encoded_batch["attention_mask"][j],
                    "labels": encoded_labels[j].float(),
                    "toxicity": batch_toxicity_scores[j],
                    "toxic_values": batch_toxic_values[j],
                })

        return processed_data
    

    def calculate_disagreement_weights(self,toxic_values):

        if self.disagreement == "variance":
            encoded_disagreement = get_variance_labels(toxic_values)
        elif self.disagreement == "distance":
            encoded_disagreement = get_distance_labels(toxic_values)
        elif self.disagreement == "entropy":
            encoded_disagreement = get_entropy_labels(toxic_values)
        else:
            raise ValueError("Disagreement method not recognized.")
        
        
        class_labels= self.get_one_hot_tensors(self.toxicity_scores)
        
        # return self.get_weights_by_class(encoded_disagreement, class_labels)
        return self.get_weights(encoded_disagreement)
    
    def calculate_toxicity_score_weights(self,toxicity_scores):

        return self.get_weights(toxicity_scores)
    
    def get_weights(self,input,num_bins=10):

        np_input = np.array(input)
        bins = np.linspace(0, 1.00000001, num_bins + 1)
        bin_indices = np.digitize(np_input, bins) - 1  # Get bin indices for each disagreement value
        bin_counts = np.bincount(bin_indices, minlength=num_bins)  # Count the number of values in each bin
        total_samples = len(np_input)
        bin_weights = total_samples / (num_bins * bin_counts.astype(float))
        weights = bin_weights[bin_indices]
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return weights_tensor
    
    def get_weights_by_class(self, input, class_labels, num_bins=10):
        """
        Calculate weights based on the input values and class labels.

        Parameters:
        input (list or array-like): Input values to be binned.
        class_labels (list or array-like): Class labels corresponding to each input value.
        num_bins (int): Number of bins to use for binning the input values.

        Returns:
        torch.Tensor: Weights tensor with class-dependent adjustments.
        """
        np_input = np.array(input)
        np_labels = np.array(class_labels)

        bins = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(np_input, bins) - 1  # Get bin indices for each input value
        # print(bin_indices)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        total_samples = len(np_input)
        
        # Initialize bin counts for each class
        bin_class_counts = {label: np.zeros(num_bins) for label in np.unique(class_labels)}
        # print(bin_class_counts)
        # Count the number of values in each bin for each class
        for i, bin_idx in enumerate(bin_indices):
            class_label = np_labels[i]
            bin_class_counts[class_label][bin_idx] += 1
        # print(bin_class_counts)
        # Compute bin weights for each class
        bin_class_weights = {}
        for class_label, counts in bin_class_counts.items():
            bin_class_weights[class_label] = total_samples / (num_bins * counts.astype(float))
        # print(bin_class_weights)
        # Create a weights array based on the input bin indices and class labels
        weights = np.zeros_like(np_input, dtype=float)
        for i, bin_idx in enumerate(bin_indices):
            class_label = np_labels[i]
            if bin_class_weights[class_label][bin_idx] != np.inf:
                weights[i] = bin_class_weights[class_label][bin_idx]
            else:
                weights[i] = 0
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return weights_tensor
    
    def get_classification_labels(self,toxicity_scores):
        # if self.soft_labels:
        #     encoded_labels = self.get_soft_labels(toxicity_scores)
        # else:
        #     encoded_labels = self.get_one_hot_tensors(toxicity_scores)
        encoded_labels = self.get_soft_labels(toxicity_scores)

        return torch.Tensor(encoded_labels)
    
    def get_regression_labels(self,toxic_values,encoded_labels):

        if self.disagreement == "variance":
            encoded_disagreement = get_variance_labels(toxic_values)
        elif self.disagreement == "distance":
            encoded_disagreement = get_distance_labels(toxic_values)
        elif self.disagreement == "entropy":
            encoded_disagreement = get_entropy_labels(toxic_values)
        else:
            raise ValueError("Disagreement method not recognized.")
        

        encoded_disagreement = torch.Tensor(encoded_disagreement)
        encoded_labels = encoded_labels.unsqueeze(1)
        encoded_disagreement = encoded_disagreement.unsqueeze(1)
        encoded_labels= torch.cat((encoded_labels, encoded_disagreement), dim=1)

        del encoded_disagreement
        
        return encoded_labels

    def obtain_labels(self,toxicity_scores,toxic_values):

        encoded_labels=self.get_classification_labels(toxicity_scores)
        encoded_labels=self.get_regression_labels(toxic_values,encoded_labels)
        return encoded_labels
    
    def update_labels(self,task="classification",disagreement="variance"):
        self.disagreement=disagreement
        self.task=task
        
        for i in tqdm(range(0, len(self.toxicity_scores), self.batch_size), desc="Updating data"):
            # print(self.dataset[i:i+self.batch_size])
            batch_toxic_scores= [data for data in self.toxicity_scores[i:i+self.batch_size]]
            batch_toxic_values= [data for data in self.toxic_values[i:i+self.batch_size]]
            encoded_labels = self.obtain_labels(batch_toxic_scores,batch_toxic_values)
            for j in range(len(encoded_labels)):
                # print(encoded_labels[j].float().device)

                self.labels[i+j]=encoded_labels[j].float()
        del batch_toxic_scores
        del batch_toxic_values
        del encoded_labels

    def get_one_hot_tensors(self,toxicity_scores, threshold=0.5):

        return [0 if score < threshold else 1 for score in toxicity_scores]
    
    def get_soft_labels(self,toxicity_scores):

        return [score for score in toxicity_scores]
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def __len__(self):
        return len(self.toxicity_scores)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if isinstance(idx, list):
            batch_data = {
                "input_ids": torch.stack([self.input_ids[i] for i in idx]),
                "attention_mask": torch.stack([self.attention_mask[i] for i in idx]),
                "labels": torch.stack([self.labels[i] for i in idx]),
                "dis_weights": torch.stack([self.disagreement_weights[i] for i in idx]),
                "tox_weights": torch.stack([self.toxicity_scores_weights[i] for i in idx])
            }
            return batch_data
        else:
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
                "dis_weights": self.disagreement_weights[idx],
                "tox_weights": self.toxicity_scores_weights[idx]
            }
       
    

class JigsawUnintendedDataModule(pl.LightningDataModule):
    def __init__(self, train_csv:str, val_csv:str,test_csv:str, tokenizer: PreTrainedTokenizerBase,
                batch_size: int = 32,
                padding: Union[str, bool] = "max_length",
                truncation: str = "only_first",
                max_length: int = 128,
                task:str="classification",
                disagreement:str="variance",
                dataset: Optional[Dataset] = None,
                preprocess_data_path:Optional[str]=None,
                num_workers:int=6):
        """
        This class is used to load the Jigsaw Unintended Bias in Toxicity Classification dataset.
            train_csv: str: Path to the training dataset file.
            val_csv: str: Path to the validation dataset file.
            test_csv: str: Path to the testing dataset file.
            batch_size: int: Batch size for the DataLoader.
            num_workers: int: Number of workers for the DataLoader.
            padding: str: Padding method for the tokenizer.
            truncation: str: Truncation method for the tokenizer.
            max_length: int: Maximum length for the tokenizer.
            task: str: Task to perform. Either classification or multitask.
            disagreement: str: Disagreement method for the multitask. It can be variance, distance or entropy.
            tokenizer: PreTrainedTokenizerBase: Tokenizer to use.
            dataset: Optional[Dataset]: Dataset to use.
            preprocess_data_path: Optional[str]: Path to save the preprocessed data.
        """
        super().__init__()
        
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.task = task
        self.disagreement = disagreement
        self.dataset = dataset
        self.preprocess_data_path= preprocess_data_path
        self.num_workers=num_workers
        self.train_dataset=None
        self.val_dataset=None
        self.test_dataset=None
    def setup(self, stage=None):
        if stage in ("fit", None):
            print("Entering fit stage")
            if self.dataset is None and self.train_dataset is None and self.val_dataset is None:
                self.train_dataset = JigsawUnintendedDataset(
                    file_path=self.train_csv, tokenizer=self.tokenizer, max_length=self.max_length,
                    batch_size=self.batch_size, padding=self.padding, truncation=self.truncation,
                    task=self.task, disagreement=self.disagreement,preprocessed_data_path=self.preprocess_data_path+"jigsaw_unintended_train.pt"

                )

                self.val_dataset = JigsawUnintendedDataset(
                    file_path=self.val_csv, tokenizer=self.tokenizer, max_length=self.max_length,
                    batch_size=self.batch_size, padding=self.padding, truncation=self.truncation,
                    task=self.task, disagreement=self.disagreement, preprocessed_data_path=self.preprocess_data_path+"jigsaw_unintended_val.pt"
                )
            elif self.dataset is None:
                self.train_dataset.update_labels(task=self.task,disagreement=self.disagreement)
                self.val_dataset.update_labels(task=self.task,disagreement=self.disagreement)
            else:
                if self.train_dataset is None and self.val_dataset is None:
                    self.train_dataset = JigsawUnintendedDataset(
                        tokenizer=self.tokenizer, max_length=self.max_length,
                        batch_size=self.batch_size, padding=self.padding, truncation=self.truncation,
                        task=self.task, disagreement=self.disagreement, data=self.dataset["train"].to_pandas(),preprocessed_data_path=self.preprocess_data_path+"jigsaw_unintended_train.pt"

                    )
                    self.val_dataset = JigsawUnintendedDataset(
                        tokenizer=self.tokenizer, max_length=self.max_length,
                        batch_size=self.batch_size, padding=self.padding, truncation=self.truncation,
                        task=self.task, disagreement=self.disagreement, data=self.dataset["valid"].to_pandas(),preprocessed_data_path=self.preprocess_data_path+"jigsaw_unintended_val.pt"
                    )
                else:
                    self.train_dataset.update_labels(task=self.task,disagreement=self.disagreement)
                    self.val_dataset.update_labels(task=self.task,disagreement=self.disagreement)

        if stage in ("test", None):
            if self.dataset is None and self.test_dataset is None :
                self.test_dataset = JigsawUnintendedDataset(
                    file_path=self.test_csv, tokenizer=self.tokenizer, max_length=self.max_length,
                    batch_size=self.batch_size, padding=self.padding, truncation=self.truncation,
                    task=self.task, disagreement=self.disagreement,preprocessed_data_path=self.preprocess_data_path+"jigsaw_unintended_test.pt"
                )
            elif self.dataset is None:
                self.test_dataset.update_labels(task=self.task,disagreement=self.disagreement)
            else:
                if self.test_dataset is None:
                    self.test_dataset = JigsawUnintendedDataset(
                        tokenizer=self.tokenizer, max_length=self.max_length,
                        batch_size=self.batch_size, padding=self.padding, truncation=self.truncation,
                        task=self.task, disagreement=self.disagreement, data=self.dataset["test"].to_pandas(),preprocessed_data_path=self.preprocess_data_path+"jigsaw_unintended_test.pt"
                    )
                else:
                    self.test_dataset.update_labels(task=self.task,disagreement=self.disagreement)
        if stage in ("validate",None):
            if self.dataset is None and self.val_dataset is None:
                self.val_dataset = JigsawUnintendedDataset(
                    file_path=self.val_csv, tokenizer=self.tokenizer, max_length=self.max_length,
                    batch_size=self.batch_size, padding=self.padding, truncation=self.truncation,
                    task=self.task, disagreement=self.disagreement, preprocessed_data_path=self.preprocess_data_path+"jigsaw_unintended_val.pt"
                )
            elif self.dataset is None:
                self.val_dataset.update_labels(task=self.task,disagreement=self.disagreement)
            else:
                if self.val_dataset is None:
                    self.val_dataset = JigsawUnintendedDataset(
                        tokenizer=self.tokenizer, max_length=self.max_length,
                        batch_size=self.batch_size, padding=self.padding, truncation=self.truncation,
                        task=self.task, disagreement=self.disagreement, data=self.dataset["valid"].to_pandas(),preprocessed_data_path=self.preprocess_data_path+"jigsaw_unintended_val.pt"
                    )
                else:
                   self.val_dataset.update_labels(task=self.task,disagreement=self.disagreement) 
   
    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            # sampler=BatchSampler(SequentialSampler(self.train_dataset), batch_size=self.batch_size, drop_last=False),
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed)
        )
   
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            # sampler=BatchSampler(SequentialSampler(self.val_dataset), batch_size=self.batch_size, drop_last=False),
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(seed)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            # sampler=BatchSampler(SequentialSampler(self.test_dataset), batch_size=self.batch_size, drop_last=False),
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(seed)
        )
