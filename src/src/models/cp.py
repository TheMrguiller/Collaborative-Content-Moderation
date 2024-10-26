import numpy as np
import torch
from typing import Union
from torch.cuda.amp import autocast
from scipy.optimize import brentq
import warnings
import math
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import KNeighborsRegressor

from sklearn.decomposition import PCA
import gc
from tqdm import tqdm
## CLASSIFICATION Conformal Prediction
class LAC:
    """
    Comformal Prediction Least Ambiguous set-valued Classifier , it uses the true labels to compute qhat.
    """

    def __init__(self,alpha=0.1) -> None:
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def compute_qhat(self, y_pred: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute the qhat for the given predictions and true labels.
        :param y_true: True labels.
        :param y_pred: Predictions.
        :return: qhat.
        """
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        
        n = y_true.size(0)
        n = torch.tensor(n, dtype=torch.float32)
        cal_scores = 1-y_pred[np.arange(n),torch.argmax(y_true, dim=1)].float()
        q_level = torch.ceil((n + 1) * (1 - self.alpha)) / n
        qhat = torch.quantile(cal_scores, q_level.float(), interpolation='higher')
        return qhat
    
    def compute_prediction_set(self, y_pred: Union[np.ndarray, torch.Tensor], qhat:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute the prediction set for the given predictions and true labels.
        :param y_true: True labels.
        :param y_pred: Predictions.
        :return: Prediction set.
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        
        prediction_sets = y_pred >= (1-qhat)
        return prediction_sets
    
class CCLAC:
    """Class-Conditional LAC (CCLAC)"""
    def __init__(self,alpha=0.1) -> None:
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def compute_qhat(self, y_pred: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute the qhat_toxic, qhat_non_toxic for the given predictions and true labels.
        :param y_true: True labels.
        :param y_pred: Predictions.
        :return: qhat_toxic, qhat_non_toxic.
        """
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)

    
        true_class_indices = torch.argmax(y_true, dim=1)
        cal_scores_toxic = 1-y_pred[true_class_indices == 1][:,1]
        cal_scores_non_toxic = 1-y_pred[true_class_indices == 0][:,0]
        n = y_true[true_class_indices == 1].size(0)
        n = torch.tensor(n, dtype=torch.float32)
        q_level = torch.ceil((n + 1) * (1 - self.alpha)) / n
        qhat_toxic = torch.quantile(cal_scores_toxic.float(), q_level.float(), interpolation='higher')
        n = y_true[true_class_indices == 0].size(0)
        n = torch.tensor(n, dtype=torch.float32)
        q_level = torch.ceil((n + 1) * (1 - self.alpha)) / n
        qhat_non_toxic = torch.quantile(cal_scores_non_toxic.float(), q_level.float(), interpolation='higher')
        return qhat_toxic, qhat_non_toxic
    
    def compute_prediction_set(self, y_pred: Union[np.ndarray, torch.Tensor], qhat_toxic:Union[np.ndarray, torch.Tensor], qhat_non_toxic:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute the prediction set for the given predictions and true labels.
        :param y_true: True labels.
        :param y_pred: Predictions.
        :return: Prediction set.
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        prediction_sets = torch.stack(((1-y_pred[:,0]) <= qhat_non_toxic, (1-y_pred[:,1]) <= qhat_toxic), dim=1)
        return prediction_sets



class CRC:
    """Conformal Risk Control (CRC)"""
    def __init__(self,alpha=0.1) -> None:
        self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def false_negative_rate(self,prediction_set, labels):
        return 1-((prediction_set * labels).sum(axis=1)/labels.sum(axis=1)).mean()
    
    def lamhat_threshold(self,lam): 
        return self.false_negative_rate(self.cal_pred>=lam, self.cal_ytrue) - ((self.n+1)/self.n*self.alpha - 1/self.n)
    
    def compute_qhat(self,y_pred:Union[np.ndarray, torch.Tensor],y_true:Union[np.ndarray, torch.Tensor]):
        
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        
        self.n = y_true.shape[0]
        self.cal_pred = y_pred
        self.cal_ytrue = y_true
        lamhat = brentq(self.lamhat_threshold, 0, 1)
        return lamhat
        
    def compute_prediction_set(self, y_pred: Union[np.ndarray, torch.Tensor], qhat:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        
        prediction_sets = y_pred >= qhat
        return torch.tensor(prediction_sets)


## REGRESSION Conformal Prediction

class AbsoluteResidual:
    def __init__(self,alpha=0.1) -> None:
        self.alpha = alpha
    def compute_qhat(self, y_pred: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        cal_scores = np.abs(y_true-y_pred)
        n= y_true.shape[0]
        del y_true,y_pred
        gc.collect()
        qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-self.alpha))/n, interpolation='higher')
        return torch.tensor(qhat)
    
    def compute_prediction_set(self, y_pred: Union[np.ndarray, torch.Tensor], qhat:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        if isinstance(qhat, torch.Tensor):
            qhat = qhat.numpy()
        # Calculate the prediction sets
        lower_bound = y_pred - qhat
        upper_bound = y_pred + qhat

        # Bound the elements between 0 and 1
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)
        
        # Combine into a prediction set
        prediction_sets = np.stack((lower_bound, upper_bound),axis=1)

        return torch.tensor(prediction_sets).t()

class Gamma:
    def __init__(self,alpha=0.1) -> None:
        self.alpha = alpha
    def compute_qhat(self, y_pred: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
            
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.numpy()
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.numpy()
            cal_scores = np.abs(y_true-y_pred)/(y_pred+1e-6)
            n= y_true.shape[0]
            qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-self.alpha))/n, interpolation='higher')
            return torch.tensor(qhat)
    def compute_prediction_set(self, y_pred: Union[np.ndarray, torch.Tensor], qhat:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        if isinstance(qhat, torch.Tensor):
            qhat = qhat.numpy()
        # Calculate the prediction sets
        lower_bound = y_pred*(1-qhat)
        upper_bound = y_pred*(1+qhat)

        # Bound the elements between 0 and 1
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)
        print(lower_bound.shape,upper_bound.shape)
        # Combine into a prediction set
        prediction_sets = np.stack((lower_bound, upper_bound),axis=1)

        return torch.tensor(prediction_sets)




class RACCP:
    """
    https://github.com/EtashGuha/R2CCP/blob/main/R2CCP/main.py#L121
    """
    def __init__(self,num_classes,alpha=0.1) -> None:
        self.alpha = alpha
        self.midpoints = torch.linspace(1/num_classes, 1, steps=num_classes)
        
        
    def invert_intervals(self, intervals):
        temp_intervals = []
        for interval in intervals:
            curr_interval = []
            for tup in interval:
                curr_interval.append((tup[0].detach().numpy().reshape(-1, 1).item(), tup[1].detach().numpy().reshape(-1, 1)))
            temp_intervals.append(curr_interval)
        return temp_intervals
    
    def get_intervals(self, pred_scores, cal_pred, y_cal):
        if isinstance(pred_scores, np.ndarray):
            pred_scores = torch.from_numpy(pred_scores)
        if isinstance(cal_pred, np.ndarray):
            cal_pred = torch.from_numpy(cal_pred)
        if isinstance(y_cal, np.ndarray):
            y_cal = torch.from_numpy(y_cal)
        intervals = self.get_cp_lists(pred_scores,cal_pred, y_cal)
        # intervals = self.invert_intervals(intervals)
        intervals = self.get_real_intervals(pred_scores, intervals)
        return torch.tensor(intervals)
    
    
    def find_interval(self,value, sorted_list):
        """
        Finds the interval [lower_bound, upper_bound) in a sorted list where the value falls.
        
        Parameters:
        value (int): The value to find the interval for.
        sorted_list (list of int): A sorted list of integers.
        
        Returns:
        tuple: A tuple (lower_bound, upper_bound) representing the interval.
            Returns (None, None) if the value is outside the range of the list.
        """
        # Handle edge cases
        if value < sorted_list[0]:
            return (None, sorted_list[0])  # Value is below the first element
        if value >= sorted_list[-1]:
            return (sorted_list[-1], value.item())  # Value is above the last element
        
        # Iterate through the list to find the correct interval
        for i in range(len(sorted_list) - 1):
            lower_bound = sorted_list[i]
            upper_bound = sorted_list[i + 1]
            
            if lower_bound <= value < upper_bound:
                return (lower_bound, upper_bound)
        
        return (None, None)  # This line should never be reached if the list is properly sorted

    def get_real_intervals(self, pred_scores, intervals):
        """
        It is a modification of the base code where we are returning the real intervals, those where the pred is inside the interval.
        The base intervals are bimodal which makes no sense in our case.
        """
        new_intervals = []
        for i in range(0,len(intervals)):
            for interval in intervals[i]:
                if interval[0] <= self.midpoints[np.argmax(pred_scores[i])] <= interval[1]:
                    new_intervals.append(interval)
                    break
            if len(new_intervals) < i+1:
                flattened_list = np.concatenate(intervals[i]).tolist()
                interval=self.find_interval(self.midpoints[np.argmax(pred_scores[i])],flattened_list)
                if interval[0] is None or interval[1] is None:
                    interval = intervals[i][-1]
                new_intervals.append(interval)
        return new_intervals
    def get_cp_lists(self,pred_scores, cal_pred, y_cal):
        scores, all_scores = self.get_all_scores(self.midpoints, cal_pred, y_cal)
        
        alpha = self.alpha

        percentile_val = self.percentile_excluding_index(all_scores, alpha)
            
        all_intervals = []
        for i in range(len(pred_scores)):
            all_intervals.append(self.find_intervals_above_value_with_interpolation(self.midpoints, pred_scores[i], percentile_val))

        return all_intervals
    def find_intervals_above_value_with_interpolation(self,x_values, y_values, cutoff):
        intervals = []
        start_x = None
        if y_values[0] >= cutoff:
            start_x = x_values[0]
        for i in range(len(x_values) - 1):
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = y_values[i], y_values[i + 1]

            if min(y1, y2) <= cutoff < max(y1, y2):
                # Calculate the x-coordinate where the line crosses the cutoff value
                x_cross = x1 + (x2 - x1) * (cutoff - y1) / (y2 - y1)

                if x1 <= x_cross <= x2:
                    if start_x is None:
                        start_x = x_cross
                    else:
                        intervals.append((start_x.item(), x_cross.item()))
                        start_x = None

        # If the line ends above cutoff, add the last interval
        if start_x is not None:
            intervals.append((start_x.item(), x_values[-1].item()))

        return intervals
    
    def percentile_excluding_index(self,vector, percentile):
        percentile_value = torch.quantile(vector, percentile)
        
        return percentile_value
    
    def get_all_scores(self,range_vals, cal_pred, y):
        step_val = (max(range_vals) - min(range_vals))/(len(range_vals) - 1)
        indices_up = torch.ceil((y - min(range_vals))/step_val).squeeze()
        indices_down = torch.floor((y - min(range_vals))/step_val).squeeze()
        
        how_much_each_direction = ((y.squeeze() - min(range_vals))/step_val - indices_down)

        weight_up = how_much_each_direction
        weight_down = 1 - how_much_each_direction

        # bad_indices = torch.where(torch.logical_or(y.squeeze() > max(range_vals), y.squeeze() < min(range_vals)))
        # indices_up[bad_indices] = 0
        # indices_down[bad_indices] = 0
        
        scores = cal_pred
        all_scores = scores[torch.arange(cal_pred.shape[0]), indices_up.long()] * weight_up + scores[torch.arange(cal_pred.shape[0]), indices_down.long()] * weight_down
        # all_scores[bad_indices] = 0
        return scores, all_scores
    

class AbsErrorErrFunc:
	"""Calculates absolute error nonconformity for regression problems.

		For each correct output in ``y``, nonconformity is defined as

		.. math::
			| y_i - \hat{y}_i |
	"""
	def apply(self, prediction, y):
		return np.abs(prediction - y)

	def apply_inverse(self, nc, significance):
		nc = np.sort(nc)[::-1]
		border = int(np.floor(significance * (nc.size + 1))) - 1
		# TODO: should probably warn against too few calibration examples
		border = min(max(border, 0), nc.size - 1)
		return np.vstack([nc[border], nc[border]])
    
class RN:
    "Residual Normalized Conformity Score (RN)"
    # https://github.com/Quilograma/ConformalPredictionTutorial/blob/main/Conformal%20Prediction.ipynb
    def __init__(self,alpha=0.1,model_name="distilbert/distilbert-base-uncased") -> None:
        self.alpha = alpha
        self.model=AutoModel.from_pretrained(model_name)
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.knn= KNeighborsRegressor(n_neighbors=5,weights='distance')
        self.err_func = AbsErrorErrFunc()
        self.pca = PCA(n_components=50)

    def get_embeddings(self, texts, batch_size=64):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs.to(self.device)
            with autocast():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()  # Move to CPU to reduce GPU memory usage
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)
    
    def train_KNN(self,texts,predictions,labels):
        X=self.get_embeddings(texts)
        # Apply PCA before UMAP
        
        X=self.pca.fit_transform(X)
        residual_error = np.abs(self.err_func.apply(predictions, labels))
        residual_error += 0.00001 # Add small term to avoid log(0)
        log_err = np.log(residual_error)
        self.knn.fit(X, log_err)
        del X,residual_error,log_err
        gc.collect()
        
    def compute_qhat(self,text, y_pred: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
            
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.numpy()
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.numpy()
            x= self.get_embeddings(text)
            x=self.pca.transform(x)
            norm_scores= np.exp(self.knn.predict(x))
            cal_scores = self.err_func.apply(y_pred, y_true) / norm_scores
            n= y_true.shape[0]
            qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-self.alpha))/n, interpolation='higher')
            del x,cal_scores
            gc.collect()
            return torch.tensor(qhat)
    
    def compute_prediction_set(self,text, y_pred: Union[np.ndarray, torch.Tensor], qhat:Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.numpy()
        if isinstance(qhat, torch.Tensor):
            qhat = qhat.numpy()
        x= self.get_embeddings(text)
        x=self.pca.transform(x)
        norm_scores= np.exp(self.knn.predict(x))
        # Calculate the prediction sets
        lower_bound = y_pred-qhat*norm_scores
        upper_bound = y_pred+qhat*norm_scores

        # Bound the elements between 0 and 1
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)

        # Combine into a prediction set
        prediction_sets = [lower_bound, upper_bound]
        del x,norm_scores,lower_bound,upper_bound
        gc.collect()
        return torch.tensor(prediction_sets).t()


import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split("src")[0]+"src/")
from src.models.metrics import base_cp_metrics,uncertainty_correlation_prediction_set,uncertainty_confusion_matrix,uncertainty_base_metrics,disagreement_bin_uncertainty_base_metrics
from src.utils.compute_disagreement import get_distance_labels
import glob

if __name__ == '__main__':
    project_path=os.path.dirname(os.path.abspath(__file__)).split("src")[0]+"src/"
    
    # class_base_test=np.load(project_path+"results/test_results/multitask_cross_entropy_Weighted_distance_WeightedR2ccpLoss.npz")
    # class_base_val=np.load(project_path+"results/val_results/multitask_cross_entropy_Weighted_distance_WeightedR2ccpLoss.npz")
    # raccp=RACCP(num_classes=class_base_test["predictions"][:,1:].shape[1],alpha=.1)
    # qhat=raccp.get_intervals(pred_scores=class_base_test["predictions"][:,1:],cal_pred=class_base_val["predictions"][:,1:],y_cal=class_base_val["labels"][:,1])
    # pass
    
    test_result_path= project_path+"results/test_results/"
    val_result_path= project_path+"results/val_results/"
    files = glob.glob(f"{test_result_path}/*") 
    for file_path in files:
        print(file_path)
        filename=file_path.rsplit("/")[-1].split(".npz")[0]
        if "regression" in filename:
            continue
        class_base_test=np.load(file_path)
        class_base_val=np.load(f"{val_result_path}{filename}.npz")
        conformalcp=WeakSupervisionCP(alpha=.1)
        y_pred=class_base_val["predictions"]
        y_true=class_base_val["labels"]
        y_true = y_true[:, 0]
        y_pred = y_pred[:, 0]
        y_true = (y_true>=0.5).astype(int)
        y_true_classwise = np.stack((1 - y_true,y_true), axis=1)
        y_pred_classwise = np.stack((1 - y_pred,y_pred), axis=1)

        lamhat=conformalcp.compute_qhat(y_pred=y_pred_classwise,y_true=y_true_classwise)
        print(lamhat)
        y_pred_test=class_base_test["predictions"]
        y_pred_test = y_pred_test[:, 0]
        y_pred_test_classwise = np.stack((1 - y_pred_test,y_pred_test), axis=1)
        prediction_set=conformalcp.compute_prediction_set(y_pred=y_pred_test_classwise,qhat=lamhat)
        distance_disagreement=get_distance_labels(class_base_test["labels"][:,0])
        uncertainty_cp=[1 if (prediction.all().item() or not prediction.any().item()) else 0 for prediction in prediction_set]
        print(uncertainty_correlation_prediction_set(uncertainty_cp,distance_disagreement))
        TP, FP, TN, FN=uncertainty_confusion_matrix(class_base_test["labels"][:,0]>=0.5,class_base_test["predictions"][:,0]>=0.5,uncertainty_cp)
        MURE,uncertain_examples_model_inaccurate,under_confident_examples,f1_score=uncertainty_base_metrics(TP, FP, TN, FN)
        print(f"MURE: {MURE}, uncertain_examples_model_inaccurate: {uncertain_examples_model_inaccurate}, under_confident_examples: {under_confident_examples}, f1_score: {f1_score}")