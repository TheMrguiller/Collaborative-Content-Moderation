import torch
import numpy as np
import torch.nn.functional as F
from mapie.metrics import classification_coverage_score,classification_mean_width_score
from scipy.stats import pointbiserialr,pearsonr


def mean_bias_error(predictions,true_values):
    """
    Computes the mean bias error between the true values and the predictions.
    
    Args:
        true_values (Tensor): A float tensor of arbitrary shape.
                The true values for each example.
        predictions (Tensor): A float tensor with the same shape as true_values.
                The predictions for each example.
    Returns:
        The mean bias error between the true values and the predictions.
    """
    # Ensure both tensors are of the same dtype
    
    return (predictions- true_values).mean()

def distance_to_interval(value, interval):
    """
    Calculate the distance from a value to an interval when the value is outside the interval.

    Parameters:
    value (float): The value to check.
    interval (tuple): A tuple (a, b) representing the interval [a, b] with a < b.

    Returns:
    float: The distance to the interval. Returns 0 if the value is within the interval.
    """
    a, b = interval
    
    if value < a:
        return a - value  # Distance to the left of the interval
    elif value > b:
        return value - b  # Distance to the right of the interval
    else:
        return 0  # The value is within the interval

def DI_metric(true_values:np.ndarray, lower_bounds:np.ndarray, upper_bounds:np.ndarray):
    """
    Distance to the interval metric. DI
    """
    metric_list=[]
    for i in range(len(true_values)):
        metric_list.append(distance_to_interval(true_values[i], (lower_bounds[i], upper_bounds[i])))
    return np.mean(metric_list)

def coverage_probability(true_values, lower_bounds, upper_bounds):
    """
    Computes the coverage probability of the lower and upper bounds for the true values. MIL
    
    Args:
        true_values (Tensor): A float tensor of arbitrary shape.
                The true values for each example.
        lower_bounds (Tensor): A float tensor with the same shape as true_values.
                The lower bounds for each example.
        upper_bounds (Tensor): A float tensor with the same shape as true_values.
                The upper bounds for each example.
    Returns:
        The coverage probability of the lower and upper bounds for the true values.
    """
    # Ensure all tensors are of the same dtype
    dtype = true_values.dtype
    lower_bounds = lower_bounds.to(dtype)
    upper_bounds = upper_bounds.to(dtype)
    
    return ((true_values >= lower_bounds) & (true_values <= upper_bounds)).float().mean()

def interval_width(lower_bounds, upper_bounds):
    
    """
    Computes the width of the intervals defined by the lower and upper bounds.
    
    Args:
        lower_bounds (Tensor): A float tensor of arbitrary shape.
                The lower bounds for each interval.
        upper_bounds (Tensor): A float tensor with the same shape as lower_bounds.
                The upper bounds for each interval.
    Returns:
        The width of the intervals defined by the lower and upper bounds.
    """
    # Ensure both tensors are of the same dtype
    # Compute the width of the intervals
    interval_widths = upper_bounds - lower_bounds
    
    # Calculate statistics
    mean_width = interval_widths.mean()
    median_width = interval_widths.median()
    quantile_25 = interval_widths.quantile(0.25)
    quantile_75 = interval_widths.quantile(0.75)
    return mean_width, median_width, quantile_25, quantile_75

def interval_width_correlation_disagreement(lower_bounds:torch.Tensor, upper_bounds:torch.Tensor,disagreement:torch.Tensor):
    """
    Computes the correlation between the interval width and the disagreement.
    
    Args:
        lower_bounds (Tensor): A float tensor of arbitrary shape.
                The lower bounds for each interval.
        upper_bounds (Tensor): A float tensor with the same shape as lower_bounds.
                The upper bounds for each interval.
        disagreement (Tensor): A float tensor with the same shape as lower_bounds.
                The disagreement for each interval.
    Returns:
        The correlation between the interval width and the disagreement.
    """
    # Ensure all tensors are of the same dtype
    dtype = lower_bounds.dtype
    upper_bounds = upper_bounds.to(dtype)
    disagreement = disagreement.to(dtype)
    
    return pearsonr((upper_bounds.numpy()-lower_bounds.numpy()), disagreement.numpy())[0]
def mean_absolute_error(true_values, predictions):
    """
    Computes the mean absolute error between the true values and the predictions.
    
    Args:
        true_values (Tensor): A float tensor of arbitrary shape.
                The true values for each example.
        predictions (Tensor): A float tensor with the same shape as true_values.
                The predictions for each example.
    Returns:
        The mean absolute error between the true values and the predictions.
    """
    # Ensure both tensors are of the same dtype
    dtype = true_values.dtype
    predictions = predictions.to(dtype)
    
    return (true_values - predictions).abs().mean()

def mean_squared_error(true_values, predictions):
    """
    Computes the mean squared error between the true values and the predictions.
    
    Args:
        true_values (Tensor): A float tensor of arbitrary shape.
                The true values for each example.
        predictions (Tensor): A float tensor with the same shape as true_values.
                The predictions for each example.
    Returns:
        The mean squared error between the true values and the predictions.
    """
    # Ensure both tensors are of the same dtype
    dtype = true_values.dtype
    predictions = predictions.to(dtype)
    
    return ((true_values - predictions) ** 2).mean()

def computelog_loss(true_values, predictions):
    """
    Computes the log loss between the true values and the predictions.

    Args:
        true_values (torch.Tensor): The ground truth values.
        predictions (torch.Tensor): The predicted probabilities.

    Returns:
        torch.Tensor: The log loss between the true values and the predictions.
    """
    # Ensure both tensors are of the same dtype
    if predictions.dtype != true_values.dtype:
        predictions = predictions.to(true_values.dtype)
    
    return torch.nn.functional.binary_cross_entropy(predictions, true_values)

########################### Code from  https://github.com/Jonathan-Pearce/calibration_library/tree/master

def calculate_ace(y_true, y_pred, num_bins=10):

    y_pred_numpy = y_pred.numpy()
    y_true_numpy = y_true.numpy()

    # b =np.linspace(0, 1.00000001, num_bins + 1)
    b = np.linspace(start=0, stop=1.0, num=num_bins)
    b = np.quantile(y_pred, b)
    b = np.unique(b)
    num_bins = len(b)
    bins = np.digitize(y_pred_numpy, bins=b, right=True)
    # print(bins)

    ece = 0
    num_samples = y_pred_numpy.shape[0]
    ece_per_bin=[]
    bin_value_bin_edges=[]
    for b in range(num_bins):
        mask = bins == b
        # print(mask)
        if np.any(mask):
            bin_pred = y_pred_numpy[mask]
            bin_pred_one_hot = (bin_pred >= 0.5).astype(int)
            bin_true = y_true_numpy[mask]
            # print(f"Bin True:{bin_true},Bin Pred:{bin_pred}")
            # print((bin_true == bin_pred_one_hot).astype(int))
            bin_accuracy = np.mean((bin_true == bin_pred_one_hot).astype(int))
            bin_confidence = np.mean(bin_pred)
            ece_per_bin.append(bin_accuracy)
            bin_value_bin_edges.append(bin_confidence)
            # print(f"Bin_accuracy:{bin_accuracy},Bin_confidence:{bin_confidence}")
            ece += np.abs(bin_accuracy - bin_confidence) 

    return ece * 1.0 / num_samples,ece_per_bin,bin_value_bin_edges

def base_cp_metrics(prediction_set:np.ndarray,labels:np.ndarray):

    label=(labels[:,0]>=0.5).astype(int)
    # Marginal coverage
    marginal_coverage=classification_coverage_score(label,prediction_set)
    # Conditional coverage
    toxic_coverage=classification_coverage_score(label[label==1],prediction_set[label==1])
    no_toxic_coverage=classification_coverage_score(label[label==0],prediction_set[label==0])
    # Prediction set
    set_length=classification_mean_width_score(prediction_set)
    # prediction set, marginal coverage, conditional coverage
    return set_length, marginal_coverage, (no_toxic_coverage,toxic_coverage)

def uncertainty_correlation_prediction_set(uncertainty_cp:list,distance_disagreement:list):
    correlation, p_value = pointbiserialr(uncertainty_cp, distance_disagreement)
    return correlation

def uncertainty_confusion_matrix(y_actual, y_pred,uncertainty_cp):
    TP = []
    FP = []
    TN = []
    FN = []

    for i in range(len(y_pred)): 
        if y_actual[i]!=y_pred[i] and uncertainty_cp[i]==1:
           TP.append(1)
        else:
            TP.append(0)
        if y_pred[i]==y_actual[i] and uncertainty_cp[i]==0:
            TN.append(1)
        else:
            TN.append(0)
        if y_actual[i]!=y_pred[i] and uncertainty_cp[i]==0:
            FN.append(1)
        else:
            FN.append(0)
        if y_pred[i]==y_actual[i] and uncertainty_cp[i]==1:
            FP.append(1)
        else:
            FP.append(0)

    return (sum(TP), sum(FP), sum(TN), sum(FN))

def uncertainty_base_metrics(TP:int,FP:int,TN:int,FN:int):
    
    if TP>0 or FP>0:
        # Review Efficiency,the fraction of inaccurate examples where the model is uncertain
        MURE= TP/(TP+FP)
        # The fraction of uncertain examples where the model is inaccurate
    else:
        MURE=None
    if TP>0 or FN>0:
        uncertain_examples_model_inaccurate= TP/(TP+FN)
    else:
        uncertain_examples_model_inaccurate=None
        # The fraction of under-confident examples among the correct predictions
    if FP>0 or TN>0:
        under_confident_examples= FP/(FP+TN)
    else:
        under_confident_examples=None
    # f1-score uncertainty
    if TP>0 or FP>0 or FN>0:
        f1_score= 2*TP/(2*TP+FP+FN)
    else:
        f1_score=None

    return MURE,uncertain_examples_model_inaccurate,under_confident_examples,f1_score

def disagreement_bin_uncertainty_base_metrics(distance_disagreement:list,predictions:np.ndarray,labels:np.ndarray,uncertainty:list):
    bins = np.arange(0,1.1,0.1)
    binned_data = np.digitize(distance_disagreement, bins, right=False)
    dict_={}
    for i in range(0,len(bins)):
        mask = (binned_data == i).astype(int)
        TP, FP, TN, FN=uncertainty_confusion_matrix((labels[:,0]>=0.5)[mask==1],(predictions[:,0]>=0.5)[mask==1],np.array(uncertainty)[mask==1])
        
        if TP >0 or FP>0 or FN:
            MURE,uncertain_examples_model_inaccurate,under_confident_examples,f1_score=uncertainty_base_metrics(TP, FP, TN, FN)
            dict_[bins[i]]={
                "MURE":MURE,
                "f1_score":f1_score,
                "uncertain_examples_model_inaccurate":uncertain_examples_model_inaccurate,
                "under_confident_examples":under_confident_examples
            }
        else:
            dict_[bins[i]]={
                "MURE":None,
                "f1_score":None,
                "uncertain_examples_model_inaccurate":None,
                "under_confident_examples":None
            }
    return dict_



