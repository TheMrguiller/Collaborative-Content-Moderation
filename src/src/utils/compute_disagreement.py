import numpy as np
import torch

def get_variance_labels(toxic_values):

    disagreement = []
    
    for value in toxic_values:
        
        disagreement.append(uncertainty_variance(value))

    return disagreement

def get_distance_labels(toxic_values):
    disagreement = []
    for value in toxic_values:
        disagreement.append(distance_based_metrics(value))

    return disagreement

def get_entropy_labels(toxic_values):
    disagreement = []
    for value in toxic_values:
        disagreement.append(entropy_binary_distribution(value))

    return disagreement

def distance_based_metrics(group):
    """
    Calculates the variance of the mean of proportions for a given group.

    Args:
    group (list): List of binary values (0 or 1) representing occurrences and non-occurrences of events.

    Returns:
    float: The variance of the mean of proportions.
    """
    mean = np.mean(group)
     # Sum of y_ij that are 1 (occurrences)
    certainty=abs(0.5-mean)
    uncertainty=1-certainty*2
    return uncertainty

def entropy_binary_distribution(group):
    """
    Calculates the entropy of a binary distribution for a given group.

    Args:
    group (list): List of binary values (0 or 1) representing occurrences and non-occurrences of events.

    Returns:
    float: The entropy of the binary distribution.
    """
    # Convert the list to a numpy array for easier computation
    group = np.array(group)
    
    # Calculate the probability of occurrence of 1s
    p = np.mean(group)
    
    # If p is 0 or 1, the entropy is 0
    if p == 0 or p == 1:
        return 0.0
    
    # Calculate the entropy of the binary distribution
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    return entropy

def uncertainty_variance(group):
    p = np.mean(group)
    uncertainty = p * (1 - p)
    # We multiply by 4 to make the values similar to the other metrics
    return uncertainty*4