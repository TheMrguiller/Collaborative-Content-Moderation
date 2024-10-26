from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
import torch
from tensorboardX import SummaryWriter


def tensorboard_calibration_curve(pred:np.ndarray, labels:np.ndarray):
    """
    Compute the calibration curve for a model and the true labels.
        prediction_model: The predicted probabilities range 0 to 1. Torch tensor.
        labels_value: The true labels range 0 to 1. Torch tensor.
    """
    x, y = calibration_curve((labels>=0.5).astype(int), pred, n_bins = 10)
    # Plot calibration curve
    fig = plt.figure(figsize=(10, 8),dpi=50)
    # Plot perfectly calibrated
    plt.title('Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    
    # Plot model's calibration curve
    plt.plot(y, x, marker = '.')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.grid(True)
    plt.close(fig)
    return fig


def tensorboard_ace(ece_per_bin:np.ndarray,bin_edges:np.ndarray):
    """
    Compute the Expected Calibration Error (ECE) for a model and the true labels.
        ece_per_bin: The expected calibration error per bin. List.
        bin_edges: The bin edges. List.
    """
    fig = plt.figure(figsize=(10, 8),dpi=50)
    plt.title('Adaptive Calibration Error (ACE)')
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')

    # Plot model's calibration curve
    plt.plot(bin_edges,ece_per_bin,  marker = '.')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.close(fig)
    return fig

def tensorboard_toxic_score(true_toxic_score:np.ndarray,pred_toxic_score:np.ndarray):
    """
    Compute the Toxic Score for a model and the true labels.
        true_toxic_score: The true toxic score. List.
        pred_toxic_score: The predicted toxic score. List.
    """
    df = pd.DataFrame({
        'true_score': true_toxic_score,
        'pred_score': pred_toxic_score
    })
    
    step=0.1000000001
    bins = np.arange(0, 1.1, step)
    
    # Bin the true scores and calculate mean predicted score within each bin
    df['bin'] = pd.cut(df['true_score'], bins, right=False)
    mean_pred_score = df.groupby('bin')['pred_score'].mean()
    mean_true_score = df.groupby('bin')['true_score'].mean()
    
    fig = plt.figure(figsize=(10, 8),dpi=50)
    plt.title('True vs Predicted Toxic Score')
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    plt.plot(mean_true_score, mean_pred_score, marker = '.', label = 'Toxic Score')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('True Toxic Score')
    plt.ylabel('Predicted Toxic Score')
    plt.title('True vs Predicted Toxic Score')
    plt.grid(True)
    plt.close(fig)
    return fig


def tensorboard_mean_regression_error_disagreement(regression_pred:np.ndarray,regression_true:np.ndarray):
    """
    Mean regression error per disagreement bin.
        regression_pred: The predicted regression values. List.
        regression_true: The true regression values. List.
        disagreement: The disagreement values. List.
    """
    class_error= abs(regression_true-regression_pred)
    df = pd.DataFrame({
        'disagreement': regression_true,
        'error': class_error,
    })

    step=0.1000000001
    bins = np.arange(0, 1.1, step)
    df['bin'] = pd.cut(df['disagreement'], bins, right=False)

    # Calcular el error medio por cada bin
    mean_errors = df.groupby('bin')['error'].mean()

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(10, 8),dpi=50)
    mean_errors.plot(kind='bar', width=0.9)
    plt.title('Disagreement Error per Bin')
    plt.xlabel('Disagreement Bins')
    plt.ylabel('Disagrement Mean Error')
    plt.grid(True)
    plt.close(fig)
    return fig


def tensorboard_regression_accuracy(true_values:np.ndarray,predictions:np.ndarray):
    """
    Regression accuracy per disagreement bin.
        true_values: The true values. List.
        predictions: The predicted values. List.
    """
    df = pd.DataFrame({
        'disagreement': true_values,
        'predictions': predictions,
    })

    # Crear bins de 0.1 para los desacuerdos
    step=0.1000000001
    bins = np.arange(0, 1.1, step)
    df['bin'] = pd.cut(df['disagreement'], bins, right=False)
    
    df['accuracy'] = df.apply(lambda row: 1 if row['predictions'] >= row['bin'].left and row['predictions'] < row['bin'].right else 0, axis=1)  
    mean_errors = df.groupby('bin')['accuracy'].mean()

    # Graficar los resultados
    fig, ax = plt.subplots(figsize=(10, 8),dpi=50)
    mean_errors.plot(kind='bar', width=0.9)
    
    plt.title('Disagreement accuracy per Bin')
    plt.xlabel('Disagreement Bins')
    plt.ylabel('Disagrement Accuracy')
    plt.grid(True)
    plt.close(fig)
    return fig

def tensorboard_class_mean_error(pred_toxic_score:np.ndarray,true_toxic_score:np.ndarray,disagreement:np.ndarray):
    """
    Mean error per disagreement bin.
        pred_toxic_score: The predicted toxic score. List.
        true_toxic_score: The true toxic score. List.
        disagreement: The disagreement values. List.
    """
    toxic_class_error= abs(true_toxic_score-pred_toxic_score)
    no_toxic_class_error= abs((1-true_toxic_score)-(1-pred_toxic_score))
    df = pd.DataFrame({
        'disagreement': disagreement,
        'toxic_class_error': toxic_class_error,
        'no_toxic_class_error': no_toxic_class_error
    })

    step=0.1000000001
    bins = np.arange(0, 1.1, step)
    df['bin'] = pd.cut(df['disagreement'], bins, right=False)

    # Calculate the mean error per bin
    toxic_mean_errors = df.groupby('bin')['toxic_class_error'].mean()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8),dpi=50)

    # Plot the bars
    ax.bar(toxic_mean_errors.index.astype(str), toxic_mean_errors.values, color='blue', edgecolor='grey')

    # Adding labels and title
    ax.set_xlabel('Disagreement Bin')
    ax.set_ylabel('Mean Toxic Class Error')
    ax.set_title('Mean Toxic Class Error by Disagreement Bins')

    # Setting x-ticks and labels
    ax.set_xticks(range(len(toxic_mean_errors)))
    ax.set_xticklabels([f'{bin.left:.1f}-{bin.right:.1f}' for bin in toxic_mean_errors.index])

    plt.close(fig)
    return fig

def tensorboard_class_accuracy(pred_toxic_score:np.ndarray,true_toxic_score:np.ndarray,disagreement:np.ndarray):
    """
    Class accuracy per disagreement bin.
        pred_toxic_score: The predicted toxic score. List.
        true_toxic_score: The true toxic score. List.
        disagreement: The disagreement values. List.
    """
    
    df = pd.DataFrame({
        'disagreement': disagreement,
        'predictions': pred_toxic_score,
        'true_scores': true_toxic_score
    })

    # Create bins of 0.1 for disagreements
    step=0.1000000001
    bins = np.arange(0, 1.1, step)
    df['bin'] = pd.cut(df['disagreement'], bins, right=False)
    df["label"]= pd.cut(df['true_scores'], bins, right=False)
    
    # Calculate accuracy per bin
    def calculate_accuracy(bin_data):
        if len(bin_data) == 0:
            return np.nan
        pred_within_range=bin_data.apply(lambda row: 1 if (row['predictions'] >= row['label'].left) & (row['predictions'] <= row['label'].right) else 0,axis=1)
        # pred_within_range = (bin_data['predictions'] >= bin_data['label'].left) & (bin_data['predictions'] <= bin_data['label'].right)
        return pred_within_range.mean()
    
    mean_accuracy = df.groupby('bin').apply(calculate_accuracy)
    # print(mean_accuracy)
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 8),dpi=50)
    mean_accuracy.plot(kind='bar', width=0.9, ax=ax)
    
    plt.title('Class Accuracy per Bin of Disagreement for Toxicity score')
    plt.xlabel('Disagreement Bins')
    plt.ylabel('Class Accuracy')
    plt.grid(True)
    plt.close(fig)
    
    return fig

def tensorboard_model_toxic_pred_real_disagreement_heatmap(pred_toxic_score,disagreement):
    pred_toxic_score= pred_toxic_score.astype(float)
    disagreement= disagreement.astype(float)
    df = pd.DataFrame({
    'disagreement': disagreement,
    'toxicity_score': pred_toxic_score,
    })
    bins_toxicity = np.arange(0, 1.1, 0.1)
    labels_annotator= [f'{i:.1f}-{i+0.1:.1f}' for i in bins_toxicity[:-1]]
    # Binning annotator counts
    df['annotator_bins'] = pd.cut(df["disagreement"], bins=bins_toxicity, labels=labels_annotator, right=False)

    # Binning toxicity scores into increments of 0.1
    bins_toxicity = np.arange(0, 1.1, 0.1)
    labels_toxicity = [f'{i:.1f}-{i+0.1:.1f}' for i in bins_toxicity[:-1]]
    df['toxicity_bins'] = pd.cut(df['toxicity_score'], bins=bins_toxicity, labels=labels_toxicity, right=False)

    # Group by the bins and count occurrences
    toxicity_distribution = df.groupby(['annotator_bins', 'toxicity_bins']).size().reset_index(name='count')

    # Pivot the table for better visualization
    toxicity_pivot = toxicity_distribution.pivot(index='toxicity_bins', columns='annotator_bins', values='count').fillna(0)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 7),dpi=50)
    sns.heatmap(toxicity_pivot, cmap="YlGnBu", cbar_kws={'label': 'Count'}, annot=True, fmt='g', ax=ax)
    
    # Customize plot
    ax.set_title('Model Distribution of Toxicity Scores for Real Different Disagreement Bins')
    ax.set_xlabel('Disagreement Count Bins')
    ax.set_ylabel('Toxicity Score Bins')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.close(fig)  # Close the figure to avoid displaying it prematurely

    return fig

def tensorboard_model_toxic_pred_pred_disagreement_heatmap(pred_toxic_score,pred_disagreement):
    pred_toxic_score= pred_toxic_score.astype(float)
    pred_disagreement= pred_disagreement.astype(float)
    df = pd.DataFrame({
    'disagreement': pred_disagreement,
    'toxicity_score': pred_toxic_score,
    })
    bins_toxicity = np.arange(0, 1.1, 0.1)
    labels_annotator= [f'{i:.1f}-{i+0.1:.1f}' for i in bins_toxicity[:-1]]
    # Binning annotator counts
    df['annotator_bins'] = pd.cut(df["disagreement"], bins=bins_toxicity, labels=labels_annotator, right=False)

    # Binning toxicity scores into increments of 0.1
    bins_toxicity = np.arange(0, 1.1, 0.1)
    labels_toxicity = [f'{i:.1f}-{i+0.1:.1f}' for i in bins_toxicity[:-1]]
    df['toxicity_bins'] = pd.cut(df['toxicity_score'], bins=bins_toxicity, labels=labels_toxicity, right=False)

    # Group by the bins and count occurrences
    toxicity_distribution = df.groupby(['annotator_bins', 'toxicity_bins']).size().reset_index(name='count')

    # Pivot the table for better visualization
    toxicity_pivot = toxicity_distribution.pivot(index='toxicity_bins', columns='annotator_bins', values='count').fillna(0)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 7),dpi=50)
    sns.heatmap(toxicity_pivot, cmap="YlGnBu", cbar_kws={'label': 'Count'}, annot=True, fmt='g', ax=ax)
    
    # Customize plot
    ax.set_title('Model Distribution of Toxicity Scores for Model Different Disagreement Bins')
    ax.set_xlabel('Disagreement Count Bins')
    ax.set_ylabel('Toxicity Score Bins')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.close(fig)  # Close the figure to avoid displaying it prematurely

    return fig