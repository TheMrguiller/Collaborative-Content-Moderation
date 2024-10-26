import glob
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split("src")[0]+"src/")
import numpy as np
from sklearn.metrics import f1_score as f1,precision_score,recall_score
from src.models.metrics import uncertainty_correlation_prediction_set,uncertainty_confusion_matrix,uncertainty_base_metrics
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize_scalar,minimize
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)

def compute_joint_classification_regression_cp_prediction_metrics(threshold:int,class_prediction_set:np.ndarray,regresion_prediction_set:np.ndarray,class_pred:np.ndarray,class_labels:np.ndarray,regre_labels:np.ndarray):
    
    regression_prediction_set_upper_bound = regresion_prediction_set[:,1]
    mask = (regression_prediction_set_upper_bound >= threshold).astype(int)
    uncertainty_cp= np.array([1 if (prediction.all() or not prediction.any()) else 0 for prediction in class_prediction_set])
    filter_condition = (uncertainty_cp == 0) & (mask == 0)
    y_pred = class_prediction_set[filter_condition].argmax(axis=1)
    y_true = (class_labels[filter_condition]>=0.5).astype(int)
    uncertainty_method= (uncertainty_cp == 1) | (mask == 1) 
    correlation = uncertainty_correlation_prediction_set(uncertainty_method,regre_labels)
    f1_score = f1(y_true,y_pred)
    CARE=recall_score((regre_labels>=threshold).astype(int),((mask==1) | (uncertainty_cp == 1)))
    return f1_score,CARE,correlation

def calculate_base_model_cp_metrics(test_class_base,threshold,regre_labels:np.ndarray):

    uncertainty_method= np.array([1 if (prediction.all() or not prediction.any()) else 0 for prediction in test_class_base["prediction_set"]])
    class_labels=test_class_base["labels"][:,0]
    class_pred = test_class_base["predictions"][:,0]
    CARE=recall_score((regre_labels>=threshold).astype(int),uncertainty_method)
    correlation = uncertainty_correlation_prediction_set(uncertainty_method,test_class_base["labels"][:,1])
    # print(f"Correlation: {correlation}")
    filter_condition = (uncertainty_method == 0)
    y_pred = test_class_base["prediction_set"][filter_condition].argmax(axis=1)
    y_true = (test_class_base["labels"][:,0][filter_condition]>=0.5).astype(int)
    f1_score = f1(y_true,y_pred)
    # print(f"F1_score Uncertainty:{f1_score}")
    return f1_score,CARE,correlation

def tpr_at_threshold(threshold, class_prediction_set, regresion_prediction_set,regression_label,type_task="multitask"):
    # Apply threshold to get binary predictions
    uncertainty_cp= np.array([1 if (prediction.all() or not prediction.any()) else 0 for prediction in class_prediction_set])
    if type_task=="multitask":
    
        regression_prediction_set_upper_bound = regresion_prediction_set[:,1]
        mask = (regression_prediction_set_upper_bound >= threshold).astype(int)
        y_pred=( (mask==1 ) | (uncertainty_cp==1))
    elif type_task=="classification":
        y_pred= uncertainty_cp
    y_true= (regression_label >= threshold).astype(int)
    # Calculate confusion matrix to get TP, FN
    if y_pred.sum()==0 or y_true.sum()==0 or y_pred.sum()==len(y_pred) or y_true.sum()==len(y_true):
        return 0,0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate TPR: TPR = TP / (TP + FN)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr,fpr

# Objective function to minimize (difference between TPR and target TPR)
def objective(threshold, class_prediction_set, regresion_prediction_set,regression_label,type_task, target_tpr=0.95):
    tpr,_ = tpr_at_threshold(threshold, class_prediction_set, regresion_prediction_set,regression_label,type_task)
    return abs(tpr - target_tpr)  # Minimize the absolute difference

def calculate_threshold(class_prediction_set, regresion_prediction_set, regression_label, type_task, target_tpr=0.95):
    thresholds = np.linspace(0, 1, 300 + 1)
    
    with ThreadPoolExecutor() as executor:
        # Launch parallel calculations
        tpr_list = list(executor.map(lambda t: tpr_at_threshold([t], class_prediction_set, regresion_prediction_set, regression_label, type_task), thresholds))

    # Find the closest TPR to the target
    closest_tpr = closest_value_numpy(thresholds, target_tpr)
    return closest_tpr

def closest_value_numpy(value_list, target):
    array = np.array(value_list)
    index = (np.abs(array - target)).argmin()
    return array[index]   
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    project_path=os.path.dirname(os.path.abspath("__file__")).split("src")[0]+"src/"
    try:
        data_mure = pd.read_excel(project_path+"results/classification_metrics.xlsx")
    except:
        raise Exception("classification_metrics.xlsx not found")
    logs_path= project_path+"results/logs/"
    cp_result_path= project_path+"results/cp_results/"
    regression_cp_methods=["AbsoluteResidual","Gamma","RACCP","RN"]
    classification_cp_methods=["LAC","CCLAC","CRC"]
    df_store_last_metrics = pd.DataFrame(columns=["model","cp_method","threshold","fpr_multitask","f1_score_cp_multitask","MURE_multitask","CARE_multitask","correlation_multitask","fpr_base","f1_score_cp_base","MURE_base","CARE_base","correlation_base","fpr_monotask","f1_score_cp_monotask","MURE_monotask","CARE_monotask","correlation_monotask"])
    logger.debug("Starting to calculate metrics")
    for classification_cp_method in classification_cp_methods:
        logger.debug(f"Classification CP Method: {classification_cp_method}")
        files_class = sorted(glob.glob(cp_result_path + classification_cp_method + "/*"),reverse=True)# Use patterns like "*.txt" for specific types
        for regression_cp_method in regression_cp_methods:
            logger.debug(f"Regression CP Method: {regression_cp_method}")
            for file_class in files_class:
                filename = file_class.rsplit("/")[-1]
                
                MURE_metric_multitask = data_mure[data_mure["Unnamed: 0"]==model_name][classification_cp_method+"_MURE"].values[0]
                classification_method_name= filename.split(".npz")[0].split("_distance_")[0].split("multitask_")[1]
                regression_method_name= filename.split(".npz")[0].split(classification_method_name+"_")[1]
                files_regre = [f"{cp_result_path}{regression_cp_method}/{filename}"]
                logger.debug(f"Filename: {filename}")
                print(f"Filename: {filename}")
                for file_regre in files_regre:
                    if "RACCP" in regression_cp_method and "R2ccpLoss" not in filename:
                        continue
                    test_class_cp_results = np.load(file_class)
                    test_regre_cp_results = np.load(file_regre)
                    
                    optimal_threshold_before=calculate_threshold(test_class_cp_results["prediction_set"], test_regre_cp_results["prediction_set"],test_regre_cp_results["labels"][:,1],"multitask")
                    initial_threshold=np.array([optimal_threshold_before])
                    result = minimize(objective, initial_threshold, args=( test_class_cp_results["prediction_set"], test_regre_cp_results["prediction_set"],test_regre_cp_results["labels"][:,1],"multitask"), method='Nelder-Mead',bounds=[(0, 1)])
                    optimal_threshold = result.x[0]
                    if abs(0.95-tpr_at_threshold(optimal_threshold_before, test_class_cp_results["prediction_set"], test_regre_cp_results["prediction_set"],test_regre_cp_results["labels"][:,1],"multitask")[0])<abs(0.95-tpr_at_threshold(optimal_threshold, test_class_cp_results["prediction_set"], test_regre_cp_results["prediction_set"],test_regre_cp_results["labels"][:,1],"multitask")[0]):
                        optimal_threshold=optimal_threshold_before
                    f1_score_multitask,CARE_multitask,correlation_multitask=compute_joint_classification_regression_cp_prediction_metrics(threshold=optimal_threshold,class_prediction_set=test_class_cp_results["prediction_set"],regresion_prediction_set=test_regre_cp_results["prediction_set"],class_pred=test_class_cp_results["predictions"][:,0],class_labels=test_class_cp_results["labels"][:,0],regre_labels=test_regre_cp_results["labels"][:,1])
                    _,fpr_multitask = tpr_at_threshold(optimal_threshold, test_class_cp_results["prediction_set"], test_regre_cp_results["prediction_set"],test_regre_cp_results["labels"][:,1],"multitask")
                    logger.debug(f"Multitask Threshold 75% uncertain: {optimal_threshold},FPR: {fpr_multitask} F1_score: {f1_score_multitask}, CARE: {CARE_multitask}, Correlation: {correlation_multitask}")
                    # print(f"Multitask Threshold 75% uncertain: {optimal_threshold},FPR: {fpr_multitask} F1_score: {f1_score_multitask}, Ambiguity Review Efficency: {review_eficciency_multitask}, Correlation: {correlation_multitask}")
                    
                    test_class_base_cp_results = np.load(f"{cp_result_path}/{classification_cp_method}/classification_focal_loss_Weighted.npz")
                    MURE_metric_base = data_mure[data_mure["Unnamed: 0"]=="classification_focal_loss_Weighted"][classification_cp_method+"_MURE"].values[0]
                    
                    
                    result = minimize(objective, initial_threshold, args=( test_class_base_cp_results["prediction_set"], test_regre_cp_results["prediction_set"],test_regre_cp_results["labels"][:,1],"classification"), method='Nelder-Mead',bounds=[(0, 1)])
                    optimal_threshold = result.x[0]
                    _,fpr_base = tpr_at_threshold(optimal_threshold, test_class_base_cp_results["prediction_set"], test_regre_cp_results["prediction_set"],test_regre_cp_results["labels"][:,1],"classification")
                    f1_score_base,CARE_base,correlation_base=calculate_base_model_cp_metrics(test_class_base=test_class_base_cp_results,threshold=optimal_threshold,regre_labels=test_regre_cp_results["labels"][:,1])
                    logger.debug(f"Base Model Threshold 75% uncertain: {optimal_threshold}, FPR: {fpr_base} F1_score: {f1_score_base}, CARE: {CARE_base}, Correlation: {correlation_base}")
                    # print(f"Base Model Threshold 75% uncertain: {optimal_threshold}, FPR: {fpr_base} F1_score: {f1_score_base}, Ambiguity Review Efficency: {review_eficciency_base}, Correlation: {correlation_base}")   
                    
                    test_class_regre_cp_results = np.load(f"{cp_result_path}{regression_cp_method}/regression_{regression_method_name}.npz")
                   
                    optimal_threshold_before=calculate_threshold(test_class_base_cp_results["prediction_set"], test_class_regre_cp_results["prediction_set"],test_class_regre_cp_results["labels"][:,1],"multitask")
                    initial_threshold=np.array([optimal_threshold_before])
                    result = minimize(objective, initial_threshold, args=( test_class_base_cp_results["prediction_set"], test_class_regre_cp_results["prediction_set"],test_class_regre_cp_results["labels"][:,1],"multitask"), method='Nelder-Mead',bounds=[(0, 1)])
                    optimal_threshold = result.x[0]
                    if abs(0.95-tpr_at_threshold(optimal_threshold_before, test_class_base_cp_results["prediction_set"], test_class_regre_cp_results["prediction_set"],test_class_regre_cp_results["labels"][:,1],"multitask")[0])<abs(0.95-tpr_at_threshold(optimal_threshold, test_class_base_cp_results["prediction_set"], test_class_regre_cp_results["prediction_set"],test_class_regre_cp_results["labels"][:,1],"multitask")[0]):
                        optimal_threshold=optimal_threshold_before
                    _,fpr_monotask = tpr_at_threshold(optimal_threshold, test_class_base_cp_results["prediction_set"], test_class_regre_cp_results["prediction_set"],test_class_regre_cp_results["labels"][:,1],"multitask")
                    f1_score_monotask,CARE_monotask,correlation_monotask=compute_joint_classification_regression_cp_prediction_metrics(threshold=optimal_threshold,class_prediction_set=test_class_base_cp_results["prediction_set"],regresion_prediction_set=test_class_regre_cp_results["prediction_set"],class_pred=test_class_base_cp_results["predictions"][:,0],class_labels=test_class_base_cp_results["labels"][:,0],regre_labels=test_class_regre_cp_results["labels"][:,1])
                    logger.debug(f"Monotask Threshold 75% uncertain: {optimal_threshold},FPR: {fpr_monotask} F1_score: {f1_score_monotask}, CARE: {CARE_monotask}, Correlation: {correlation_monotask}")
                    # print(f"Monotask Threshold 75% uncertain: {optimal_threshold},FPR: {fpr_monotask} F1_score: {f1_score_monotask}, Ambiguity Review Efficency: {review_eficciency_monotask}, Correlation: {correlation_monotask}")
                    
                    df_store_last_metrics.loc[len(df_store_last_metrics)] = [filename.split(".npz")[0],classification_cp_method+"+"+regression_cp_method,optimal_threshold,fpr_multitask,f1_score_multitask,MURE_metric_multitask,CARE_multitask,correlation_multitask,fpr_base,f1_score_base,MURE_metric_base,CARE_base,correlation_base,fpr_monotask,f1_score_monotask,MURE_metric_base,CARE_monotask,correlation_monotask]
                
                    pass
    df_store_last_metrics.to_excel(f"{cp_result_path}/framework_results_metrics.xlsx", index=False)

    