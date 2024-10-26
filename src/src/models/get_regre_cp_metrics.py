import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split("src")[0]+"src/")
from src.utils.compute_disagreement import get_distance_labels
from src.utils.store_predictions import store_cp_values
import glob
import numpy as np
from src.models.cp import AbsoluteResidual,Gamma,RACCP,RN
from src.data.utils import does_path_exist
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
from src.models.model import TransformersForSequenceClassificationMultitask,TransformersRegression,TransformersForSequenceClassificationMultitaskRAC,TransformersRegressionRAC
import gc
from src.utils.store_cp_graphs import store_value
from src.models.metrics import coverage_probability,DI_metric,interval_width,interval_width_correlation_disagreement
import torch
from tensorboardX import SummaryWriter
import logging
from tqdm import tqdm

# Set up logging to console
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.StreamHandler()  # Log to the console
    ]
)
# Create a logger
logger = logging.getLogger("MyApp")
def get_last_created_folder(directory):
    # List all entries in the directory with full paths
    all_entries = [os.path.join(directory, entry) for entry in os.listdir(directory)]
    
    # Filter only directories
    directories = [entry for entry in all_entries if os.path.isdir(entry)]
    
    if not directories:
        return None  # If there are no directories, return None
    
    # Sort directories by creation time, most recent last
    directories.sort(key=os.path.getctime, reverse=True)
    
    # Return the most recent directory
    return directories[0]

def kNNCP(filename,project_path,cp_method,val_distance_disagreement,val_predictions,test_predictions):
    
    dataset = load_dataset("TheMrguiller/Uncertainty_Toxicity")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    train_text=dataset["train"]["comment_text"]
    distance_disagreement=get_distance_labels(dataset["train"]["toxicity"])
    val_text=dataset["valid"]["comment_text"]
    test_text=dataset["test"]["comment_text"]
    if "regression" in filename and "R2ccpLoss" in filename:
        model = TransformersRegressionRAC.load_from_checkpoint(project_path+"models/checkpoints/"+filename+".ckpt")
    elif "regression" in filename:
        model = TransformersRegression.load_from_checkpoint(project_path+"models/checkpoints/"+filename+".ckpt")
    elif "R2ccpLoss" in filename and "multitask" in filename:
        model = TransformersForSequenceClassificationMultitaskRAC.load_from_checkpoint(project_path+"models/checkpoints/"+filename+".ckpt")
    elif "multitask" in filename:
        model=TransformersForSequenceClassificationMultitask.load_from_checkpoint(project_path+"models/checkpoints/"+filename+".ckpt")
    del dataset
    batch_size = 64
    train_predictions = []
    logger.debug("Computing train predictions")
    for i in tqdm(range(0, len(train_text), batch_size)):
        batch = train_text[i:i + batch_size]
        batch_predictions = model.predict(batch).cpu().numpy()
        train_predictions.extend(batch_predictions.tolist())
    train_predictions = np.array(train_predictions)
    if "regression" in filename and "R2ccpLoss" in filename:
        midpoints_step_division = np.linspace(0, 1, train_predictions.shape[1]+1)
        midpoints = (midpoints_step_division[:-1] + midpoints_step_division[1:]) / 2
        train_predictions=np.array([midpoints[class_index] for class_index in np.argmax(train_predictions,axis=1)])
    elif "regression" in filename:
        train_predictions=train_predictions[:,0]
    elif "R2ccpLoss" in filename and "multitask" in filename:
        train_predictions=train_predictions[:,1:]
        midpoints_step_division = np.linspace(0, 1, train_predictions.shape[1]+1)
        midpoints = (midpoints_step_division[:-1] + midpoints_step_division[1:]) / 2
        train_predictions=np.array([midpoints[class_index] for class_index in np.argmax(train_predictions,axis=1)])
    elif "multitask" in filename:
        train_predictions=train_predictions[:,1]
    del model
    gc.collect()
    logger.debug("Training KNN")
    cp_method.train_KNN(texts=train_text,labels=distance_disagreement,predictions=train_predictions)
    if isinstance(cp_method,RN):
        logger.debug("Computing cal scores")
        qhat=cp_method.compute_qhat(text=val_text,y_true=val_distance_disagreement,y_pred=val_predictions)
        logger.debug("Computing prediction set")
        prediction_set=cp_method.compute_prediction_set(text=test_text,y_pred=test_predictions,qhat=qhat)
    return prediction_set

def store_regression_metrics(log_dir,method_name,prediction_set,test_label):
    writer = SummaryWriter(log_dir)
    logger.debug("Storing metrics")
    logger.debug(f"Prediction set shape: {prediction_set.shape}, Test label shape: {test_label.shape}")
    icp_metric=coverage_probability(true_values=torch.tensor(test_label[:,1]),lower_bounds=torch.tensor(prediction_set[:,0]),upper_bounds=torch.tensor(prediction_set[:,1]))
    distance_to_interval_metric=DI_metric(true_values=torch.tensor(test_label[:,1]),lower_bounds=torch.tensor(prediction_set[:,0]),upper_bounds=torch.tensor(prediction_set[:,1]))
    mean_width, median_width, quantile_25, quantile_75=interval_width(lower_bounds=torch.tensor(prediction_set)[:,0].float(),upper_bounds=torch.tensor(prediction_set)[:,1].float())
    correlation_interval_size_disagreement=interval_width_correlation_disagreement(lower_bounds=torch.tensor(prediction_set[:,0]),upper_bounds=torch.tensor(prediction_set[:,1]),disagreement=torch.tensor(test_label[:,1]))
    store_value(icp_metric,writer,method_name+" ICP")
    store_value(distance_to_interval_metric,writer,method_name+" DI")
    store_value(mean_width,writer,method_name+" Mean Interval Size")
    store_value(median_width,writer,method_name+" Median Interval Size")
    store_value(quantile_25,writer,method_name+" 25th Quantile Interval Size")
    store_value(quantile_75,writer,method_name+" 75th Quantile Interval Size")
    store_value(correlation_interval_size_disagreement,writer,method_name+" Correlation Interval Size Disagreement")
    writer.close()

def sort_key(s):
    # Check if the string contains "RCPP2logloss"
    if "R2ccpLoss" in s:
        # If it does, return a tuple with a low priority value and the original string
        return (0, s)
    else:
        # If it doesn't, return a tuple with a high priority value and the original string
        return (1, s)

if __name__ == "__main__":

    alpha = 0.1
    model_name="distilbert/distilbert-base-uncased"
    login(token =os.environ['HUGGINGFACE_TOKEN'])
    project_path=os.path.dirname(os.path.abspath(__file__)).split("src")[0]+"src/"
    logs_path= project_path+"results/logs/"
    test_result_path= project_path+"results/test_results/"
    val_result_path= project_path+"results/val_results/"
    files = glob.glob(f"{test_result_path}/*")  # Use patterns like "*.txt" for specific types
    cp_result_path= project_path+"results/cp_results/"
    if not does_path_exist(path=cp_result_path):
        os.makedirs(cp_result_path)
    for cp_class in ["AbsoluteResidual","Gamma","RACCP","RN"]:
        if not does_path_exist(path=cp_result_path+cp_class+"/"):
            os.makedirs(cp_result_path+cp_class+"/")
    
    # files = sorted(files, key=sort_key)
    
    for file_path in files:
        filename=file_path.rsplit("/")[-1].split(".npz")[0]
        logger.debug(filename)
        if "classification" not in filename:
            test_results=np.load(file_path)
            val_results=np.load(f"{val_result_path}{filename}.npz")

            if "regression" in filename and "R2ccpLoss" in filename:
                midpoints_step_division = np.linspace(0, 1, test_results["predictions"].shape[1]+1)
                midpoints = (midpoints_step_division[:-1] + midpoints_step_division[1:]) / 2
                val_pred=np.array([midpoints[class_index] for class_index in np.argmax(val_results["predictions"],axis=1)])
                test_pred=np.array([midpoints[class_index] for class_index in np.argmax(test_results["predictions"],axis=1)])     
                val_labels=val_results["labels"][:,1]
                test_labels=test_results["labels"][:,1]
            elif "regression" in filename:
                test_pred = test_results["predictions"][:,0]
                test_labels = test_results["labels"][:,1]
                val_pred = val_results["predictions"][:,0]
                val_labels = val_results["labels"][:,1]
            elif "multitask" in filename and "R2ccpLoss" in filename:
                midpoints_step_division = np.linspace(0, 1, test_results["predictions"][:,1:].shape[1]+1)
                midpoints = (midpoints_step_division[:-1] + midpoints_step_division[1:]) / 2
                val_pred=np.array([midpoints[class_index] for class_index in np.argmax(val_results["predictions"][:,1:],axis=1)])
                test_pred=np.array([midpoints[class_index] for class_index in np.argmax(test_results["predictions"][:,1:],axis=1)])     
                val_labels=val_results["labels"][:,1]
                test_labels=test_results["labels"][:,1]
            elif "multitask" in filename:
                test_pred = test_results["predictions"][:,1]
                test_labels = test_results["labels"][:,1]
                val_pred = val_results["predictions"][:,1]
                val_labels = val_results["labels"][:,1]
            
            logs_path= project_path+"results/logs/"+filename+"/"
            log_dir=get_last_created_folder(logs_path)
            if log_dir is None:
                continue
            log_dir=log_dir+"/"
            for cp_method in [AbsoluteResidual(alpha=alpha),Gamma(alpha=alpha),RACCP(num_classes=20,alpha=alpha)]:
                
                if isinstance(cp_method,RACCP) and "R2ccpLoss" in filename:
                    if "multitask" in filename:
                        cp_method_= RACCP(num_classes=test_results["predictions"][:,1:].shape[1],alpha=alpha)
                        prediction_set=cp_method_.get_intervals(pred_scores=test_results["predictions"][:,1:],cal_pred=val_results["predictions"][:,1:],y_cal=val_results["labels"][:,1])
                    else:
                        cp_method_= RACCP(num_classes=test_results["predictions"].shape[1],alpha=alpha)
                        prediction_set=cp_method_.get_intervals(pred_scores=test_results["predictions"],cal_pred=val_results["predictions"],y_cal=val_results["labels"][:,1])

                else:
                    if isinstance(cp_method,AbsoluteResidual):
                        qhat=cp_method.compute_qhat(y_pred=val_pred,y_true=val_labels)
                        prediction_set=cp_method.compute_prediction_set(y_pred=test_pred,qhat=qhat).t()
                    elif isinstance(cp_method,Gamma):
                        qhat=cp_method.compute_qhat(y_pred=val_pred,y_true=val_labels)
                        prediction_set=cp_method.compute_prediction_set(y_pred=test_pred,qhat=qhat)
                    elif isinstance(cp_method,RN):
                        prediction_set=kNNCP(filename=filename,project_path=project_path,cp_method=cp_method,val_distance_disagreement=val_labels,val_predictions=val_pred,test_predictions=test_pred)
            
                if isinstance(cp_method,AbsoluteResidual):
                    logger.debug("AbsoluteResidual")
                    store_regression_metrics(log_dir=log_dir,method_name="AbsoluteResidual",prediction_set=prediction_set,test_label=test_results["labels"])
                    store_cp_values(prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"],log_dir=cp_result_path+"AbsoluteResidual/"+filename)
                elif isinstance(cp_method,Gamma):
                    logger.debug("Gamma")
                    store_regression_metrics(log_dir=log_dir,method_name="Gamma",prediction_set=prediction_set,test_label=test_results["labels"])
                    store_cp_values(prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"],log_dir=cp_result_path+"Gamma/"+filename)
                elif isinstance(cp_method,RACCP) and "R2ccpLoss" in filename:
                    logger.debug("RACCP")
                    store_regression_metrics(log_dir=log_dir,method_name="RACCP",prediction_set=prediction_set,test_label=test_results["labels"])
                    store_cp_values(prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"],log_dir=cp_result_path+"RACCP/"+filename)
                elif isinstance(cp_method,RN):
                    logger.debug("RN")
                    store_regression_metrics(log_dir=log_dir,method_name="RN",prediction_set=prediction_set,test_label=test_results["labels"])
                    store_cp_values(prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"],log_dir=cp_result_path+"RN/"+filename)
        elif "classification" in filename:
            continue
