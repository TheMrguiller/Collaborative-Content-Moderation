import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)).split("src")[0]+"src/")
from src.utils.compute_disagreement import get_distance_labels
from src.utils.store_cp_graphs import store_class_cp_metrics
from src.utils.store_predictions import store_cp_values
import glob
import numpy as np
from src.models.cp import LAC,CCLAC,CRC
from src.data.utils import does_path_exist


def obtain_cp_prediction_set(cp_class,predictions_val,labels_val,predictions_test):
    
    qhat=cp_class.compute_qhat(y_pred=predictions_val,y_true=labels_val)
    if isinstance(cp_class, LAC):
        return cp_class.compute_prediction_set(y_pred=predictions_test,qhat=qhat)
    elif isinstance(cp_class, CCLAC):
        return cp_class.compute_prediction_set(y_pred=predictions_test,qhat_toxic=qhat[0],qhat_non_toxic=qhat[1])
    elif isinstance(cp_class, CRC):
        return cp_class.compute_prediction_set(y_pred=predictions_test,qhat=qhat)
    
def prepare_predictions_to_class(pred):
    return np.stack((1 - pred,pred), axis=1)   
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

if __name__ == "__main__":
    project_path=os.path.dirname(os.path.abspath(__file__)).split("src")[0]+"src/"
    logs_path= project_path+"results/logs/"
    test_result_path= project_path+"results/test_results/"
    val_result_path= project_path+"results/val_results/"
    files = glob.glob(f"{test_result_path}/*")  # Use patterns like "*.txt" for specific types
    print(files)
    cp_result_path= project_path+"results/cp_results/"
    if not does_path_exist(path=cp_result_path):
        os.makedirs(cp_result_path)
    if not does_path_exist(path=cp_result_path+"LAC/"):
        os.makedirs(cp_result_path+"LAC/")
    if not does_path_exist(path=cp_result_path+"CCLAC/"):
        os.makedirs(cp_result_path+"CCLAC/")
    if not does_path_exist(path=cp_result_path+"CRC/"):
        os.makedirs(cp_result_path+"CRC/")
        
    for file_path in files:
        print(file_path)
        filename=file_path.rsplit("/")[-1].split(".npz")[0]
        if "regression" in filename:
            continue
        test_results=np.load(file_path)
        val_results=np.load(f"{val_result_path}{filename}.npz")
        # Base CP
        logs_path= project_path+"results/logs/"+filename+"/"
        log_dir=get_last_created_folder(logs_path)
        if log_dir is None:
            continue
        log_dir=log_dir+"/"
        distance_disagreement=get_distance_labels(test_results["labels"][:,0])
        y_pred_val= prepare_predictions_to_class(val_results["predictions"][:, 0])
        y_true_val = prepare_predictions_to_class((val_results["labels"][:, 0]>=0.5).astype(int))
        y_pred_test = prepare_predictions_to_class(test_results["predictions"][:, 0])
        for cp_method in [LAC(),CCLAC(),CRC()]:
            prediction_set=obtain_cp_prediction_set(cp_class=cp_method,predictions_val=y_pred_val,labels_val=y_true_val,predictions_test=y_pred_test)
            uncertainty_cp=[1 if (prediction.all().item() or not prediction.any().item()) else 0 for prediction in prediction_set]
            if isinstance(cp_method, LAC):
                store_class_cp_metrics(log_dir=log_dir,method_name="LAC",prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"])
                store_cp_values(prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"],log_dir=cp_result_path+"LAC/"+filename)
            if isinstance(cp_method, CCLAC):
                store_class_cp_metrics(log_dir=log_dir,method_name="CCLAC",prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"])
                store_cp_values(prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"],log_dir=cp_result_path+"CCLAC/"+filename)
            if isinstance(cp_method, CRC):
                store_class_cp_metrics(log_dir=log_dir,method_name="CRC",prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"])
                store_cp_values(prediction_set=prediction_set,labels=test_results["labels"],predictions=test_results["predictions"],log_dir=cp_result_path+"CRC/"+filename)