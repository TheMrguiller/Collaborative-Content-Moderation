from pytorch_lightning import Trainer
import torch
import numpy as np
from src.data.utils import does_path_exist
import os
from src.models.metrics import disagreement_bin_uncertainty_base_metrics
from src.utils.compute_disagreement import get_distance_labels

def generate_prediction_classification(trainer:Trainer):
    logits=trainer.model.logits
    labels=trainer.model.labels
    logits_value=[]
    for logit in logits:
        for element in logit:
            logits_value.append(element)
    prediction_model=torch.nn.functional.sigmoid(torch.stack(logits_value))
    labels_value=[]
    for label in labels:
        for element in label:
            labels_value.append(element)
    true_labels_value=torch.stack(labels_value)
    return prediction_model.numpy(),true_labels_value.numpy()

def generate_prediction_classification_RAC(trainer:Trainer,type_model:str):
    logits=trainer.model.logits
    labels=trainer.model.labels
    logits_value=[]
    for logit in logits:
        for element in logit:
            logits_value.append(element)
    print(logits_value)
    print(torch.stack(logits_value).shape)

    if type_model=="multitask":
        prediction_model_class=torch.nn.functional.sigmoid(torch.stack(logits_value)[:,0])
        prediction_model_regression=torch.nn.functional.softmax(torch.stack(logits_value)[:,1:])
    elif type_model=="single":
        prediction_model_regression=torch.nn.functional.softmax(torch.stack(logits_value))
    labels_value=[]
    for label in labels:
        for element in label:
            labels_value.append(element)
    true_labels_value=torch.stack(labels_value)
    if type_model=="multitask":
        prediction_model = torch.cat((prediction_model_class.unsqueeze(1),prediction_model_regression),1)
    elif type_model=="single":
        prediction_model = prediction_model_regression
    return prediction_model.numpy(),true_labels_value.numpy()


def store_predictions(pred,labels,model_name,proyect_path,test_type="test"):
    
    if test_type=="test":
        if not does_path_exist(path=f"{proyect_path}results/test_results"):
            path=f"{proyect_path}results/test_results"
            os.makedirs(f"{proyect_path}/src/results/test_results")
        else:
            path=f"{proyect_path}results/test_results"
    elif test_type=="validation":
        if not does_path_exist(path=f"{proyect_path}results/val_results"):
            path=f"{proyect_path}results/val_results"
            os.makedirs(f"{proyect_path}/src/results/val_results")
        else:
            path=f"{proyect_path}results/val_results"
    
    np.savez(path+"/"+model_name+".npz", predictions=pred, labels=labels)
    
def store_cp_values(log_dir: str,prediction_set:np.ndarray,labels:np.ndarray,predictions:np.ndarray):
    
    distance_disagreement=get_distance_labels(labels[:,0])
    uncertainty_cp=[1 if (prediction.all().item() or not prediction.any().item()) else 0 for prediction in prediction_set]
    disagreement_bin_uncertainty_metrics=disagreement_bin_uncertainty_base_metrics(distance_disagreement=distance_disagreement,predictions=predictions,labels=labels,uncertainty=uncertainty_cp)
    np.savez(log_dir+".npz", prediction_set=prediction_set, labels=labels,predictions=predictions,uncertainty_cp=uncertainty_cp,distance_disagreement=distance_disagreement,disagreement_bin_uncertainty_metrics=disagreement_bin_uncertainty_metrics)
    