import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("__file__")).split("src")[0]+"src/")

from src.models.model import TransformersForSequenceClassification,TransformersForSequenceClassificationMultitask,TransformersForSequenceClassificationMultiOutput,TransformersRegression,TransformersForSequenceClassificationMultitaskRAC,TransformersRegressionRAC
from src.models.dataset import JigsawUnintendedDataModule
import argparse
import yaml
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,LearningRateMonitor

from pytorch_lightning.loggers import TensorBoardLogger
from datasets import load_dataset
from huggingface_hub import login
import torch
import random
import numpy as np
from src.utils.store_calibration_graphs import log_base_multitask_tensorboard,log_base_class_tensorboard,log_base_regression_tensorboard
from src.utils.store_predictions import generate_prediction_classification,store_predictions,generate_prediction_classification_RAC
from src.data.utils import does_path_exist,determine_soft_labels_need,seed_everything
import gc


ARGUMENTS_TO_CHECK = ["model_name", "task", "disagreement","learning_rate",
    "experiment_name","data_batch_size","weight_decay","num_nodes",
    "accumulate_grad_batches","num_workers","regression_name"]

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train_procedure_classification(task,model_name,config,experiment_name,project_path,data_module,fast_dev_run=False):
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    earlystopping_callback=EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min"
    )
    
    model = TransformersForSequenceClassification(
        model_name=model_name,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        experiment_name=experiment_name,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=project_path+"models/checkpoints",
        filename=f"{task}_{experiment_name}",
        verbose=True,
        save_on_train_epoch_end=False,
        save_top_k=1,
        mode="min",
    )
    log_dir=project_path+"results/logs"
    logger = TensorBoardLogger(log_dir, name=f"{task}_{experiment_name}")
    # Load the trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=100,
        precision="16-mixed",
        logger=logger,
        min_epochs=2,
        num_nodes=config["num_nodes"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback,earlystopping_callback,lr_monitor],
        fast_dev_run=fast_dev_run
    )
    data_module.task=task
    data_module.disagreement="distance"
    print(f"Training the model with the following characteristics:\nmodel_name:{model_name}\nTask:{data_module.task}\nExperiment name: {experiment_name}")

    # Train the model
    trainer.fit(model, data_module)
    ## Store the values for further use in CP
    trainer.validate(datamodule=data_module)
    class_pred,class_true=generate_prediction_classification(trainer)
    store_predictions(pred=class_pred,labels=class_true,model_name=f"{task}_{experiment_name}",proyect_path=project_path,test_type="validation")
    log_base_class_tensorboard(trainer=trainer,log_dir=logger.log_dir,pred=class_pred.squeeze(),labels=class_true)
    
    trainer.model.on_validation_epoch_start()
    trainer.test(datamodule=data_module)
    class_pred,class_true=generate_prediction_classification(trainer)
    store_predictions(pred=class_pred,labels=class_true,model_name=f"{task}_{experiment_name}",proyect_path=project_path,test_type="test")
    log_base_class_tensorboard(trainer=trainer,log_dir=logger.log_dir,pred=class_pred.squeeze(),labels=class_true)
    trainer.model.on_test_epoch_start()
    del model
    
    del trainer

def train_procedure_regression(task,model_name,config,disagreement,regression_name,project_path,data_module,fast_dev_run=False):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    earlystopping_callback=EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min"
    )
    if "R2ccpLoss" in regression_name:
        model = TransformersRegressionRAC(
            model_name=model_name,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            regression_name=regression_name
        )
    else:
        model = TransformersRegression(
            model_name=model_name,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            regression_name=regression_name
        )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=project_path+"models/checkpoints",
        filename=f"{task}_{disagreement}_{regression_name}",
        verbose=True,
        save_on_train_epoch_end=False,
        save_top_k=1,
        mode="min",
    )
    log_dir=project_path+"results/logs"
    logger = TensorBoardLogger(log_dir, name=f"{task}_{disagreement}_{regression_name}")
    # Load the trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=100,
        precision="16-mixed",
        logger=logger,
        min_epochs=1,
        num_nodes=config["num_nodes"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback,earlystopping_callback,lr_monitor],
        fast_dev_run=fast_dev_run
    )
    data_module.task=task
    data_module.disagreement=disagreement
    print(f"Training the model with the following characteristics:\nmodel_name:{model_name}\nTask:{data_module.task}\nDisagreement method: {data_module.disagreement}\n Regression method:{regression_name}")

    # Train the model
    trainer.fit(model, data_module)
    trainer.validate(datamodule=data_module)
    if "R2ccpLoss" in regression_name:
        class_pred,class_true=generate_prediction_classification_RAC(trainer,type_model="single")
        regression_proba= class_pred
        regression_pred=np.array([trainer.model.midpoints[class_index] for class_index in np.argmax(class_pred,axis=1)])
        class_pred = regression_pred
        store_predictions(pred=regression_proba,labels=class_true,model_name=f"{task}_{disagreement}_{regression_name}",proyect_path=project_path,test_type="validation")

    else: 
        class_pred,class_true=generate_prediction_classification(trainer)
        store_predictions(pred=class_pred,labels=class_true,model_name=f"{task}_{disagreement}_{regression_name}",proyect_path=project_path,test_type="validation")
    log_base_regression_tensorboard(trainer=trainer,log_dir=logger.log_dir,pred=class_pred.squeeze(),labels=class_true)
    trainer.model.on_validation_epoch_start()
    trainer.test(datamodule=data_module)
    
    if "R2ccpLoss" in regression_name:
        class_pred,class_true=generate_prediction_classification_RAC(trainer,type_model="single")

        regression_proba= class_pred
        regression_pred=np.array([trainer.model.midpoints[class_index] for class_index in np.argmax(class_pred,axis=1)])

        class_pred = regression_pred
        
        store_predictions(pred=regression_proba,labels=class_true,model_name=f"{task}_{disagreement}_{regression_name}",proyect_path=project_path,test_type="test")

    else: 
        class_pred,class_true=generate_prediction_classification(trainer)
        store_predictions(pred=class_pred,labels=class_true,model_name=f"{task}_{disagreement}_{regression_name}",proyect_path=project_path,test_type="test")
    
    log_base_regression_tensorboard(trainer=trainer,log_dir=logger.log_dir,pred=class_pred.squeeze(),labels=class_true)
    trainer.model.on_test_epoch_start()
    del model
    
    del trainer
def train_procedure_multi(task,model_name,config,disagreement,regression_name,experiment_name,project_path,data_module,fast_dev_run=False):
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    earlystopping_callback=EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min"
    )
    if task == "multitask":
        if "R2ccpLoss" in regression_name:
            model = TransformersForSequenceClassificationMultitaskRAC(
                model_name=model_name,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                experiment_name=experiment_name,
                regression_name=regression_name
            )
        else:
            model = TransformersForSequenceClassificationMultitask(
                model_name=model_name,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                experiment_name=experiment_name,
                regression_name=regression_name
            )
    elif task == "multioutput":
        if "R2ccpLoss" in regression_name:
            model = TransformersForSequenceClassificationMultitaskRAC(
                model_name=model_name,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                experiment_name=experiment_name,
                regression_name=regression_name
            )
        else:
            model = TransformersForSequenceClassificationMultiOutput(
                model_name=model_name,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                experiment_name=experiment_name,
                regression_name=regression_name
            )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=project_path+"models/checkpoints",
        filename=f"{task}_{experiment_name}_{disagreement}_{regression_name}",
        verbose=True,
        save_on_train_epoch_end=False,
        save_top_k=1,
        mode="min",
    )
    log_dir=project_path+"results/logs"
    logger = TensorBoardLogger(log_dir, name=f"{task}_{experiment_name}_{disagreement}_{regression_name}")
    # Load the trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=100,
        precision="16-mixed",
        logger=logger,
        min_epochs=2,
        num_nodes=config["num_nodes"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback,earlystopping_callback,lr_monitor],
        fast_dev_run=fast_dev_run
    )
    data_module.task=task
    data_module.disagreement=disagreement
    print(f"Training the model with the following characteristics:\nmodel_name:{model_name}\nTask:{data_module.task}\nDisagreement method: {data_module.disagreement}\nExperiment name: {experiment_name}\n Regression method:{regression_name}")

    # Train the model
    trainer.fit(model, data_module)
    trainer.validate(datamodule=data_module)


    if "R2ccpLoss" in regression_name:
        class_pred,class_true=generate_prediction_classification_RAC(trainer,type_model="multitask")
        classification_pred = class_pred[:,0]
        regression_proba= class_pred[:,1:]
        regression_pred=np.array([trainer.model.midpoints[class_index] for class_index in np.argmax(regression_proba,axis=1)])

        class_pred = np.column_stack((classification_pred, regression_pred))
        
        store_predictions(pred=np.column_stack((classification_pred, regression_proba)),labels=class_true,model_name=f"{task}_{experiment_name}_{disagreement}_{regression_name}",proyect_path=project_path,test_type="validation")

    else: 
        class_pred,class_true=generate_prediction_classification(trainer)
        store_predictions(pred=class_pred,labels=class_true,model_name=f"{task}_{experiment_name}_{disagreement}_{regression_name}",proyect_path=project_path,test_type="validation")

    log_base_multitask_tensorboard(trainer=trainer,log_dir=logger.log_dir,pred=class_pred,labels=class_true)
    trainer.model.on_validation_epoch_start()
    trainer.test(datamodule=data_module)

    if "R2ccpLoss" in regression_name:
        class_pred,class_true=generate_prediction_classification_RAC(trainer,type_model="multitask")
        classification_pred = class_pred[:,0]
        regression_proba= class_pred[:,1:]
        regression_pred=np.array([trainer.model.midpoints[class_index] for class_index in np.argmax(regression_proba,axis=1)])
        class_pred = np.column_stack((classification_pred, regression_pred))
        
        store_predictions(pred=np.column_stack((classification_pred, regression_proba)),labels=class_true,model_name=f"{task}_{experiment_name}_{disagreement}_{regression_name}",proyect_path=project_path,test_type="test")

    else:
        class_pred,class_true=generate_prediction_classification(trainer)
        store_predictions(pred=class_pred,labels=class_true,model_name=f"{task}_{experiment_name}_{disagreement}_{regression_name}",proyect_path=project_path,test_type="test")

    log_base_multitask_tensorboard(trainer=trainer,log_dir=logger.log_dir,pred=class_pred,labels=class_true)
    trainer.model.on_test_epoch_start()
    del model
    
    del trainer
if __name__ == '__main__':
    
    torch.set_float32_matmul_precision("high")
    seed = 42
    seed_everything(seed)
    
    project_path=os.path.dirname(os.path.abspath("__file__")).split("src")[0]+"src/"
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    

    if args.config:
        config_file_path = args.config
    else:
        raise ValueError("Configuration file path is required")
    

    
    config=yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)
    print(f"Configuration loaded: {config}")
    for arg in ARGUMENTS_TO_CHECK:
        if arg not in config:
            raise ValueError(f"Configuration file is missing required field: {arg}")

    if not does_path_exist(project_path+"data/final/"):
        os.makedirs(project_path+"data/final/")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    fast_dev_run=False

    if "fast_dev_run" in config:
        fast_dev_run = config["fast_dev_run"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    print("Loading the dataset")
    disagreement= config["disagreement"][0] if len(config["disagreement"])>0 else "distance"

    if "dataset_name" in config:
        
        try:
            login(token =os.environ['HUGGINGFACE_TOKEN'])
            dataset = load_dataset(config["dataset_name"])

            data_module = JigsawUnintendedDataModule(
                tokenizer=tokenizer,
                dataset=dataset,
                train_csv=None,
                val_csv=None,
                test_csv=None,
                batch_size=config["data_batch_size"],
                padding="max_length",
                truncation="only_first",
                max_length=tokenizer.model_max_length,
                task=config["task"][0],
                disagreement=disagreement,
                preprocess_data_path=project_path+"data/final/"
                )
            del dataset
        except Exception as e:
            raise ValueError(str(e)+" If the error persist you should use the local dataset file.")
    else:
        data_module = JigsawUnintendedDataModule(
            tokenizer=tokenizer,
            dataset=None,
            train_csv=project_path+config["train_file"],
            val_csv=project_path+config["valid_file"],
            test_csv=project_path+config["test_file"],
            batch_size=config["data_batch_size"],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            task=config["task"][0],
            disagreement=disagreement,
            preprocess_data_path=project_path+"data/final/",
            num_workers=config["num_workers"]
        )
    print(f"Data module created")

    if not does_path_exist(project_path+"models/checkpoints"):
        os.makedirs(project_path+"models/checkpoints")
    if not does_path_exist(project_path+"results/test_results"):
        os.makedirs(project_path+"results/test_results")
    if not does_path_exist(project_path+"results/val_results"):
        os.makedirs(project_path+"results/val_results")
    
    if not does_path_exist(project_path+"results/logs"):
        os.makedirs(project_path+"results/logs")
    print(f"Training started")
    model_name = config["model_name"]
    disagreement_once=0
    for task in config["task"]:
        if task=="classification":

            for experiment_name in config["experiment_name"]:
                train_procedure_classification(task=task,model_name=model_name,config=config,experiment_name=experiment_name,
                project_path=project_path,data_module=data_module,fast_dev_run=fast_dev_run)
                gc.collect()
        elif task=="regression":
            for regression_name in config["regression_name"]:

                for disagreement in config["disagreement"]:
                        train_procedure_regression(task=task,model_name=model_name,config=config,disagreement=disagreement,
                        regression_name=regression_name,project_path=project_path,
                        data_module=data_module,fast_dev_run=fast_dev_run)
                        gc.collect()
        else:
            for regression_name in config["regression_name"]:

                for disagreement in config["disagreement"]:
                    for experiment_name in config["experiment_name"]:
                        train_procedure_multi(task=task,model_name=model_name,config=config,disagreement=disagreement,
                        regression_name=regression_name,experiment_name=experiment_name,project_path=project_path,
                        data_module=data_module,fast_dev_run=fast_dev_run)
                        gc.collect()



                    
