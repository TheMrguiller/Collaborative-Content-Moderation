# A Collaborative Content Moderation Framework for Toxicity Detection based on Conformalized Estimates of Annotation Disagreement
<div align="center">

<!---[[Website]](https://eureka-research.github.io)-->
[[Spotify]](https://podcasters.spotify.com/pod/show/themrguiller/episodes/A-Collaborative-Content-Moderation-Framework-for-Toxicity-Detection-based-on-Conformalized-Estimates-of-Annotation-Disagreement-e2q5rgn)
[[arXiv]]()
<!---[[PDF]](https://eureka-research.github.io/assets/eureka_paper.pdf)-->

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://github.com/TheMrguiller/Collaborative-Content-Moderation)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/eureka-research/Eureka)](LICENSE)
______________________________________________________________________

![**Review process.** The STL review process begins by extracting the representation **CLS**(xᵢ) of the content **xᵢ** and generating its corresponding classification **ŷᵢ**. After calibrating the CP algorithm, we generate the prediction set **C(xᵢ)** for classification. If the size of **C(xᵢ)** is 2 (i.e., the total number of possible classes in the problem), the comment is flagged for review by a human moderator. Otherwise, the classifier’s output is considered confident, and **ŷᵢ** is deemed a reliable prediction of the toxicity of **xᵢ**. For the composite STL models, in addition to generating the classification **ŷᵢ**, we also compute the annotation disagreement estimate **di(xᵢ)**, a regression value learned from **X** and **A**. Once the CP algorithm for regression is calibrated, we produce the confidence interval **I(xᵢ)**. If this interval exceeds a predefined ambiguity threshold **γ**, the comment is marked for human review. Otherwise, we rely on **C(xᵢ)** to assess the confidence of the classifier’s output. In the MTL approach, the only difference from the composite model is that both **ŷᵢ** and **di(xᵢ)** are generated from the same representation **CLS(xᵢ)** of the content.](images/Review_process.png)
</div>
<p><strong>Figure 1: Review process.</strong> The STL review process begins by extracting the representation 
<strong>CLS</strong>(x<sub>i</sub>) of the content <strong>x<sub>i</sub></strong> and generating its corresponding 
classification <strong>ŷ<sub>i</sub></strong>. After calibrating the CP algorithm, we generate the prediction 
set <strong>C(x<sub>i</sub>)</strong> for classification. If the size of <strong>C(x<sub>i</sub>)</strong> is 2 
(i.e., the total number of possible classes in the problem), the comment is flagged for review by a human moderator. 
Otherwise, the classifier’s output is considered confident, and <strong>ŷ<sub>i</sub></strong> is deemed a reliable 
prediction of the toxicity of <strong>x<sub>i</sub></strong>. For the composite STL models, in addition to generating 
the classification <strong>ŷ<sub>i</sub></strong>, we also compute the annotation disagreement estimate 
<strong>d<sub>i</sub>(x<sub>i</sub>)</strong>, a regression value learned from <strong>X</strong> and <strong>A</strong>. 
Once the CP algorithm for regression is calibrated, we produce the confidence interval <strong>I(x<sub>i</sub>)</strong>. 
If this interval exceeds a predefined ambiguity threshold <strong>γ</strong>, the comment is marked for human review. 
Otherwise, we rely on <strong>C(x<sub>i</sub>)</strong> to assess the confidence of the classifier’s output. 
In the MTL approach, the only difference from the composite model is that both <strong>ŷ<sub>i</sub></strong> 
and <strong>d<sub>i</sub>(x<sub>i</sub>)</strong> are generated from the same representation 
<strong>CLS(x<sub>i</sub>)</strong> of the content.
</p>


## Abstract

Content moderation typically combines the efforts of human moderators and machine learning models. However, these systems often rely on data where significant disagreement occurs during moderation, reflecting the subjective nature of toxicity perception. Rather than dismissing this disagreement as noise, we interpret it as a valuable signal that highlights the inherent ambiguity of the content—an insight missed when only the majority label is considered. In this work, we introduce a novel content moderation framework that emphasizes the importance of capturing annotation disagreement. Our approach uses multitask learning, where toxicity classification serves as the primary task and annotation disagreement is addressed as an auxiliary task. Additionally, we leverage uncertainty estimation techniques, specifically Conformal Prediction, to account for both the ambiguity in comment annotations and the model's inherent uncertainty in predicting toxicity and disagreement. The framework also allows moderators to adjust thresholds for annotation disagreement, offering flexibility in determining when ambiguity should trigger a review. We demonstrate that our joint approach enhances model performance, calibration, and uncertainty estimation, while offering greater parameter efficiency and improving the review process in comparison to single-task methods.
# Installation
Eureka requires Python ≥ 3.10. We have tested on Ubuntu 22.04, pytorch version 2.2 and cuda 12.1.0.

**Local environment:**
1. Create a new virtualenv environment with:
   ```
    virtualenv venv --python=python3.10
    source venv/bin/activate
    ```
2. Install dependecies:
   ```
    pip install -r requirements.txt

    ```
**Docker version:**
1. Having previously install [Docker](https://docs.docker.com/engine/install/ubuntu/) and being in the folder where the docker compose is:
   ```
    sudo docker compose up
    ```

If you want to have access to the dataset via HuggingFace you need to have your [HuggingFace token](https://huggingface.co/docs/hub/security-tokens) as your environment variable HUGGINGFACE_TOKEN.

# Folder structure
Scheme of folder structure
```
├── images                  # Collaborative Content Moderation Diagram
├── src                     # Main folder
│    ├── data               # Dataset folder
│    │    ├── final         # Clean dataset
│    │    ├── processed     # Medium processed dataset
│    │    └── raw           # Raw dataset
│    ├── models             # Folder for model checkpoints
│    ├── notebooks          # Dataset analysis folder
│    ├── results            # Metric store folder
│    └── src                # Main code
│        ├── config         # Training configuration files
│        ├── data           # Code for data processing
│        ├── models         # Code for model training and metrics
│        ├── utils          # Code for supporting models
│        ├── visualization  # Code for visuals generation
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt       
```
# Getting Started
Our project has different modules you need to know about: dataset, model training and results.
## Dataset
We provide several options for downloading the dataset: Hugging Face or local preprocessing.

- For those who want to download the dataset cleaned, you can download it using the following reference: [TheMrguiller/Uncertainty_Toxicity](https://huggingface.co/datasets/TheMrguiller/Uncertainty_Toxicity) 

- For those who want to download and preprocess the data do the following:
    ```
    python src/data/download_datasets.py
    ```
    ```
    python src/data/data_preprocess.py
    ```
## Model training
We provide in `src/config` a parametrized model training yaml for generating the needed models. We have provided with the configuration used for the model training of the article STL and MTL models.

The description of the yaml files are the following:
```
model_name: # Hugging Face model name
learning_rate: 
weight_decay: 
accumulate_grad_batches: 
experiment_name: # A list of the classification losses wanted to perform for the toxic detection  models focal_loss_Weighted or focal_loss
task: # A list of the task we want to perform classification, regression or multitask
disagreement: # A list of the disagreement computation techniques we want to use distance, variance and entropy
regression_name: # A list of the regression losses you want to use for the training BCE,WeightedBCE, MSE, WeightedMSE, R2ccpLoss and WeightedR2ccpLoss
dataset_name: # If Hugging Face dataset is used put "TheMrguiller/Uncertainty_Toxicity"
train_file: "data/processed/jigsaw-unintended-bias-in-toxicity-classification_train_data_clean.csv"
test_file: "data/processed/jigsaw-unintended-bias-in-toxicity-classification_test_data_clean.csv"
valid_file: "data/processed/jigsaw-unintended-bias-in-toxicity-classification_valid_data_clean.csv"
data_batch_size: # Batch size for training
num_nodes: # Num nodes for training
fast_dev_run: # If we want to do a test step
num_workers: # Num workers for dataset preparation
```
Onced the yaml is configured you can start the training by:

```
python src/src/models/train.py
```
## Results
To obtain the content moderation metrics for the classification and content moderation framework you need to use the following scripts:
1. Conformal Prediction for Classification:
   ```
   python src/src/models/get_class_cp_metrics.py
   ```
2. Conformal Prediction for Regression:
     ```
   python src/src/models/get_regre_cp_metrics.py
   ```
3. Collaborative Content Moderation performance metrics:
   ```
   python src/src/models/get_class_cp_metrics.py
   ```
   
# License
This codebase is released under [MIT License](LICENSE).

# Citation
If you find our work useful, please consider citing us!

```bibtex
@article{,
  title   = {A Collaborative Content Moderation Framework for
Toxicity Detection based on Conformalized Estimates of
Annotation Disagreement},
  author  = {Guillermo Villate-Castillo and Javier del Ser and Borja Sanz},
  year    = {2024},
  journal = {}
}
