import torch
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoConfig
from src.models.loss import FocalLoss, WeightedBCE,WeightedMSE,R2ccpLoss
import torch.nn.functional as F
from torchmetrics import Accuracy,F1Score
from torchmetrics.classification import BinaryCalibrationError
from src.models.metrics import coverage_probability,mean_absolute_error,mean_squared_error,calculate_ace,computelog_loss,mean_bias_error


class BaseTransformers(pl.LightningModule):
    def __init__(self, model_name, learning_rate, weight_decay, config=None):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Load the configuration
        if config is None:
            self.config = AutoConfig.from_pretrained(self.model_name)
        else:
            self.config = config
        
        # Load the model and tokenizer with the configuration
        self.model = AutoModel.from_pretrained(self.model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.accu = Accuracy(task="binary")
        self.f1 = F1Score(task="binary",average="macro")
        self.binary_calibration_error = BinaryCalibrationError(n_bins=10,norm="l1")
    
    
class TransformersForSequenceClassification(BaseTransformers):
    def __init__(self, model_name, learning_rate, weight_decay, config=None,experiment_name="cross_entropy"):
        super().__init__(model_name, learning_rate, weight_decay, config)
        torch.manual_seed(42)
        # Add a classification head
        self.classification_head = torch.nn.Linear(self.config.hidden_size, 1)
        
        
        if experiment_name == "focal_loss":
            # We want to center the loss around the hard examples and not around the easy examples. Being hard examples the ones that should be more uncertain.
            self.loss = FocalLoss()
        elif experiment_name == "focal_loss_Weighted":
            self.loss = FocalLoss()
        else:
            raise ValueError(f"Invalid experiment name: {experiment_name}")
        self.experiment_name=experiment_name
        self.logits = []
        self.labels = []
        self.losses = []
        self.ece_per_bin =0
        self.bin_edges =0
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Extract the [CLS] token's embedding from the last hidden state
        cls_token = last_hidden_state[:, 0, :]
        
        # Pass the dropout-applied [CLS] token through the classification head
        logits = self.classification_head(cls_token)
        return logits

    def training_step(self, batch, batch_idx):
        loss,logits,labels=self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True,on_step=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):

        loss,logits,labels=self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True,on_step=False)
        # add logits and labels to instance attributes, but make sure to detach them
        # from the computational graph first
        probabilities = F.sigmoid(logits)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(loss.detach().cpu())
        return {'loss': loss, 'logits': logits, 'labels': labels,"prediction":probabilities}
    
    def test_step(self, batch, batch_idx):

        loss,logits,labels=self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True,on_step=False)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(loss.detach().cpu())
        probabilities = F.sigmoid(logits)
        return {'loss': loss, 'logits': logits, 'labels': labels,"prediction":probabilities}
        
    def on_test_epoch_end(self):
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        classification_values=F.sigmoid(logits)

        ece = self.binary_calibration_error(classification_values,(labels[:,0] >= 0.5).int().view(-1,1))
        ace,ece_per_bin,bin_edges = calculate_ace((labels[:,0] >= 0.5).int().view(-1,1),classification_values,10)
        log_loss = computelog_loss(labels[:,0].view(-1,1),classification_values)
        accuracy=self.accu(classification_values,(labels[:,0] >= 0.5).int().view(-1,1))
        f1_score = self.f1(classification_values,(labels[:,0] >= 0.5).int().view(-1,1))

        self.log_dict({"test_loss":loss,"test_accuracy":accuracy,"test_f1_score":f1_score,"test_ece":ece,"test_ace":ace,"test_class_logloss":log_loss},on_epoch=True, prog_bar=True,on_step=False)
        self.ece_per_bin =ece_per_bin
        self.bin_edges = bin_edges
        return {"predictions":classification_values,"labels":labels,"ece_per_bin":ece_per_bin,"bin_edges":bin_edges}

    def on_test_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        self.ece_per_bin = 0
        self.bin_edges = 0
        return super().on_test_epoch_start()
    
    def on_validation_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        self.ece_per_bin = 0
        self.bin_edges = 0
        return super().on_validation_epoch_start()
        
    # TODO metrics
    def on_validation_epoch_end(self):
        
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        classification_values=F.sigmoid(logits)

        ece = self.binary_calibration_error(classification_values,(labels[:,0] >= 0.5).int().view(-1,1))
        ace,ece_per_bin,bin_edges = calculate_ace((labels[:,0] >= 0.5).int().view(-1,1),classification_values,10)
        log_loss = computelog_loss(labels[:,0].view(-1,1),classification_values)
        accuracy=self.accu(classification_values,(labels[:,0] >= 0.5).int().view(-1,1))
        f1_score = self.f1(classification_values,(labels[:,0] >= 0.5).int().view(-1,1))

        self.log_dict({"val_loss":loss,"val_accuracy":accuracy,"val_f1_score":f1_score,"val_ece":ece,"val_ace":ace,"val_class_logloss":log_loss},on_epoch=True, prog_bar=True,on_step=False)
        self.ece_per_bin =ece_per_bin
        self.bin_edges = bin_edges
        return {"predictions":classification_values,"labels":labels,"ece_per_bin":ece_per_bin,"bin_edges":bin_edges}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=1)

        return (
        {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        })

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add your custom logic to run directly before `optimizer.step()`
        if epoch == 0:
            lr_scale = (batch_idx + 1) / (len(self.trainer.train_dataloader) * (epoch + 1))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate
        optimizer.step(closure=optimizer_closure)
        
    def predict(self, text):
        self.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():  # Disable gradient calculations
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to device
            outputs = self(**inputs)
            probabilities = torch.sigmoid(outputs)
        return probabilities
    
    def _common_step(self, batch, batch_idx):
        if "soft_label" not in self.experiment_name:
            labels = (batch['labels'][:,0]>=0.5).float()
        else:
            labels = batch['labels'][:,0]
        labels = labels.view(-1, 1)
        # labels should be a tensor with shape [[0.0, 1.0],[1.0, 0.0],...] as if it was a sof label. 
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        logits = self(input_ids, attention_mask)

        if "Weighted" in self.experiment_name:
            tox_weights = batch['tox_weights']
            tox_weights = tox_weights.view(-1, 1)
            classification_loss = self.loss(logits, labels,tox_weights)
        else:
            classification_loss = self.loss(logits, labels)
        return classification_loss, logits, batch['labels']
    
class TransformersForSequenceClassificationMultiOutput(TransformersForSequenceClassification):
    def __init__(self, model_name, learning_rate, weight_decay, config=None,experiment_name="cross_entropy",regression_name="MSE"):
        super().__init__(model_name, learning_rate, weight_decay, config,experiment_name)
        torch.manual_seed(42)
        # Add a classification head
        self.classification_head = torch.nn.Linear(self.config.hidden_size, 2)
        if regression_name == "MSE":
            self.regression_loss = torch.nn.MSELoss()
        elif regression_name == "BCE":
            self.regression_loss = torch.nn.BCEWithLogitsLoss()
        elif regression_name == "WeightedMSE":
            self.regression_loss = WeightedMSE()
        elif regression_name == "WeightedBCE":
            self.regression_loss = WeightedBCE()
        else:
            raise ValueError(f"Invalid regression loss name: {regression_name}")
        self.regression_name=regression_name
        self.save_hyperparameters()
    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Extract the [CLS] token's embedding from the last hidden state
        cls_token = last_hidden_state[:, 0, :]
        
        # Pass the dropout-applied [CLS] token through the classification head
        logits = self.classification_head(cls_token)
        return logits

    def training_step(self, batch, batch_idx):

        classification_loss,regression_loss,logits,labels=self._common_step(batch, batch_idx)
        loss = classification_loss+regression_loss
        self.log('train_loss', loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('classification_loss', classification_loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('regression_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        return {'loss': loss, 'logits': logits}
    
    def validation_step(self, batch, batch_idx):

        classification_loss,regression_loss,logits,labels=self._common_step(batch, batch_idx)
        loss = classification_loss+regression_loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('val_classification_loss', classification_loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('val_regression_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        # add logits and labels to instance attributes, but make sure to detach them
        # from the computational graph first
        
        prediction= F.sigmoid(logits)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(loss.detach().cpu())
        return {'loss': loss, 'logits': logits, 'labels': labels,"prediction":prediction}
    
    def test_step(self, batch, batch_idx):

        classification_loss,regression_loss,logits,labels=self._common_step(batch, batch_idx)
        loss = classification_loss+regression_loss
        self.log('test_loss', loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('test_classification_loss', classification_loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('test_regression_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        prediction= F.sigmoid(logits)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(loss.detach().cpu())
        return {'loss': loss, 'logits': logits, 'labels': labels,"prediction":prediction}
    
    def on_test_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        self.ece_per_bin = 0
        self.bin_edges = 0
        return super().on_test_epoch_start()
    
    def on_test_epoch_end(self):
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        classification_values=F.sigmoid(logits[:,0])
        regression_value = F.sigmoid(logits[:,1])

        ece = self.binary_calibration_error(classification_values,(labels[:,0] >= 0.5).int())
        ace,ece_per_bin,bin_edges = calculate_ace((labels[:,0] >= 0.5).int(),classification_values,10)
        log_loss = computelog_loss(labels[:,0],classification_values)
        regression_log_loss = computelog_loss(labels[:,1],regression_value)
        accuracy=self.accu(classification_values,(labels[:,0] >= 0.5).int())
        f1_score = self.f1(classification_values,(labels[:,0] >= 0.5).int())

        mae = mean_absolute_error(regression_value,labels[:,1])
        mse = mean_squared_error(regression_value,labels[:,1])
        mbe = mean_bias_error(regression_value,labels[:,1])
        if classification_values.dim() == 1:
            classification_values = classification_values.unsqueeze(1)
        if regression_value.dim() == 1:
            regression_value = regression_value.unsqueeze(1)

        predictions=torch.cat((classification_values, regression_value), dim=1)
        
        self.ece_per_bin = ece_per_bin
        self.bin_edges = bin_edges
        self.log_dict({"test_loss":loss,"test_accuracy":accuracy,"test_f1_score":f1_score,"test_ece":ece,"test_ace":ace,"test_class_logloss":log_loss,"test_regre_logloss":regression_log_loss,"test_mae":mae,"test_mse":mse,"test_mbe":mbe},on_epoch=True, prog_bar=True,on_step=False)
        
        return {"predictions":predictions,"labels":labels,"ece_per_bin":ece_per_bin,"bin_edges":bin_edges}

    def on_validation_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        self.ece_per_bin = 0
        self.bin_edges = 0
        return super().on_validation_epoch_start()
    # TODO metrics
    def on_validation_epoch_end(self):
        
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        
        classification_values=F.sigmoid(logits[:,0])
        regression_value = F.sigmoid(logits[:,1])
        ece = self.binary_calibration_error(classification_values,(labels[:,0] >= 0.5).int())
        ace,ece_per_bin,bin_edges = calculate_ace((labels[:,0] >= 0.5).int(),classification_values,10)
        log_loss = computelog_loss(labels[:,0],classification_values)
        regression_log_loss = computelog_loss(labels[:,1],regression_value)

        accuracy=self.accu(classification_values,(labels[:,0] >= 0.5).int())
        f1_score = self.f1(classification_values,(labels[:,0] >= 0.5).int())
        
        mae = mean_absolute_error(regression_value,labels[:,1])
        mse = mean_squared_error(regression_value,labels[:,1])
        mbe = mean_bias_error(regression_value,labels[:,1])
        self.log_dict({"val_loss":loss,"val_accuracy":accuracy,"val_f1_score":f1_score,"val_ece":ece,"val_ace":ace,"val_class_logloss":log_loss,"val_regre_logloss":regression_log_loss,"val_mae":mae,"val_mse":mse,"val_mbe":mbe},on_epoch=True, prog_bar=True,on_step=False)
        
        if classification_values.dim() == 1:
            classification_values = classification_values.unsqueeze(1)
        if regression_value.dim() == 1:
            regression_value = regression_value.unsqueeze(1)
        
        self.ece_per_bin = ece_per_bin
        self.bin_edges = bin_edges
        # Concatenate along the correct dimension
        predictions = torch.cat((classification_values, regression_value), dim=1)

        return {"predictions":predictions,"labels":labels,"ece_per_bin":ece_per_bin,"bin_edges":bin_edges}

        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=1)

        return (
        {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        })

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add your custom logic to run directly before `optimizer.step()`
        if epoch == 0:
            lr_scale = (batch_idx + 1) / (len(self.trainer.train_dataloader) * (epoch + 1))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate
        optimizer.step(closure=optimizer_closure)

    def predict(self, text):
        self.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():  # Disable gradient calculations
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to device
            outputs = self(**inputs)
            probabilities = torch.sigmoid(outputs)
        return probabilities
    
    def _common_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        logits = self(input_ids, attention_mask)
   
        ## Classification loss
        if "soft_label" not in self.experiment_name:
            class_labels = (batch['labels'][:,0]>=0.5).float()
        else:
            class_labels = batch['labels'][:,0]

        
        if "Weighted" in self.experiment_name:
            tox_weights = batch['tox_weights']
            tox_weights = tox_weights.view(-1) 
            classification_loss = self.loss(logits[:,0], class_labels,tox_weights)
        else:
            classification_loss = self.loss(logits[:,0], class_labels)

        # Regression loss
        regression = F.sigmoid(logits[:,1])
        regression_labels = batch['labels'][:,1]
        

        if "Weighted" in self.regression_name:
            dis_weights = batch['dis_weights']
            dis_weights = dis_weights.view(-1) 
            if isinstance(self.regression_loss, (WeightedBCE)):
                regression_loss=self.regression_loss(logits[:,1],regression_labels,dis_weights)
            else:
                regression_loss=self.regression_loss(regression,regression_labels,dis_weights)
        else:
            if isinstance(self.regression_loss, (torch.nn.BCEWithLogitsLoss)):
                regression_loss=self.regression_loss(logits[:,1],regression_labels)
            else:
                regression_loss=self.regression_loss(regression,regression_labels)
    
        return classification_loss,regression_loss, logits, batch['labels']
    
    def logits_to_binary_distributions(self, probabilities:torch.Tensor, eps=1e-7):
        """
        Convert probabilities to binary distributions.
        
        Args:
        - probabilities (torch.Tensor): Tensor of probabilities with shape (batch_size, num_classes).
        - eps (float): Small value for numerical stability. 1e-7 due to float32 precision.
        
        Returns:
        - binary_distributions (torch.Tensor): Tensor of binary distributions with shape (batch_size, num_classes, 2).
        """
        # Clamp probabilities to avoid extreme values
        
        probabilities = torch.clamp(probabilities.float(), min=eps, max=1.0 - eps)
        
        # Convert probabilities to binary distributions
        binary_distributions = torch.stack([1 - probabilities, probabilities], dim=-1)
        
        return binary_distributions
        


class TransformersForSequenceClassificationMultitask(TransformersForSequenceClassificationMultiOutput):
    def __init__(self, model_name, learning_rate, weight_decay, config=None,experiment_name="cross_entropy",regression_name="MSE"):
        super().__init__(model_name, learning_rate, weight_decay, config,experiment_name,regression_name)
        
        torch.manual_seed(42)
        # Add a classification head
        self.classification_head = torch.nn.Linear(self.config.hidden_size, 1)
        self.regression_head = torch.nn.Linear(self.config.hidden_size, 1)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Extract the [CLS] token's embedding from the last hidden state
        cls_token = last_hidden_state[:, 0, :]
        
        
        classification_logits = self.classification_head(cls_token)
        regression_logits = self.regression_head(cls_token)
        return torch.cat((classification_logits, regression_logits), dim=1)
    

class TransformersRegression(BaseTransformers):
    def __init__(self, model_name, learning_rate, weight_decay, config=None,regression_name="MSE"):
        super().__init__(model_name, learning_rate, weight_decay, config)
        torch.manual_seed(42)
        # Add a classification head
        self.regression_head = torch.nn.Linear(self.config.hidden_size, 1)
        if regression_name == "MSE":
            self.regression_loss = torch.nn.MSELoss()
        elif regression_name == "BCE":
            self.regression_loss = torch.nn.BCEWithLogitsLoss()
        elif regression_name == "WeightedMSE":
            self.regression_loss = WeightedMSE()
        elif regression_name == "WeightedBCE":
            self.regression_loss = WeightedBCE()
        else:
            raise ValueError(f"Invalid regression loss name: {regression_name}")
        self.regression_name=regression_name
        self.logits=[]
        self.labels=[]
        self.losses=[]
        
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Extract the [CLS] token's embedding from the last hidden state
        cls_token = last_hidden_state[:, 0, :]
        
        # Pass the dropout-applied [CLS] token through the classification head
        logits = self.regression_head(cls_token)
        return logits

    def training_step(self, batch, batch_idx):
        regression_loss,logits,labels=self._common_step(batch, batch_idx)
        self.log('train_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        
        return {'loss': regression_loss, 'logits': logits}
    
    def validation_step(self, batch, batch_idx):

        regression_loss,logits,labels=self._common_step(batch, batch_idx)
        
        self.log('val_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        
        
        prediction= F.sigmoid(logits)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(regression_loss.detach().cpu())
        return {'loss': regression_loss, 'logits': logits, 'labels': labels,"prediction":prediction}
    
    def test_step(self, batch, batch_idx):

        regression_loss,logits,labels=self._common_step(batch, batch_idx)
        
        self.log('test_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        
        prediction= F.sigmoid(logits)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(regression_loss.detach().cpu())
        return {'loss': regression_loss, 'logits': logits, 'labels': labels,"prediction":prediction}
    
    def on_test_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        return super().on_test_epoch_start()
    
    def on_test_epoch_end(self):
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        prediction= F.sigmoid(logits)
        mae = mean_absolute_error(prediction,labels[:,1].view(-1, 1))
        mse = mean_squared_error(prediction,labels[:,1].view(-1, 1))
        mbe = mean_bias_error(prediction,labels[:,1].view(-1, 1))
        regression_log_loss = computelog_loss(labels[:,1].view(-1, 1),prediction)
        self.log_dict({"test_loss":loss,"test_regre_logloss":regression_log_loss,"test_mae":mae,"test_mse":mse,"test_mbe":mbe},on_epoch=True, prog_bar=True,on_step=False)
        
        return {"predictions":prediction,"labels":labels}
    
    def on_validation_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
            
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        prediction= F.sigmoid(logits)
        mae = mean_absolute_error(prediction,labels[:,1].view(-1, 1))
        mse = mean_squared_error(prediction,labels[:,1].view(-1, 1))
        mbe = mean_bias_error(prediction,labels[:,1].view(-1, 1))
        regression_log_loss = computelog_loss(labels[:,1].view(-1, 1),prediction)
        self.log_dict({"val_loss":loss,"val_regre_logloss":regression_log_loss,"val_mae":mae,"val_mse":mse,"val_mbe"},on_epoch=True, prog_bar=True,on_step=False)
        
        return {"predictions":prediction,"labels":labels}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=1)

        return (
        {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        })
    
    def predict(self, text):

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self(**inputs)
            probabilities = torch.sigmoid(outputs)
        return probabilities
    
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add your custom logic to run directly before `optimizer.step()`
        if epoch == 0:
            lr_scale = (batch_idx + 1) / (len(self.trainer.train_dataloader) * (epoch + 1))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate
        optimizer.step(closure=optimizer_closure)
    
    def _common_step(self, batch, batch_idx):
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        logits = self(input_ids, attention_mask)
        labels = batch['labels'][:,1]
        labels = labels.view(-1, 1)
        
        regression = F.sigmoid(logits)
        if "Weighted" in self.regression_name:
            dis_weights = batch['dis_weights']
            if dis_weights.dim() == 1:
                dis_weights = dis_weights.view(-1, 1)  # Ensure shape [N, 1]
            if isinstance(self.regression_loss, (WeightedBCE)):
                regression_loss=self.regression_loss(logits,labels,dis_weights)
            else:
                regression_loss=self.regression_loss(regression,labels,dis_weights)
        else:
            if isinstance(self.regression_loss, (torch.nn.BCEWithLogitsLoss)):
                regression_loss=self.regression_loss(logits,labels)
            else:
                regression_loss=self.regression_loss(regression,labels)
        return regression_loss, logits, batch['labels']

class TransformersRegressionRAC(BaseTransformers):
    def __init__(self, model_name, learning_rate, weight_decay, config=None,regression_name="R2ccpLoss",num_classes=20,tau=0.1,p=0.5):
        super().__init__(model_name, learning_rate, weight_decay, config)
        torch.manual_seed(42)
        
        self.regression_head = torch.nn.Linear(self.config.hidden_size, num_classes)
        midpoints_step_division = torch.linspace(0, 1, steps=num_classes+1)

        
        self.midpoints = (midpoints_step_division[:-1] + midpoints_step_division[1:]) / 2
        self.regression_loss = R2ccpLoss(p=p,tau=tau,midpoints=self.midpoints)

        self.num_classes=num_classes
        self.isWeighted=False
        if "Weighted" in regression_name:
            self.isWeighted=True
        self.logits=[]
        self.labels=[]
        self.losses=[]
        self.save_hyperparameters()
        
    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Extract the [CLS] token's embedding from the last hidden state
        cls_token = last_hidden_state[:, 0, :]
        
        # Pass the dropout-applied [CLS] token through the classification head
        logits = self.regression_head(cls_token)
        return logits
    
    def _common_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        logits = self(input_ids, attention_mask)
        
        # print(f"Shape of regression logits:{logits[:,1:].shape}")
        regression = torch.nn.functional.softmax(logits, dim=1)#regression = logits[:,1:]#
        # print(f"Shape of prediction:{regression.shape}")
        regression_labels = batch['labels'][:,1]
        if self.isWeighted:
            dis_weights = batch['dis_weights']
            
            regression_loss=self.regression_loss(regression,regression_labels,dis_weights)
        else:
            regression_loss=self.regression_loss(regression,regression_labels)

        
        return regression_loss, logits, batch['labels']
    
    def predict(self, text):
        self.model.eval()  # Ensure model is in eval mode
        with torch.no_grad():  # Disable gradient calculations
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move inputs to device
            outputs = self(**inputs)
            probabilities = F.softmax(outputs,dim=1)
        return probabilities
    
    def training_step(self, batch, batch_idx):
        regression_loss,logits,labels=self._common_step(batch, batch_idx)
        self.log('train_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        
        return {'loss': regression_loss, 'logits': logits}
    
    def validation_step(self, batch, batch_idx):
            
        regression_loss,logits,labels=self._common_step(batch, batch_idx)
        self.log('val_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        # add logits and labels to instance attributes, but make sure to detach them
        # from the computational graph first
        
        prediction_reg=F.softmax(logits,dim=1)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(regression_loss.detach().cpu())
        return {'loss': regression_loss, 'logits': logits, 'labels': labels,"prediction":prediction_reg}
    
    def test_step(self, batch, batch_idx):
        regression_loss,logits,labels=self._common_step(batch, batch_idx)
        
        self.log('test_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)

        prediction_reg=F.softmax(logits,dim=1)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(regression_loss.detach().cpu())
        return {'loss': regression_loss, 'logits': logits, 'labels': labels,"prediction":prediction_reg}
    
    def on_test_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        return super().on_test_epoch_start()
    
    def on_validation_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        return super().on_validation_epoch_start()
    
    def on_test_epoch_end(self):
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)

        prediction_reg=F.softmax(logits,dim=1)

        prediction_reg = prediction_reg.argmax(dim=1)
        # Convert the tensor of indices to a list (optional, but can be useful)
        prediction_reg= prediction_reg.tolist()

        # Populate the midpoints based on indices
        populated_midpoints = [self.midpoints[idx] for idx in prediction_reg]
        populated_midpoints = torch.tensor(populated_midpoints)
        mae = mean_absolute_error(labels[:,1],populated_midpoints)
        mse = mean_squared_error(labels[:,1],populated_midpoints)
        mbe = mean_bias_error(populated_midpoints,labels[:,1])
        regression_log_loss = computelog_loss(labels[:,1],populated_midpoints)
        self.log_dict({"test_loss":loss,"test_regre_logloss":regression_log_loss,"test_mae":mae,"test_mse":mse,"test_mbe":mbe},on_epoch=True, prog_bar=True,on_step=False)
        
        return {"predictions":torch.tensor(prediction_reg),"labels":labels}

    def on_validation_epoch_end(self):
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        prediction_reg=F.softmax(logits,dim=1)

        prediction_reg = prediction_reg.argmax(dim=1)
        # Convert the tensor of indices to a list (optional, but can be useful)
        prediction_reg= prediction_reg.tolist()

        # Populate the midpoints based on indices
        populated_midpoints = [self.midpoints[idx] for idx in prediction_reg]
        populated_midpoints = torch.tensor(populated_midpoints)
        mae = mean_absolute_error(labels[:,1],populated_midpoints)
        mse = mean_squared_error(labels[:,1],populated_midpoints)
        mbe = mean_bias_error(populated_midpoints,labels[:,1])
        regression_log_loss = computelog_loss(labels[:,1],populated_midpoints)
        self.log_dict({"val_loss":loss,"val_regre_logloss":regression_log_loss,"val_mae":mae,"val_mse":mse,"val_mbe":mbe},on_epoch=True, prog_bar=True,on_step=False)
        


        return {"predictions":torch.tensor(prediction_reg),"labels":labels,}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=1)

        return (
        {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        })

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add your custom logic to run directly before `optimizer.step()`
        if epoch == 0:
            lr_scale = (batch_idx + 1) / (len(self.trainer.train_dataloader) * (epoch + 1))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate
        optimizer.step(closure=optimizer_closure)
    

class TransformersForSequenceClassificationMultitaskRAC(TransformersForSequenceClassification):

    def __init__(self, model_name, learning_rate, weight_decay, config=None,experiment_name="cross_entropy",regression_name="R2ccpLoss",num_classes=20,tau=0.1,p=0.5):
        super().__init__(model_name, learning_rate, weight_decay, config,experiment_name)
        torch.manual_seed(42)
        # Add a classification head
        self.classification_head = torch.nn.Linear(self.config.hidden_size, 1)
       
        
        self.regression_head = torch.nn.Linear(self.config.hidden_size, num_classes)

        midpoints_step_division = torch.linspace(0, 1, steps=num_classes+1)

        
        self.midpoints = (midpoints_step_division[:-1] + midpoints_step_division[1:]) / 2
        self.regression_loss = R2ccpLoss(p=p,tau=tau,midpoints=self.midpoints)

        self.num_classes=num_classes
        self.isWeighted=False
        if "Weighted" in regression_name:
            self.isWeighted=True
        
        self.save_hyperparameters()
    def forward(self, input_ids, attention_mask):
            
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Extract the [CLS] token's embedding from the last hidden state
        cls_token = last_hidden_state[:, 0, :]
        
        classification_logits = self.classification_head(cls_token)
        regression_logits = self.regression_head(cls_token)
        # classification_logits = classification_logits.view(-1, 1)
        return torch.cat((classification_logits, regression_logits), dim=1)
    
    def training_step(self, batch, batch_idx):
        classification_loss,regression_loss,logits,labels=self._common_step(batch, batch_idx)
        loss = classification_loss+regression_loss
        self.log('train_loss', loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('classification_loss', classification_loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('regression_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        return {'loss': loss, 'logits': logits}
    
    def _common_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        logits = self(input_ids, attention_mask)
        # print(f"Shape of logits:{logits.shape}")
        # Classification loss
        class_labels = batch['labels'][:,0]
        if "Weighted" in self.experiment_name:
            tox_weights = batch['tox_weights']
            tox_weights = tox_weights.view(-1) 
            classification_loss = self.loss(logits[:,0], class_labels,tox_weights)
        else:
            classification_loss = self.loss(logits[:,0], class_labels)
        
        # print(f"Shape of regression logits:{logits[:,1:].shape}")
        regression = torch.nn.functional.softmax(logits[:,1:], dim=1)#regression = logits[:,1:]#
        # print(f"Shape of prediction:{regression.shape}")
        regression_labels = batch['labels'][:,1]
        if self.isWeighted:
            dis_weights = batch['dis_weights']
            
            regression_loss=self.regression_loss(regression,regression_labels,dis_weights)
        else:
            regression_loss=self.regression_loss(regression,regression_labels)

        
        return classification_loss,regression_loss, logits, batch['labels']
    
    def validation_step(self, batch, batch_idx):
            
        classification_loss,regression_loss,logits,labels=self._common_step(batch, batch_idx)
        loss = classification_loss+regression_loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('val_classification_loss', classification_loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('val_regression_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        # add logits and labels to instance attributes, but make sure to detach them
        # from the computational graph first
        
        prediction_class= F.sigmoid(logits[:,0])
        prediction_reg=F.softmax(logits[:,1:],dim=1)
        prediction = torch.cat((prediction_class.view(-1,1),prediction_reg),dim=1)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(loss.detach().cpu())
        return {'loss': loss, 'logits': logits, 'labels': labels,"prediction":prediction}
    
    def test_step(self, batch, batch_idx):
        classification_loss,regression_loss,logits,labels=self._common_step(batch, batch_idx)
        loss = classification_loss+regression_loss
        self.log('test_loss', loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('test_classification_loss', classification_loss, on_epoch=True, prog_bar=True,on_step=False)
        self.log('test_regression_loss', regression_loss, on_epoch=True, prog_bar=True,on_step=False)
        prediction_class= F.sigmoid(logits[:,0])
        prediction_reg=F.softmax(logits[:,1:],dim=1)
        prediction = torch.cat((prediction_class.view(-1,1),prediction_reg),dim=1)
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(loss.detach().cpu())
        return {'loss': loss, 'logits': logits, 'labels': labels,"prediction":prediction}
    
    def on_test_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        return super().on_test_epoch_start()
    
    def on_validation_epoch_start(self) -> None:
        self.logits.clear()
        self.labels.clear()
        self.losses.clear()
        return super().on_validation_epoch_start()
    
    def on_test_epoch_end(self):
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        prediction_class= F.sigmoid(logits[:,0])
        prediction_reg=F.softmax(logits[:,1:],dim=1)
        predictions = torch.cat((prediction_class.view(-1,1),prediction_reg),dim=1)
        ece = self.binary_calibration_error(prediction_class,(labels[:,0] >= 0.5).int())
        ace,ece_per_bin,bin_edges = calculate_ace((labels[:,0] >= 0.5).int(),prediction_class,10)
        log_loss = computelog_loss(labels[:,0],prediction_class)

        accuracy=self.accu(prediction_class,(labels[:,0] >= 0.5).int())
        f1_score = self.f1(prediction_class,(labels[:,0] >= 0.5).int())
        prediction_reg = prediction_reg.argmax(dim=1)
        # Convert the tensor of indices to a list (optional, but can be useful)
        prediction_reg= prediction_reg.tolist()

        # Populate the midpoints based on indices
        populated_midpoints = [self.midpoints[idx] for idx in prediction_reg]
        populated_midpoints = torch.tensor(populated_midpoints)
        mae = mean_absolute_error(labels[:,1],populated_midpoints)
        mse = mean_squared_error(labels[:,1],populated_midpoints)
        mbe = mean_bias_error(populated_midpoints,labels[:,1])
        regression_log_loss = computelog_loss(labels[:,1],populated_midpoints)
        self.log_dict({"test_loss":loss,"test_accuracy":accuracy,"test_f1_score":f1_score,"test_ece":ece,"test_ace":ace,"test_class_logloss":log_loss,"test_regre_logloss":regression_log_loss,"test_mae":mae,"test_mse":mse,"test_mbe":mbe},on_epoch=True, prog_bar=True,on_step=False)
        
        
        self.ece_per_bin = ece_per_bin
        self.bin_edges = bin_edges

        return {"predictions":predictions,"labels":labels,"ece_per_bin":ece_per_bin,"bin_edges":bin_edges}

    def on_validation_epoch_end(self):
        loss = torch.stack(self.losses).mean()
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        prediction_class= F.sigmoid(logits[:,0])
        prediction_reg=F.softmax(logits[:,1:],dim=1)
        predictions = torch.cat((prediction_class.view(-1,1),prediction_reg),dim=1)
        ece = self.binary_calibration_error(prediction_class,(labels[:,0] >= 0.5).int())
        ace,ece_per_bin,bin_edges = calculate_ace((labels[:,0] >= 0.5).int(),prediction_class,10)
        log_loss = computelog_loss(labels[:,0],prediction_class)
        accuracy=self.accu(prediction_class,(labels[:,0] >= 0.5).int())
        f1_score = self.f1(prediction_class,(labels[:,0] >= 0.5).int())

        prediction_reg = prediction_reg.argmax(dim=1)
        # Convert the tensor of indices to a list (optional, but can be useful)
        prediction_reg= prediction_reg.tolist()

        # Populate the midpoints based on indices
        populated_midpoints = [self.midpoints[idx] for idx in prediction_reg]
        populated_midpoints = torch.tensor(populated_midpoints)
        mae = mean_absolute_error(labels[:,1],populated_midpoints)
        mse = mean_squared_error(labels[:,1],populated_midpoints)
        mbe = mean_bias_error(populated_midpoints,labels[:,1])
        regression_log_loss = computelog_loss(labels[:,1],populated_midpoints)
        self.log_dict({"val_loss":loss,"val_accuracy":accuracy,"val_f1_score":f1_score,"val_ece":ece,"val_ace":ace,"val_class_logloss":log_loss,"val_regre_logloss":regression_log_loss,"val_mae":mae,"val_mse":mse,"val_mbe":mbe},on_epoch=True, prog_bar=True,on_step=False)
        self.ece_per_bin = ece_per_bin
        self.bin_edges = bin_edges

        return {"predictions":predictions,"labels":labels,"ece_per_bin":ece_per_bin,"bin_edges":bin_edges}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=1)

        return (
        {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        })

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Add your custom logic to run directly before `optimizer.step()`
        if epoch == 0:
            lr_scale = (batch_idx + 1) / (len(self.trainer.train_dataloader) * (epoch + 1))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate
        optimizer.step(closure=optimizer_closure)

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self(**inputs)
            probabilities = torch.sigmoid(outputs[:,0])
            regression = F.softmax(outputs[:,1:])
            predictions = torch.cat((probabilities.view(-1,1),regression),dim=1)
        return predictions
    
    
    
    
