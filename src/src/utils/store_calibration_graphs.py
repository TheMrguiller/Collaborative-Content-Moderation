from src.visualization.tensorboard_graph import tensorboard_ace,tensorboard_calibration_curve,tensorboard_class_accuracy,tensorboard_class_mean_error,tensorboard_mean_regression_error_disagreement,tensorboard_regression_accuracy,tensorboard_toxic_score,tensorboard_model_toxic_pred_pred_disagreement_heatmap,tensorboard_model_toxic_pred_real_disagreement_heatmap
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
from pytorch_lightning import Trainer
from src.utils.compute_disagreement import get_distance_labels

def log_plots(writer, name_plot, figure):
    # Initialize SummaryWriter with the existing log directory
    

    # Convert figure to a numpy array
    figure.canvas.draw()
    image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    # Log the image as a tensor
    writer.add_image(name_plot, image, dataformats='HWC')

    

def log_base_class_tensorboard(trainer:Trainer,log_dir,pred,labels):
    writer = SummaryWriter(log_dir)
    distance_disagreement = np.array(get_distance_labels(labels[:,0]))
    log_plots(writer=writer,name_plot="Adaptive Calibration Error (ACE)",figure=tensorboard_ace(ece_per_bin=trainer.model.ece_per_bin,bin_edges=trainer.model.bin_edges))
    log_plots(writer=writer,name_plot="Calibration Curve",figure=tensorboard_calibration_curve(pred=pred,labels=labels[:,0]))
    log_plots(writer=writer,name_plot="Class Accuracy per Bin of Disagreement for Toxicity score",figure=tensorboard_class_accuracy(pred,labels[:,0],distance_disagreement))
    log_plots(writer=writer,name_plot="Mean Toxic Class Error by Disagreement Bins",figure=tensorboard_class_mean_error(pred,labels[:,0],distance_disagreement))
    log_plots(writer=writer,name_plot="True vs Predicted Toxic Score",figure=tensorboard_toxic_score(labels[:,0],pred))
    log_plots(writer=writer,name_plot="Model Distribution of Toxicity Scores for Real Different Disagreement Bins",figure=tensorboard_model_toxic_pred_real_disagreement_heatmap(pred,distance_disagreement))
    writer.close()

def log_base_regression_tensorboard(trainer:Trainer,log_dir,pred,labels):
    writer = SummaryWriter(log_dir)
    
    log_plots(writer=writer,name_plot="Mean Regression Error by Disagreement Bins",figure=tensorboard_mean_regression_error_disagreement(pred,labels[:,1]))
    log_plots(writer=writer,name_plot="Regression Accuracy",figure=tensorboard_regression_accuracy(labels[:,1],pred))
    writer.close()

def log_base_multitask_tensorboard(trainer:Trainer,log_dir,pred,labels):
    writer = SummaryWriter(log_dir)
    distance_disagreement = np.array(get_distance_labels(labels[:,0]))
    log_plots(writer=writer,name_plot="Adaptive Calibration Error (ACE)",figure=tensorboard_ace(ece_per_bin=trainer.model.ece_per_bin,bin_edges=trainer.model.bin_edges))
    
    log_plots(writer=writer,name_plot="Calibration Curve",figure=tensorboard_calibration_curve(pred=pred[:,0],labels=labels[:,0]))
    
    log_plots(writer=writer,name_plot="Class Accuracy per Bin of Disagreement for Toxicity score",figure=tensorboard_class_accuracy(pred[:,0],labels[:,0],distance_disagreement))
    
    log_plots(writer=writer,name_plot="Mean Toxic Class Error by Disagreement Bins",figure=tensorboard_class_mean_error(pred[:,0],labels[:,0],distance_disagreement))
    
    log_plots(writer=writer,name_plot="True vs Predicted Toxic Score",figure=tensorboard_toxic_score(labels[:,0],pred[:,0]))
    
    log_plots(writer=writer,name_plot="Model Distribution of Toxicity Scores for Real Different Disagreement Bins",figure=tensorboard_model_toxic_pred_real_disagreement_heatmap(pred[:,0],distance_disagreement))
    
    log_plots(writer=writer,name_plot="Model Distribution of Toxicity Scores for Model Different Disagreement Bins",figure=tensorboard_model_toxic_pred_pred_disagreement_heatmap(pred[:,0],pred[:,1]))
    
    log_plots(writer=writer,name_plot="Mean Regression Error by Disagreement Bins",figure=tensorboard_mean_regression_error_disagreement(pred[:,1],labels[:,1]))
    
    log_plots(writer=writer,name_plot="Regression Accuracy",figure=tensorboard_regression_accuracy(labels[:,1],pred[:,1]))
    
    writer.close()
