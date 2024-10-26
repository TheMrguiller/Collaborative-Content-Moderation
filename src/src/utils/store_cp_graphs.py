from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
from src.utils.compute_disagreement import get_distance_labels
from src.utils.store_calibration_graphs import log_plots
from src.visualization.cp_graph import box_plot_uncertainty_disagreement,violinplot_uncertainty_disagreement,certainty_quantity_plot
from src.models.metrics import base_cp_metrics,uncertainty_correlation_prediction_set,uncertainty_confusion_matrix,uncertainty_base_metrics,disagreement_bin_uncertainty_base_metrics

from sklearn.metrics import accuracy_score,f1_score as f1


def store_value(value,writer,name):
    writer.add_scalar(name,value,1)
def store_class_cp_metrics(log_dir,method_name,prediction_set,labels,predictions):

    distance_disagreement=get_distance_labels(labels[:,0])
    uncertainty_cp=[1 if (prediction.all().item() or not prediction.any().item()) else 0 for prediction in prediction_set]
    prediction_set= prediction_set.numpy()
    writer = SummaryWriter(log_dir)
    set_length, marginal_coverage, conditional_coverage=base_cp_metrics(prediction_set,labels)
    store_value(set_length,writer,method_name+" Prediction Set Length")
    store_value(marginal_coverage,writer,method_name+" Marginal Coverage")
    store_value(conditional_coverage[0],writer,method_name+" No Toxic Conditional Coverage")
    store_value(conditional_coverage[1],writer,method_name+" Toxic Conditional Coverage")
    uncertaintY_correlation=uncertainty_correlation_prediction_set(uncertainty_cp,distance_disagreement)
    store_value(uncertaintY_correlation,writer,method_name+" Uncertainty Correlation to Prediction Set")
    log_plots(writer=writer,name_plot=method_name+" Box plot Uncertainty Disagreement",figure=box_plot_uncertainty_disagreement(uncertainty_cp,distance_disagreement))
    log_plots(writer=writer,name_plot=method_name+" Violin plot Uncertainty Disagreement",figure=violinplot_uncertainty_disagreement(uncertainty_cp,distance_disagreement))
    log_plots(writer=writer,name_plot=method_name+" Certainty Quantity",figure=certainty_quantity_plot(uncertainty_cp,distance_disagreement))
    TP, FP, TN, FN=uncertainty_confusion_matrix(labels[:,0]>=0.5,predictions[:,0]>=0.5,uncertainty_cp)
    MURE,uncertain_examples_model_inaccurate,under_confident_examples,f1_score=uncertainty_base_metrics(TP, FP, TN, FN)
    store_value(MURE,writer,method_name+" MURE")
    store_value(uncertain_examples_model_inaccurate,writer,method_name+" Uncertain Examples Model Inaccurate")
    store_value(under_confident_examples,writer,method_name+" Under Confident Examples")
    store_value(f1_score,writer,method_name+" F1 Score Uncertainty")
    
    cp_accurary=accuracy_score((predictions[np.array(uncertainty_cp)==0][:,0]>= 0.5).astype(int),(labels[np.array(uncertainty_cp)==0][:,0] >= 0.5).astype(int))
    store_value(cp_accurary,writer,method_name+" CP Accuracy")
    cp_f1_score = f1((predictions[np.array(uncertainty_cp)==0][:,0]>= 0.5).astype(int),(labels[np.array(uncertainty_cp)==0][:,0] >= 0.5).astype(int))
    store_value(cp_f1_score,writer,method_name+" CP F1 Score")
    writer.close()