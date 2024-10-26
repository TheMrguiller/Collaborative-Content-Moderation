from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath("__file__")).split("src")[0]+"src/")
import glob

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


project_path=os.path.dirname(os.path.abspath("__file__")).split("src")[0]+"/src/"
logs_path= project_path+"results/logs"
directories = sorted([d for d in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, d))])
classification_pd=pd.DataFrame(columns=['test_f1_score','test_accuracy', 'test_ece', 'test_ace', 'test_class_logloss',
'LAC_CP_Accuracy','LAC_CP_F1_Score' ,'CCLAC_CP_Accuracy','CCLAC_CP_F1_Score','CRC_CP_Accuracy','CRC_CP_F1_Score',
'LAC_Prediction_Set_Length', 'LAC_Marginal_Coverage','LAC_Uncertainty_Correlation_to_Prediction_Set', 'LAC_Review_Efficiency', 'LAC_Uncertain_Examples_Model_Inaccurate', 'LAC_Under_Confident_Examples', 'LAC_F1_Score_Uncertainty', 
'CCLAC_Prediction_Set_Length', 'CCLAC_Marginal_Coverage','CCLAC_Uncertainty_Correlation_to_Prediction_Set', 'CCLAC_Review_Efficiency', 'CCLAC_Uncertain_Examples_Model_Inaccurate', 'CCLAC_Under_Confident_Examples', 'CCLAC_F1_Score_Uncertainty',
'CRC_Prediction_Set_Length', 'CRC_Marginal_Coverage','CRC_Uncertainty_Correlation_to_Prediction_Set', 'CRC_Review_Efficiency', 'CRC_Uncertain_Examples_Model_Inaccurate', 'CRC_Under_Confident_Examples', 'CRC_F1_Score_Uncertainty',
])
regression_pd=pd.DataFrame(columns=['test_mse','test_mae','test_mbe',

'AbsoluteResidual_ICP', 'AbsoluteResidual_DI', 'AbsoluteResidual_Mean_Interval_Size', 'AbsoluteResidual_Median_Interval_Size', 'AbsoluteResidual_25th_Quantile_Interval_Size', 'AbsoluteResidual_75th_Quantile_Interval_Size', "AbsoluteResidual_Correlation_Interval_Size_Disagreement",
'Gamma_ICP', 'Gamma_DI', 'Gamma_Mean_Interval_Size', 'Gamma_Median_Interval_Size', 'Gamma_25th_Quantile_Interval_Size', 'Gamma_75th_Quantile_Interval_Size', "Gamma_Correlation_Interval_Size_Disagreement",
'RACCP_ICP', 'RACCP_DI', 'RACCP_Mean_Interval_Size', 'RACCP_Median_Interval_Size', 'RACCP_25th_Quantile_Interval_Size', 'RACCP_75th_Quantile_Interval_Size', "RACCP_Correlation_Interval_Size_Disagreement",
'RN_ICP', 'RN_DI', 'RN_Mean_Interval_Size', 'RN_Median_Interval_Size', 'RN_25th_Quantile_Interval_Size', 'RN_75th_Quantile_Interval_Size', "RN_Correlation_Interval_Size_Disagreement"])
for filename in directories:
    log_dir = os.path.join(logs_path, filename)
    log_dir=get_last_created_folder(log_dir)
    log_dir=log_dir+"/"
    for event_filename in os.listdir(log_dir):
        if event_filename.startswith("events.out.tfevents"):
            filepath = os.path.join(log_dir, event_filename)
            acc = EventAccumulator(
                filepath
            )
            acc.Reload()
            tags=acc.Tags()
            for metric_name in tags["scalars"]:
                if metric_name in classification_pd.columns:
                    classification_pd.loc[filename,metric_name]=acc.Scalars(metric_name)[-1].value
                if metric_name in regression_pd.columns:
                    regression_pd.loc[filename,metric_name]=acc.Scalars(metric_name)[-1].value
classification_pd.to_excel(project_path+"results/classification_metrics.xlsx")
regression_pd.to_excel(project_path+"results/regression_metrics.xlsx")