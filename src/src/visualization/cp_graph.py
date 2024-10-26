import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def box_plot_uncertainty_disagreement(uncertainty_cp: list, distance_disagreement: list):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=50)
    sns.boxplot(x=uncertainty_cp, y=distance_disagreement, ax=ax)
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Disagreement')
    ax.grid(True)  # Adds grid lines
    plt.close(fig)  # Close to prevent immediate display if needed
    return fig

def violinplot_uncertainty_disagreement(uncertainty_cp: list, distance_disagreement: list):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=50)
    sns.violinplot(x=uncertainty_cp, y=distance_disagreement, ax=ax)
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Disagreement')
    ax.grid(True)  # Adds grid lines
    plt.close(fig)  # Close to prevent immediate display if needed
    return fig

def certainty_quantity_plot(uncertainty_cp: list,distance_disagreement: list):
    bins = np.arange(0,1.1,0.1)
    binned_data = np.digitize(distance_disagreement, bins, right=False)
    certain_list=[]
    uncertain_list=[]
    for i in range(1,len(bins)+1):
        mask = (binned_data == i).astype(int)
        uncertainty_cp_= np.array(uncertainty_cp)
        uncertain_bin=sum(uncertainty_cp_[mask==1])
        certain_bin=len(uncertainty_cp_[mask==1])-uncertain_bin
        uncertain_list.append(uncertain_bin)
        certain_list.append(certain_bin)
    
    fig=plt.figure(figsize=(10, 8), dpi=50)

    # Plot for certain_list
    plt.plot(bins, certain_list, marker='o', linestyle='-', color='b', label='Certain')

    # Plot for uncertain_list
    plt.plot(bins, uncertain_list, marker='s', linestyle='--', color='r', label='Uncertain')

    # Add labels and title
    plt.xlabel('Disagreement bins')
    plt.ylabel('Quantity')
    plt.title('Plot of Certain vs. Uncertain')
    plt.xticks(bins)  # Ensure x-axis ticks match the bins
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.close(fig)
    return fig
