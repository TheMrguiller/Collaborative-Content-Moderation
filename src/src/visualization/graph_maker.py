from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def show_data_distribution(dataset,column_name, title='Data Distribution'):
    n, bins, patches = plt.hist(dataset[column_name], bins=10, color='skyblue', width=0.1)

    # Change color for toxic bars to red and add labels
    for i in range(len(bins) - 1):
        if bins[i] >= 0.5:
            patches[i].set_facecolor('red')

    # Add count values for each bin
    for count, edge, patch in zip(n, bins, patches):
        plt.text(edge + 0.05, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=8)

    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True)
    plt.show()

def show_number_of_comments_greater_and_less_than_threshold(dataset, column_name, threshold):
    # Get number of comments greater than threshold
    upper_cut = dataset[dataset[column_name] >= 1-threshold].shape[0]

    # Get number of comments less than threshold
    lower_cut = dataset[dataset[column_name] <= threshold].shape[0]

    # number_of_elements = len(upper_cut) + len(lower_cut)
    total_elements = dataset.shape[0]
     # Number of elements in between
    in_between = total_elements - upper_cut - lower_cut

    # Calculate percentages
    upper_cut_percentage = (upper_cut / total_elements) * 100
    lower_cut_percentage = (lower_cut / total_elements) * 100
    in_between_percentage = (in_between / total_elements) * 100

    # # Print the results
    # print(f"Number of comments greater than {threshold}: {upper_cut} ({upper_cut_percentage:.2f}%)")
    # print(f"Number of comments less than or equal to {threshold}: {lower_cut} ({lower_cut_percentage:.2f}%)")
    # print(f"Number of comments in between: {in_between} ({in_between_percentage:.2f}%)")
    # print(f"Total number of comments: {total_elements}")

    # Data for the pie chart
    labels = ['Greater than threshold', 'Less than or equal to threshold', 'In between']
    sizes = [upper_cut_percentage, lower_cut_percentage, in_between_percentage]
    colors = ['#ff9999','#66b3ff','#99ff99']
    explode = (0.1, 0.1, 0.1)  # explode a slice if required

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Distribution of Comments Relative to Threshold')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def plot_toxicity_distribution(df, annotator_column, toxicity_column):
    # Filter the DataFrame for each bin
    bin_10_20 = df[(df[annotator_column] > 10) & (df[annotator_column] <= 20)]
    bin_20_50 = df[(df[annotator_column] > 20) & (df[annotator_column] <= 50)]
    bin_greater_50 = df[df[annotator_column] > 50]

    # Create a new DataFrame to hold the bin data
    bin_data = [
        (bin_10_20[toxicity_column], '10-20'),
        (bin_20_50[toxicity_column], '20-50'),
        (bin_greater_50[toxicity_column], '>50')
    ]

    # Concatenate the data for plotting
    concatenated_data = pd.concat([pd.Series(data, name='toxicity').to_frame().assign(bin=bin_name) for data, bin_name in bin_data])

    # Plot the distribution
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='bin', y='toxicity', data=concatenated_data, palette="Set3")
    plt.title('Distribution of Toxicity Scores by Annotator Count Bins')
    plt.xlabel('Annotator Count Bins')
    plt.ylabel('Toxicity Score')
    plt.show()

def annotator_count_heatmap(df,bins_annotator,labels_annotator):
    
    df['annotator_bins'] = pd.cut(df['toxicity_annotator_count'], bins=bins_annotator, labels=labels_annotator, right=False)

    # Binning toxicity scores into increments of 0.1
    bins_toxicity = np.arange(0, 1.1, 0.1)
    labels_toxicity = [f'{i:.1f}-{i+0.1:.1f}' for i in bins_toxicity[:-1]]
    df['toxicity_bins'] = pd.cut(df['toxicity'], bins=bins_toxicity, labels=labels_toxicity, right=False)

    # Group by the bins and count occurrences
    toxicity_distribution = df.groupby(['annotator_bins', 'toxicity_bins']).size().reset_index(name='count')

    # Pivot the table for better visualization
    toxicity_pivot = toxicity_distribution.pivot(index='toxicity_bins', columns='annotator_bins', values='count').fillna(0)

    # Plotting
    plt.figure(figsize=(14, 7))
    sns.heatmap(toxicity_pivot, cmap="YlGnBu", cbar_kws={'label': 'Count'}, annot=True, fmt='g')
    plt.title('Distribution of Toxicity Scores for Different Annotator Bins')
    plt.xlabel('Annotator Count Bins')
    plt.ylabel('Toxicity Score Bins')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

def plot_toxic_label_count(df):
    total_count = len(df)
    toxic_count = len(df[df['toxicity'] >= 0.5])
    non_toxic_count = len(df[df['toxicity'] < 0.5])
    # print(f'Total count: {total_count}')
    # print(f'Toxic count: {toxic_count}')
    # print(f'Non-toxic count: {non_toxic_count}')
    # Calculate percentages
    toxic_percentage = (toxic_count / total_count) * 100
    non_toxic_percentage = (non_toxic_count / total_count) * 100

    # Plotting
    plt.bar(['Non-Toxic', 'Toxic'], [non_toxic_percentage, toxic_percentage], color=['skyblue', 'red'])

    # Adding the percentage values on top of the bars
    for i, percentage in enumerate([non_toxic_percentage, toxic_percentage]):
        plt.text(i, percentage + 0.5, f"{percentage:.2f}%", ha='center', va='bottom')

    plt.xlabel('Toxicity')
    plt.ylabel('Percentage of Elements')
    plt.title('Toxic vs Non-Toxic Elements (Percentage)')
    plt.grid(axis='y')

    plt.legend(['Non-Toxic', 'Toxic'], loc='upper right')  # Adding legend

    plt.show()

def annotation_distribution(df,custom_bins):
    hist, bins = np.histogram(df['toxicity_annotator_count'], bins=custom_bins)
    # Plotting
    plt.bar([f'{bins[i]}-{bins[i+1]-1}' if i < len(bins) - 2 else f'>{bins[i]}' for i in range(len(bins) - 1)], hist, color='skyblue', edgecolor='black')
    for i, count in enumerate(hist):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    plt.xlabel('Toxicity Annotator Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Toxicity Annotator Count')
    plt.grid(axis='y')
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
    plt.show()

def annotation_toxicity_distribution(df,custom_bins):
    toxic = df[df['toxicity'] >= 0.5]
    non_toxic = df[df['toxicity'] < 0.5]

    # Count total elements within each bin
    total_counts, _ = np.histogram(df['toxicity_annotator_count'], bins=custom_bins)

    # Count toxic and non-toxic elements within each bin
    toxic_counts, _ = np.histogram(toxic['toxicity_annotator_count'], bins=custom_bins)
    non_toxic_counts, _ = np.histogram(non_toxic['toxicity_annotator_count'], bins=custom_bins)

    # Convert counts to percentages
    toxic_percentages = (toxic_counts / total_counts) * 100
    non_toxic_percentages = (non_toxic_counts / total_counts) * 100

    # Plotting
    width = 0.35  # Width of the bars
    x = np.arange(len(custom_bins) - 1)  # x locations for the groups

    plt.bar(x - width/2, non_toxic_percentages, width, color='skyblue', edgecolor='black', label='Non-Toxic')
    plt.bar(x + width/2, toxic_percentages, width, color='red', edgecolor='black', label='Toxic')

    # Adding percentages on top of bars
    for i, percentage in enumerate(non_toxic_percentages):
        plt.text(i - width/2, percentage + 0.5, f"{percentage:.2f}%", ha='center', va='bottom')
    for i, percentage in enumerate(toxic_percentages):
        plt.text(i + width/2, percentage + 0.5, f"{percentage:.2f}%", ha='center', va='bottom')

    plt.xlabel('Toxicity Annotator Count')
    plt.ylabel('Percentage')
    plt.title('Distribution of Toxicity Annotator Count')
    plt.grid(axis='y')
    plt.xticks(range(len(custom_bins) - 1), [f'{custom_bins[i]}-{custom_bins[i+1]-1}' if i < len(custom_bins) - 2 else f'>{custom_bins[i]}' for i in range(len(custom_bins) - 1)], rotation=45, ha='right')

    # Adding legend
    plt.legend()

    plt.show()
    
def disagreement_toxicity_distribution(df,custom_bins,disagreement_column_name):
    toxic = df[df['toxicity'] >= 0.5]
    non_toxic = df[df['toxicity'] < 0.5]

    # Count total elements within each bin
    total_counts, _ = np.histogram(df[disagreement_column_name], bins=custom_bins)

    # Count toxic and non-toxic elements within each bin
    toxic_counts, _ = np.histogram(toxic[disagreement_column_name], bins=custom_bins)
    non_toxic_counts, _ = np.histogram(non_toxic[disagreement_column_name], bins=custom_bins)

    # Convert counts to percentages
    toxic_percentages = (toxic_counts / total_counts) * 100
    non_toxic_percentages = (non_toxic_counts / total_counts) * 100

    # Plotting
    width = 0.35  # Width of the bars
    x = np.arange(len(custom_bins) - 1)  # x locations for the groups

    plt.bar(x - width/2, non_toxic_percentages, width, color='skyblue', edgecolor='black', label='Non-Toxic')
    plt.bar(x + width/2, toxic_percentages, width, color='red', edgecolor='black', label='Toxic')

    # Adding percentages on top of bars
    for i, percentage in enumerate(non_toxic_percentages):
        plt.text(i - width/2, percentage + 0.5, f"{percentage:.2f}%", ha='center', va='bottom')
    for i, percentage in enumerate(toxic_percentages):
        plt.text(i + width/2, percentage + 0.5, f"{percentage:.2f}%", ha='center', va='bottom')

    plt.xlabel('Disagreement Count')
    plt.ylabel('Percentage')
    plt.title('Distribution of Toxicity Annotator Count')
    plt.grid(axis='y')
    plt.xticks(range(len(custom_bins) - 1), [f'{custom_bins[i]}-{custom_bins[i+1]-1}' if i < len(custom_bins) - 2 else f'>{custom_bins[i]}' for i in range(len(custom_bins) - 1)], rotation=45, ha='right')

    # Adding legend
    plt.legend()

    plt.show()

def disagreement_distribution(df,disagreement_column_name):
    plt.hist(df[disagreement_column_name], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(disagreement_column_name)
    plt.ylabel('Frequency')
    plt.title('Distribution of '+disagreement_column_name)
    plt.grid(axis='y')
    plt.show()

def disagreement_method_comparison(df,method1_column,method2_column):
    plt.figure(figsize=(10, 6))
    # yticks = np.arange(0, 0.26, 0.01)
    plt.scatter(df[method1_column], df[method2_column], alpha=0.5)
    plt.xlabel(method1_column)
    plt.ylabel(method2_column)
    plt.title(method1_column+ " vs " +method2_column)
    plt.grid(True)
    # plt.yticks(yticks)
    plt.show()
    
