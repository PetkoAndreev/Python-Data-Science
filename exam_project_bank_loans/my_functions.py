import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Function to remove the columns having percentage of null values > 45%.
def remove_nan_columns(df, percent=0.45):
    new_df_shape = df.shape
    colnames = (df.isna().sum() / len(df))
    colnames = list(colnames[colnames.values >= percent].index)
    df.drop(labels=colnames, axis=1, inplace=True)

    print("Number of columns dropped\t: ", len(colnames))
    print("\nOld dataset rows, columns", new_df_shape, "\nNew dataset rows, columns", df.shape)

    return df


# Function to fill all NAN's with the mode of the choosen column from dataframe.
def replace_nan_with_mode(col):
    # Read the column to see what values it have with dropna = False to show NAN's too.
    print(f'Values before NAN replacement:\n {col.value_counts(dropna=False)}')
    # Get the mode of the column (most frequent value).
    print(f'Mode value: {col.mode()[0]}')
    # Fill NAN with mode value.
    col = col.fillna(col.mode()[0])
    '''
    We can see now that value of mode column is increased with the value of NAN's and there are no NAN's. 
    All other values remains the same.
    '''
    print(f'Values after NAN replacement:\n {col.value_counts()}')

    return col


# Function to fill all NAN's with the median of the choosen column from dataframe.
def replace_nan_with_median(col):
    # Read the column to see what values it have with dropna = False to show NAN's too.
    print(f'Values before NAN replacement:\n {col.value_counts(dropna=False)}')
    # Get the median of the column (middle value).
    print(f'Median value: {col.median()}')
    # Fill NAN with mode value.
    col = col.fillna(col.median())
    '''
    We can see now that value of median column is increased with the value of NAN's and there are no NAN's. 
    All other values remains the same.
    '''
    print(f'Values after NAN replacement:\n {col.value_counts(dropna=False)}')

    return col


# Function for correlations in dataframe.
def get_correlations_greater_than_or_equal_to_05(df):
    # Create correlation matrix.
    corr = round(df.corr(method="pearson"), 2)

    # Select upper triangle of correlation matrix.
    corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

    # Unstack and reset index.
    corr_new = corr.unstack().reset_index()

    # Set column names.
    corr_new.columns = ["variable1", "variable2", "correlation"]

    # Drop NaN values.
    corr_new.dropna(subset=["correlation"], inplace=True)

    # Convert all negative correlation to their ABS values.
    corr_new["corr_abs"] = corr_new["correlation"].abs()

    # Filter the data - show only highly correlated data greater than or equal to 0.5.
    corr_new = corr_new[corr_new["correlation"] >= 0.5]

    # Show correlations
    return corr_new.sort_values('corr_abs', ascending=False).reset_index(drop=True)


# Function for correlations in dataframe.
def get_correlations_less_than_05(df):
    # Create correlation matrix.
    corr = round(df.corr(method="pearson"), 2)

    # Select upper triangle of correlation matrix.
    corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

    # Unstack and reset index.
    corr_new = corr.unstack().reset_index()

    # Set column names.
    corr_new.columns = ["variable1", "variable2", "correlation"]

    # Drop NaN values.
    corr_new.dropna(subset=["correlation"], inplace=True)

    # Convert all negative correlation to their ABS values.
    corr_new["corr_abs"] = corr_new["correlation"].abs()

    # Filter the data - show only highly correlated data less than 0.5.
    corr_new = corr_new[corr_new["correlation"] < 0.5]

    # Show correlations
    return corr_new.sort_values('corr_abs', ascending=False).reset_index(drop=True)


# Function to plot boxplot charts.
def plot_boxplot_chart(input_data, input_label, input_title):
    fig = plt.figure(figsize=(7, 5))
    plt.boxplot(input_data)

    plt.ylabel(input_label)
    plt.yscale('log')
    plt.grid()
    plt.title(input_title, pad=20)

    plt.show()


# Function to plot scatterplot charts.
def plot_scatter_chart(input_data_1, input_data_2, input_xlabel, input_ylabel, input_title, input_color):
    plt.figure(figsize=(10, 5))
    plt.scatter(input_data_1, input_data_2, c=input_color)

    plt.title(input_title, pad=20)
    plt.xlabel(input_xlabel)
    plt.ylabel(input_ylabel)

    plt.show()


# Function to plot pie charts.
def plot_pie_chart(input_data, input_labels, input_colors, input_title, input_explode):
    if input_colors != []:
        plt.pie(input_data,
                autopct="%1.2f%%",
                colors=input_colors,
                explode=input_explode,
                shadow=True)
    else:
        plt.pie(input_data,
                autopct="%1.2f%%",
                explode=input_explode,
                shadow=True)
    #     circ = plt.Circle((0,0),.9,color="white")
    #     plt.gca().add_artist(circ)

    plt.title(input_title)
    plt.legend(input_labels, loc="upper right")

    # plt.show()


# Function to plot horizontal bar charts.
def plot_barh_chart(input_data_1, input_data_2, input_color, input_xlabel, input_title):
    fig = plt.figure(figsize=(10, 7))
    plt.barh(input_data_1, input_data_2, color=input_color)

    plt.xlabel(input_xlabel)
    plt.title(input_title, pad=20)

    plt.show()


# Function to plot bar charts.
def plot_bar_chart(input_data_1, input_data_2, input_color, input_ylabel, input_title):
    fig = plt.figure(figsize=(15, 7))
    plt.bar(input_data_1, input_data_2, color=input_color)

    plt.ylabel(input_ylabel)
    plt.tick_params(rotation=90)
    plt.title(input_title, pad=20)

    plt.show()


# Function to plot pie subplots
def plot_two_pie_subplots(data_1, labels_1, colors_1, title_1, data_2, labels_2, colors_2, title_2, explode):
    fig, ax1 = plt.subplots(1, 2, figsize=(15, 10))

    plt.subplot(121)
    plot_pie_chart(data_1,
                   labels_1,
                   colors_1,
                   title_1,
                   explode
                   )

    plt.subplot(122)
    plot_pie_chart(data_2,
                   labels_2,
                   colors_2,
                   title_2,
                   explode
                   )

    plt.show()


def plot_hist_boxplot_subplots(
        hist1_data, hist2_data,
        box_data,
        title, labels, xlabel, ylabel
):
    fig, ax1 = plt.subplots(1, 2, figsize=(15, 10))

    plt.subplot(121)
    plt.hist(hist1_data, bins='fd', density=True, alpha=0.7)
    plt.hist(hist2_data, bins='fd', density=True, alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, pad=20)
    plt.legend(labels, loc='upper right')

    plt.subplot(122)
    plt.boxplot(box_data)

    plt.grid()
    plt.title(title, pad=20)
    plt.labels = labels
    plt.xticks([1, 2], labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def plot_hist_boxplot_subplots_without_density(
        hist1_data, hist2_data,
        box_data,
        title, labels, xlabel, ylabel
):
    fig, ax1 = plt.subplots(1, 2, figsize=(15, 10))

    plt.subplot(121)
    plt.hist(hist1_data, bins='fd', alpha=0.7)
    plt.hist(hist2_data, bins='fd', alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, pad=20)
    plt.legend(labels, loc='upper right')
    plt.yscale("log")

    plt.subplot(122)
    plt.boxplot(box_data, showfliers=False)

    plt.grid()
    plt.title(title, pad=20)
    plt.labels = labels
    plt.xticks([1, 2], labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale("log")

    plt.show()


# Function to boxplot bivariate
def plot_seaborn_boxplot(col1, col2, df1, df2, title1, title2, outliers=False, input_ylim=None):
    plt.figure(figsize=(25, 10))

    plt.subplot(121)
    plt.title(title1)
    sns.boxplot(x=col1, y=col2, data=df1, showfliers=outliers)
    if input_ylim:
        plt.ylim(input_ylim)

    plt.subplot(122)
    plt.title(title2)
    sns.boxplot(x=col1, y=col2, data=df2, showfliers=outliers)
    if input_ylim:
        plt.ylim(input_ylim)

    plt.show()


# Function to plot seaborn heatmaps
def plot_heatmap_subplots(input_data1, input_data2, list_data, group_data, title1, title2):
    plt.figure(figsize=[20, 5])

    plt.subplot(121)
    plt.title(title1)
    res = pd.pivot_table(data=input_data1,
                         values=list_data,
                         index=group_data,
                         aggfunc=np.mean)
    sns.heatmap(res, annot=True, cmap="Spectral", center=0.117)

    plt.subplot(122)
    plt.title(title2)
    res = pd.pivot_table(data=input_data2,
                         values=list_data,
                         index=group_data,
                         aggfunc=np.mean)
    sns.heatmap(res, annot=True, cmap="Spectral", center=0.117)

    plt.show()
