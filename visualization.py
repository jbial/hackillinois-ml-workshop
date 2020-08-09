"""Contains visualization tools used in the notebook
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML


def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    
    FROM: 
    https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.head().style.set_table_attributes("style='display:inline'").\
                  set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0"
    display(HTML(output))
    

def show_corr_matrix(df):
    """Plots the correlation matrix of the numeric columns in a dataframe
    
    Args:
        df (DataFrame): dataframe dataset
    """
    # get numeric column names
    labels = df.select_dtypes(exclude="object").columns
    
    # plot matrix and add string labels
    plt.figure(dpi=100)
    plt.title("Attribute Correlations")
    plt.imshow(df.corr(method='pearson'))
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    
def show_bivariates(df, variate, num_features=None, params=None):
    """Plots bivariate plots from a dataframe
    
    - box and whisker plots for categorical variables
    - scatter plots for numeric variables
    
    Args:
        df (DataFrame): dataframe dataset
        variate (str): column name of labels y
        num_features (int): number of input features
        params (np.array): learned parameters
    """
    num_plots = num_features or len(df.columns) - 1
    # get categorical and numeric column names
    cat_cols = df.select_dtypes(include="object")
    num_cols = df.select_dtypes(exclude="object")
    if len(cat_cols.columns) == len(num_cols.columns) - 1:
        fig, axes = plt.subplots(2, len(cat_cols.columns), figsize=(12,4))
    else:
        fig, axes = plt.subplots(1, num_plots, figsize=(12,4))
    
    for i, col in enumerate(cat_cols):
        if len(cat_cols.columns) == len(num_cols.columns) - 1:
            axis = axes[0][i]
        else:
            axis = axes[i]
        sns.boxplot(x=col, y=variate, data=df, ax=axis)
    for i, col in enumerate(num_cols[num_cols.columns[:-1]]):
        if len(cat_cols.columns) == len(num_cols.columns) - 1:
            axis = axes[1][i]
        else:
            axis = axes[i + cat_cols.shape[-1]]
        df.plot.scatter(col, variate, ax=axis, s=1)
        if params is not None:
            ys = df[[col]] * params[i + 1] + params[0]
            axis.plot(df[[col]], ys, c='r')
    fig.suptitle("Bivariate Plots")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
