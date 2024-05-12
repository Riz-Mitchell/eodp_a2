import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from math import sqrt
from statsmodels.graphics.gofplots import qqplot

def create_linear_regression(x, y, test_x, test_y, filename):
    """Creates a linear regression from the data and outputs some attributes"""
    model = LinearRegression()
    model.fit(x, y)
    filename.write(f"Intercept: {model.intercept_}\n")
    filename.write(f"Coefficients: {model.coef_}\n")
    prediction = model.predict(test_x)
    rmse = sqrt(mean_squared_error(test_y, prediction))
    filename.write(f"Root Mean Squared Error: {rmse}\n")


def cal_correlation(x, y, df, filename):
    """Calculates the correlation between x and y"""
    corr = df[[x, y]].corr(method='pearson')
    filename.write(f"Correlation: {corr}\n")


def scatterplot_3var(x, y, z, x_label, y_label, z_label, filename):
    """Creates a scatterplot of x, y and z"""
    fig, ax = plt.subplots()
    scatterplot = plt.scatter(x, y, c=z, cmap="viridis", s=6, linewidths=1)
    title = f"Relationship between {x_label}, {y_label} and {z_label}"
    plt.title(title, fontsize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    colorbar = fig.colorbar(scatterplot, ax = ax)
    colorbar.set_label(z_label, rotation=270)
    plt.savefig(filename)
    plt.show()


def create_histogram(data, bins, title, xlabel, filename):
    """Creates a histogram of the data"""  
    fig, ax = plt.subplots()
    ax.hist(data, bins=bins)
    plt.xticks(bins, fontsize=6)
    ax.set_xticklabels(bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.savefig(filename)
    plt.show()  

def create_qqplot(data, dist, y_label, filename):
    """Creates a qq plot of the data"""
    qqplot(data, dist, fit=True, line ='45') 
    title = f"QQ-Plot of {y_label} against a normal distribution"
    plt.title(title)
    plt.xlabel("Normal Distribution")
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show() 
