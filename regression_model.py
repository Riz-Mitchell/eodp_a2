import regression_functions as func
import pandas as pd
from scipy.stats import norm

def run_regression_model(df, test_df):
    """Outputs all the data for a regression model"""
    # Get a subsample of the data
    sample_df = df.sample(n=3000,random_state=1000)


    # Linear regression for user age & year of publication against book rating
    with open("regression_files/Linear Regression.txt", "w") as output:
        x = df[["User-Age", "Year-Of-Publication"]]
        y = df["Book-Rating"]
        
        test_x = test_df[["User-Age", "Year-Of-Publication"]]
        test_y = test_df["Book-Rating"]
        output.write(("Linear regression for user age & year of publication against book rating\n"))
        func.create_linear_regression(x, y, test_x, test_y, output)

        # Another regression with user age against year of publication
        x = df[["User-Age"]]
        y = df["Year-Of-Publication"]
        test_x = test_df[["User-Age"]]
        test_y = test_df["Year-Of-Publication"]
        output.write(("\nLinear regression for user age against year of publication\n"))
        func.create_linear_regression(x, y, test_x, test_y, output)


        # Calculate correlations
        output.write("\nCorrelation between user age and year of publication\n")
        func.cal_correlation("User-Age", "Year-Of-Publication", df, output)

        output.write("\nCorrelation between user age and book rating\n")
        func.cal_correlation("User-Age", "Book-Rating", df, output)

        output.write("\nCorrelation between year of publication and book rating\n")
        func.cal_correlation("Year-Of-Publication", "Book-Rating", df, output)



    # Create histograms to see the distribution of the variables
    user_age = df["User-Age"]
    year_of_publication = df["Year-Of-Publication"]
    book_rating = df["Book-Rating"]
    ua_bins = [i*10 for i in range(1,11)]
    yob_bins = [i*5 + 1920 for i in range(18)]
    br_bins = [i for i in range(11)]

    func.create_histogram(user_age, ua_bins, "Histogram of User Age", "User Age", "regression_files/HistUA.png")
    func.create_histogram(year_of_publication, yob_bins, "Histogram of Year Of Publication", "Year of Publication", "regression_files/HistYoP.png")
    func.create_histogram(book_rating, br_bins, "Histogram of Book Rating", "Book Rating", "regression_files/HistBR.png")


    # Plot scatterplots to visualize the data
    func.scatterplot_3var(user_age, year_of_publication, book_rating, "User Age", "Year Of Publication", "Book Rating", "regression_files/Scatter_3var.png")
    

    # Smaller sample for visual clarity
    sample_user_age = sample_df["User-Age"]
    sample_year_of_publication = sample_df["Year-Of-Publication"]
    sample_book_rating = sample_df["Book-Rating"]

    func.scatterplot_3var(sample_user_age, sample_year_of_publication, sample_book_rating,
        "User Age", "Year Of Publication", "Book Rating", "regression_files/Scatter_3var_s.png")

    # Creates QQ plots to check distribution of the variables
    func.create_qqplot(year_of_publication, norm, "Year Of Publication", "regression_files/qqplot_yop.png")
    func.create_qqplot(user_age, norm, "User Age", "regression_files/qqplot_ua.png")
    func.create_qqplot(book_rating, norm, "Book Rating", "regression_files/qqplot_br.png")
