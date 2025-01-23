from sklearn import datasets
import pandas as pd

# Load iris dataset
iris = datasets.load_iris()

# Since this is a bunch, create a dataframe
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target

# Rename columns for better readability
iris_df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

# TASK 1: Find the number and mean of missing data
print("Number of missing values per column:")
print(iris_df.isnull().sum())  # Number of missing values in each column
print("\nProportion of missing values per column:")
print(iris_df.isnull().mean())  # Proportion of missing values in each column

# Clean data by removing rows with all missing values
cleaned_data = iris_df.dropna(how="all", inplace=True)

# Subset the first 5 rows and the first four feature columns (sepal_len, sepal_wid, petal_len, petal_wid)
iris_X = iris_df.iloc[:5, [0, 1, 2, 3]]
print("\nFirst 5 rows and selected feature columns:")
print(iris_X)

# TASK 2: Calculate the correlation matrix among features
correlation_matrix = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

