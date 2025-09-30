# Titanic Dataset Analysis and Preprocessing

This notebook performs an initial analysis and preprocessing steps on the Titanic dataset to prepare it for machine learning model training.

## Data Source

The dataset used in this notebook is the classic Titanic dataset, typically found as "Titanic-Dataset.csv". It contains information about passengers on the ill-fated Titanic voyage, including whether they survived.

## Steps Performed

The following steps were performed in this notebook:

1.  **Import and Explore Data**:
    *   The dataset was imported into a pandas DataFrame.
    *   Basic information about the dataset was explored, including checking for null values (`df.info()`) and examining data types. The head of the DataFrame was also displayed.

2.  **Handle Missing Values**:
    *   Missing values in the 'Age' column were imputed with the median age, as a box plot revealed outliers that would skew the mean.
    *   Missing values in the 'Cabin' column were filled with the string 'missing'.
    *   Missing values in the 'Embarked' column were filled with the mode of the column.

3.  **Convert Categorical Features**:
    *   Categorical features ('Sex', 'Survived', 'Pclass', 'Embarked') were converted into numerical format using one-hot encoding (`pd.get_dummies`).

4.  **Normalize Numerical Features**:
    *   The 'Fare' and 'Age' features were normalized using `MinMaxScaler` to scale their values to a range between 0 and 1. New columns 'fare\_Normalised' and 'Age\_normalised' were created for the normalized values.

5.  **Visualize and Remove Outliers**:
    *   Box plots were generated for the normalized 'Fare' and 'Age' columns ('fare\_Normalised' and 'Age\_normalised') to visualize the presence of outliers.
    *   Outliers in both the 'fare\_Normalised' and 'Age\_normalised' columns were identified using the Interquartile Range (IQR) method (1.5 * IQR rule).
    *   Rows containing these outliers were directly dropped from the DataFrame.

## Libraries Used

*   `pandas` for data manipulation and analysis.
*   `matplotlib.pyplot` and `seaborn` for data visualization (specifically box plots).
*   `sklearn.preprocessing.MinMaxScaler` for feature normalization.

## How to Run the Notebook

1.  Ensure you have the "Titanic-Dataset.csv" file in the appropriate directory (`/content/`).
2.  Run the cells sequentially in the notebook.

This notebook provides a clean and preprocessed dataset (`df_encoded`) ready for building machine learning models to predict survival on the Titanic.# task1
this repo has all the codes and datasets and a short readme for the understanding of the readers for the assignment 1
