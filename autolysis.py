# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "numpy",
#     "openai",
#     "pandas",
#     "python-dotenv",
#     "requests",
#     "scikit-learn",
#     "seaborn",
# ]
# ///
import os
import pandas as pd
import numpy as np
import sys as sys
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import requests
import chardet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
import base64
import re


load_dotenv()

"""# Read csv file and put the data into pandas DataFrame"""

def read_csv(file_path):
  """
  Function:
    Reads a CSV file.
  Args:
    file_path: Path to the CSV file.
  Returns:
    A dataframe having data from csv file.
  """
  print(file_path)
  try:
    # Get the encoding of the file
    with open(file_path, 'rb') as f:
      result = chardet.detect(f.read())
      encoding = result['encoding']
      print(encoding)
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, encoding=encoding, encoding_errors='ignore')
    print(df.head())
    return df
  except FileNotFoundError:
    print(f"Error: File not found at path: {file_path}")
    return None
  except pd.errors.ParserError:
    print(f"Error: Could not parse the CSV file. Check its format.")
    return None
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return None

"""# Check the number of rows, columns and missing values"""

def analyze_df(df):
  """
   Gets dataframe, analyzes its rows, columns, and missing values.
  Args:
    df: dataframe to analyze.
  Returns:
    A dictionary containing the analysis results.
  """

  try:
    # get the shape of dataframe
    num_rows, num_cols = df.shape

    # Calculate the number of missing values in each column
    missing_values = df.isnull().sum()

    # Calculate the percentage of missing values in each column
    missing_percentages = (missing_values / num_rows) * 100


    # Store the analysis results in a dictionary
    results = {
        "rows": num_rows,
        "columns": num_cols,
        "percentage_missing_values": missing_percentages.to_dict()  # Convert to dictionary
        }
    print(results)
    return results
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return None

"""# Get the descriptive statistics for the numerical features in the dataframe"""

def get_descriptive_stats(df):
  """
  Function:
    To get the descriptive statistics for given dataframe
  Args:
    df: dataframe to be analyzed.
  Returns:
    dataframe with descriptive statistics
  """
  desc_stats = df.describe()
  return desc_stats

"""# Get the numeric and categorical columns for analysing the data"""

#  Identifies numeric and categorical columns in a Pandas DataFrame ####
def get_column_types(df):
  """
  Identifies numeric and categorical columns in a Pandas DataFrame.
  Args:
    df: The input Pandas DataFrame.
  Returns:
    A tuple containing two lists: numeric column names and categorical column names.
  """
  numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
  categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
  return numeric_cols, categorical_cols

"""# Generate correlation heatmap"""

def generate_correlation_heatmap(df, title="Correlation Heatmap"):
  """
  Generates a correlation heatmap for numerical variables in a Pandas DataFrame.
  Args:
    df: The input Pandas DataFrame.
    title: The title of the heatmap (default: "Correlation Heatmap").
  """

  # Select only numerical features for correlation analysis
  numerical_df = df.select_dtypes(include=['number'])

  # Calculate the correlation matrix
  correlation_matrix = numerical_df.corr()
  # Create the heatmap using seaborn
  plt.figure(figsize=(10, 8))  # Adjust figure size as needed
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title(title)
  plt.savefig('heatmap.png', dpi=64)
  plt.show()
  return correlation_matrix

"""# Get the LLM response for given prompt"""

def get_LLMResponse(prompt):
  """
  Function:
    Generates a response from LLM for the given prompt.
  Args:
    prompt: the prompt to be sent to LLM
  Returns:
    response from LLM
  """
  AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
  #AIPROXY_TOKEN = userdata.get('AIPROXY_TOKEN')
  API_URL = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

  headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
    }
  data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }

  response = requests.post(API_URL, headers=headers, json=data)
  print("LLM Response code: ", response.status_code)
  if response.status_code == 200:
    return response
  else:
    exit(1)

def get_LLMResponseForImage(Image, prompt):
  AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
  #AIPROXY_TOKEN = userdata.get('AIPROXY_TOKEN')
  API_URL = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'


  # Read the image file as binary data
  with open(Image, 'rb') as image_file:
      image_data = image_file.read()

  # Encode the image data to base64
  base64_image = base64.b64encode(image_data).decode('utf-8')

  headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
    }

  data = {
      "model": "gpt-4o-mini",
      "messages": [
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": prompt
                  },
                  {
                      "type": "image_url",
                      "image_url": {
                          "detail": "low",
                          # Instead of passing the image URL, we create a base64 encoded data URL
                          "url": f"data:image/jpeg;base64,{base64_image}"
                      }
                  }
              ]
          }
      ]
  }
  response = requests.post(API_URL, headers=headers, json=data)
  print("LLM Response code: ", response.status_code)
  if response.status_code == 200:
    return response
  else:
    exit(1)

def get_LLMResponseForFeatures(df, column_info, summary_stats):
    """
    Prompts the LLM to suggest feature engineering strategies.

    Args:
        df: The input DataFrame.
        column_info: A dictionary containing column names and their data types.
        summary_stats: Summary statistics of the DataFrame.

    Returns:
        A dictionary containing suggested features for dropping, one-hot encoding, and ordinal encoding.
    """

    prompt = f"""
    For data analysis, Identify features that can be dropped(ex. features that are not useful for predictive modeling or clustering with a brief explanation for each column).
    Identify features for one-hot encoding.
    Identify features for ordinal encoding.
    Provide your response in JSON format with the following structure:
    json {{ "drop_features": ["feature1", "feature2", ...], "onehot_features": ["feature3", "feature4", ...], "ordinal_features": ["feature5", "feature6", ...]

    Column Information:
    {column_info}

    Summary Statistics:
    {summary_stats}

    Data Head:
    {df.head()}
    """

    # Send the prompt to the LLM (using your preferred method, e.g., get_LLMResponse)
    response = get_LLMResponse(prompt)

    if response.status_code == 200:
        return response
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None

"""# Get Percentage of outliers in each column"""

def get_outlier_percentage(df, column):
  """
  Function:
    Gets the percentage of outliers for given column.
  Args:
    df: Dataframe
    column: Column to analyse outliers
  Returns:
    Percentage of outliers in given column.
  """
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
  total_entries = df[column].count()
  return (outliers / total_entries) * 100

"""# Identify Clusters in the given dataset"""

def generate_elbow_chart(X_scaled, k_range=(1, 11), filename="elbow_chart.png"):
    """
    Generates an elbow chart for KMeans clustering and saves it as an image.

    Args:
        X_scaled: Scaled data for clustering.
        k_range: Range of k values to consider.
        filename: Name of the file to save the image (default: "elbow_chart.png").
    """
    wcss = []  # Within-cluster sum of squares

    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method
    plt.figure(figsize=(8, 8))  # Adjust figure size for desired aspect ratio
    plt.plot(range(k_range[0], k_range[1]), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.savefig(filename, dpi=64, bbox_inches='tight', pad_inches=0.5)  # Adjust dpi and padding as needed
    plt.show()

def identify_clusters(X_scaled, k=3):
    """
    Identifies clusters in a DataFrame using KMeans clustering.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    X_scaled['cluster'] = kmeans.fit_predict(X_scaled)
    return X_scaled

def apply_pca_with_variance(X_scaled, variance_threshold=0.90):
    """
    Applies PCA to scaled data and retains components explaining a specified variance threshold.

    Args:
        X_scaled: Scaled data for PCA.
        variance_threshold: Desired variance explained (default: 0.80).

    Returns:
        A tuple containing:
            - pca: The fitted PCA object.
            - X_pca: The transformed data with reduced dimensions.
    """
    pca = PCA()  # Initialize PCA without specifying n_components
    pca.fit(X_scaled)  # Fit PCA to the scaled data

    # Calculate cumulative explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # Find the number of components explaining the desired variance
    n_components = len(cumulative_variance[cumulative_variance <= variance_threshold]) + 1

    # if n_components == len(cumulative_variance) + 1: # if n_components is greater than or equal to maximum components required to explain the desired variance, then we will set n_componenets to a number to prevent error
    #     n_components = len(cumulative_variance)
    # print(f"Number of components explaining {variance_threshold*100:.2f}% variance: {n_components}")

    # Apply PCA with the determined number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca

"""# Preprocess data"""

def preprocess_data(df_data, numeric_features, categorical_features):
    """
    Imputes missing values, scales numerical features, and encodes categorical features.

    Args:
        df_data: The input DataFrame.
        numeric_features: A list of numerical features to process.
        categorical_features: A list of categorical features to process.

    Returns:
        A DataFrame with preprocessed features.
    """

    # 1. Impute and scale numerical features
    num_imputer = SimpleImputer(strategy='mean')
    df_data[numeric_features] = num_imputer.fit_transform(df_data[numeric_features])
    scaler = StandardScaler()
    df_data[numeric_features] = scaler.fit_transform(df_data[numeric_features])
    print(df_data.shape)

    # 2. Impute and encode categorical features
    cat_imputer = SimpleImputer(strategy='most_frequent')  # Impute with most frequent value
    df_data[categorical_features] = cat_imputer.fit_transform(df_data[categorical_features])
    # if No. of unique values in a column is more than 20, then drop the column
    for col in df_data[categorical_features]:
      if df_data[col].nunique() > 20:
        print(f"Dropping column '{col}' due to more than 20 unique values.")
        df_data = df_data.drop(columns=[col])
    categorical_features = [col for col in categorical_features if col in df_data.columns]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # One-hot encoding
    encoded_data = encoder.fit_transform(df_data[categorical_features])
    encoded_df_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
    df_data = df_data.drop(categorical_features, axis=1)  # Drop original categorical columns
    df_data = pd.concat([df_data, encoded_df_data], axis=1)  # Concatenate encoded features
    print(df_data.shape)
    return df_data

"""# Generate cluster map in 2D"""

# Apply PCA to reduce dimensionality to 2
def generate_cluster_map(X_scaled):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    # Visualize clusters using scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_scaled['cluster'], cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clusters Visualization')
    plt.savefig('clusters_2d.png', dpi=64)
    plt.show()
    return df

"""# Copy the files to correct directory"""

def copy_file(source_path, destination_path):
       """Copies a file using lower-level file I/O."""
       with open(source_path, 'rb') as source_file, open(destination_path, 'wb') as destination_file:
           destination_file.write(source_file.read())

"""# Generate the mardown file from the response got from LLM"""

def generate_report(df, numeric_columns, file_path, response_json):
    """
    Generates a report folder with README.md and cluster image.

    Args:
        df: The input DataFrame.
        numeric_columns: List of numerical columns for clustering.
        file_path: Path to the CSV file.
        response_json: JSON response from OpenAI.
    """

    # 1. Create folder named after CSV file (without extension)
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension
    folder_path = file_name  # Create folder with the same name
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

    # 2. Copy cluster image to the folder
    image_source = 'clusters_2d.png'
    image_destination = os.path.join(folder_path, image_source)
    copy_file(image_source, image_destination)  # Copy image to folder
    image_source = 'heatmap.png'
    image_destination = os.path.join(folder_path, image_source)
    copy_file(image_source, image_destination)  # Copy image to folder


    # 3. Generate README.md from OpenAI response JSON
    try:
        readme_content = response_json['choices'][0]['message']['content']
        readme_path = os.path.join(folder_path, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    except (KeyError, IndexError) as e:
        print(f"Error extracting README content from JSON: {e}")
        print("Using default README content.")
        # ... (add your default README content here) ...

    print(f"Report generated in folder: {folder_path}")

"""# Main Function"""

############ MAIN FUNCTION############

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python script_name.py <csv_file_path>")
    sys.exit(1)

  file_path = sys.argv[1]
  df = read_csv(file_path)
  if df is None:
      exit(1)
  descriptive_stats = get_descriptive_stats(df)
  #print(descriptive_stats)
  df_info = analyze_df(df)
  numeric_columns, categorical_columns = get_column_types(df)
  corr_mat = generate_correlation_heatmap(df)
  outlier_percentages = {}
  for column in numeric_columns:
    outlier_percentage = get_outlier_percentage(df, column)
    outlier_percentages[column] = outlier_percentage

  print("Outlier Percentages:")
  for column, percentage in outlier_percentages.items():
    print(f"  {column}: {percentage:.2f}%")
  X_scaled = preprocess_data(df, numeric_columns, categorical_columns)
  n_component, X_pca = apply_pca_with_variance(X_scaled)
  print(X_scaled.shape, n_component, X_pca.shape)
  generate_elbow_chart(X_scaled)
  prompt = "I have an elbow chart for KMeans clustering. Determine the optimal number of clusters (k) from chart.Return only optimal k as a dictionary variable"
  response = get_LLMResponseForImage('elbow_chart.png', prompt)
  # Extract the content from the response
  content = response.json()['choices'][0]['message']['content']

  # Use a regular expression to find the integer value of k
  match = re.search(r'\d+', content)
  # Let the optimal k value be 3 by default
  optimal_k=3
  if match:
    optimal_k = int(match.group())
    print(f"Optimal k: {optimal_k}")
  clustered_df = identify_clusters(X_scaled,optimal_k )
  generate_cluster_map(clustered_df)
  file_name = file_path
  df_info
  column_info = df.dtypes.to_dict()  # Column names and types
  summary_stats = df.describe().to_string()  # Summary statistics
  outlier_percentages
  corr_mat
  cluster_image = plt.imread('clusters_2d.png')
  prompt = f"""
  Generate a README.md file that summarizes the data {df.head()}
  """
  # Send the prompt to the LLM (using your preferred method, e.g., get_LLMResponse)
  response = get_LLMResponse(prompt)
  if response:
    try:
        response_json = response.json()
        generate_report(df, numeric_columns, file_path, response_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
else:
    print(f"Request failed with status code: {response.status_code}")
