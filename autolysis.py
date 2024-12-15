# /// script
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn",
#   "requests",
#   "chardet",
#   "numpy",
#   "joblib",
#   "folium",
#   "plotly",
#   "Pillow",
#   "geopy",
#   "typing",
#   "argparse",
# ]
# ///

import os
import sys
import subprocess
def ensure_pip():
    """Ensures pip is installed."""
    try:
        import pip  # Check if pip is already available
    except ImportError:
        print("pip not found. Installing pip...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("pip installed successfully.")

# Call ensure_pip at the start of your script
ensure_pip()

# Function to install a Python package if not already installed
def install_package(package_name, submodules=None):
    """Installs a Python package if not already installed."""
    try:
        __import__(package_name)
        if submodules:
            for submodule in submodules:
                __import__(f"{package_name}.{submodule}")
    except ImportError:
        print(f"Package {package_name} or submodule {submodules} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def install_package(package_name):
    """Installs a Python package if not already installed."""
    try:
        __import__(package_name)
    except ImportError:
        print(f"Package {package_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Check and install dependencies
dependencies = [
    "pandas", "seaborn", "matplotlib", "scikit-learn", "requests", "chardet", 
    "numpy", "joblib", "folium", "plotly", "Pillow", "geopy"
]
for dep in dependencies:
    install_package(dep)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from geopy.distance import geodesic
import plotly.express as px
import argparse
from typing import List, Dict
import requests
import warnings

warnings.filterwarnings("ignore")

# === AI Proxy Token Validation ===
try:
    AI_PROXY_TOKEN = os.environ("AI_PROXY")
except KeyError:
    print("Error: AI_PROXY environment variable is not set.")
    sys.exit(1)

# === Utility Functions ===
def detect_encoding(filepath: str) -> str:
    """Detect file encoding using chardet."""
    try:
        import chardet
    except ImportError:
        print("Error: chardet is not installed.")
        sys.exit(1)

    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset with detected encoding."""
    encoding = detect_encoding(filepath)
    return pd.read_csv(filepath, encoding=encoding)

def save_plot(filename: str, fig=None):
    """Save the provided figure as a PNG file."""
    if fig is None:
        plt.savefig(filename, format="png", bbox_inches="tight")
    else:
        fig.savefig(filename, format="png", bbox_inches="tight")
    plt.close()

def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric columns for analysis."""
    return df.select_dtypes(include=[np.number])

# === Analysis Functions ===
def outlier_detection(df: pd.DataFrame, output_path: str):
    """Detect outliers using IQR method."""
    numeric_df = clean_numeric_data(df)
    if numeric_df.empty:
        return None  # No numeric columns to analyze

    # Compute the IQR (Interquartile Range)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR)))

    # Plotting the outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric_df, orient="h")
    plt.title("Outlier Detection (IQR)")
    save_plot(output_path)
    return output_path

def time_series_analysis(df, column, output_path):
    """
    Perform time series analysis on the given column and save the visualization.

    Args:
        df (pd.DataFrame): The dataset.
        column (str): The name of the time series column.
        output_path (str): The file path to save the visualization.

    Returns:
        str: Path to the saved visualization or None if the operation fails.
    """
    try:
        # Ensure the column is in datetime format
        df[column] = pd.to_datetime(df[column], errors="coerce")

        # Drop rows where datetime conversion failed
        df = df.dropna(subset=[column])

        # Set the column as index for analysis
        df = df.set_index(column).sort_index()

        # Example visualization: Plotting the number of records over time
        plt.figure(figsize=(10, 6))
        df.resample("M").size().plot(title="Records Over Time", xlabel="Date", ylabel="Count")
        plt.savefig(output_path)
        plt.close()

        return output_path
    except Exception as e:
        print(f"Error in time series analysis: {e}")
        return None

def geographic_analysis(df: pd.DataFrame, lat_col: str, lon_col: str, output_path: str):
    """Analyze geographic data if latitude and longitude columns are present."""
    if lat_col not in df.columns or lon_col not in df.columns:
        return None

    # Plotting the geographic data using Plotly
    fig = px.scatter_geo(df, lat=lat_col, lon=lon_col, title="Geographic Distribution")
    fig.write_image(output_path)
    return output_path

def correlation_heatmap(df: pd.DataFrame, output_path: str):
    """Generate and save a correlation heatmap of the numeric columns."""
    numeric_df = clean_numeric_data(df)
    if numeric_df.empty:
        return None  # No numeric columns to analyze

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    save_plot(output_path)
    return output_path

def regression_analysis(df: pd.DataFrame, output_path: str):
    """Perform regression analysis if a target column is suitable."""
    numeric_df = clean_numeric_data(df)
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None  # Not enough data for regression

    target_col = numeric_df.columns[-1]  # Assume the last numeric column as target
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    # Impute missing values in X and y
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Plot true vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Regression Analysis\nMSE: {mse:.2f}, R2: {r2:.2f}")
    save_plot(output_path)  # Ensure the plot is saved
    return output_path

def feature_importance_analysis(df: pd.DataFrame, output_path: str):
    """Perform feature importance analysis if no target column is clearly defined."""
    numeric_df = clean_numeric_data(df)
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None  # Not enough data for feature importance analysis

    target_col = numeric_df.columns[-1]  # Assume the last numeric column as target
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Fit a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Get feature importances
    feature_importances = pd.Series(model.feature_importances_, index=numeric_df.columns[:-1])
    feature_importances = feature_importances.sort_values(ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
    plt.title("Feature Importance Analysis")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    save_plot(output_path)  # Ensure the plot is saved
    return output_path

def auto_select_analysis(df: pd.DataFrame, output_path_prefix: str):
    """Automatically switch between Regression and Feature Importance Analysis."""
    numeric_df = clean_numeric_data(df)
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None

    target_col = numeric_df.columns[-1]  # Assume the last column is the target
    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    # If the target column has more than 10 unique values, assume regression
    if y.nunique() > 10:
        return regression_analysis(df, f"{output_path_prefix}_regression.png")
    else:
        return feature_importance_analysis(df, f"{output_path_prefix}_feature_importance.png")

# === Data Summarization Function ===
def summarize_data(df: pd.DataFrame) -> Dict[str, str]:
    """Summarize the dataset (e.g., missing values, summary stats)."""
    summary_stats = df.describe().transpose()
    missing_values = df.isnull().sum()

    summary = {
        "summary_stats": summary_stats.to_string(),
        "missing_values": missing_values.to_string(),
    }
    return summary

def detect_and_plot_outliers(df: pd.DataFrame, output_path: str):
    """Detect and plot outliers using KMeans."""
    numeric_df = clean_numeric_data(df)
    if numeric_df.empty:
        return None
    
    # Imputation
    imputer = SimpleImputer(strategy="mean")
    data_imputed = imputer.fit_transform(numeric_df)
    
    # Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    # KMeans Outlier Detection
    kmeans = KMeans(n_clusters=3, random_state=42).fit(data_scaled)
    distances = kmeans.transform(data_scaled).min(axis=1)
    threshold = distances.mean() + 3 * distances.std()
    outliers = distances > threshold

    # Plot outliers vs non-outliers
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.where(~outliers)[0], distances[~outliers], c='blue', label="Non-Outliers", alpha=0.6)
    ax.scatter(np.where(outliers)[0], distances[outliers], c='red', label="Outliers", alpha=0.6)
    ax.set_title("Outlier Detection (KMeans)", fontsize=16)
    ax.set_xlabel("Data Points", fontsize=12)
    ax.set_ylabel("Distance from Centroid", fontsize=12)
    ax.legend()

    save_plot(output_path, fig)
    return output_path

def iterative_agentic_analysis(df: pd.DataFrame, summary: Dict) -> str:
    """
    Generate a series of insights through an iterative process. This function will iterate over different 
    analysis types and summarize the findings based on the data.
    """
    insights = []  # Generate iterative insights
    if "correlation" in summary:
        insights.append("Correlation Analysis: Strong correlations found between ...")
    if "outliers" in summary:
        insights.append("Outlier Detection: Outliers identified in the following areas ...")
    if "regression" in summary:
        insights.append("Regression Analysis: The following regression results were found ...")

    return "\n".join(insights)

def suggest_visualizations(df: pd.DataFrame, summary: Dict) -> List[str]:
    """
    Suggest visualizations based on the provided summary and the data analysis results.
    """
    visualizations = []
    if "correlation" in summary:
        visualizations.append("correlation_heatmap.png")
    if "outliers" in summary:
        visualizations.append("outliers.png")
    if "regression" in summary:
        visualizations.append("regression_analysis.png")
    return visualizations

# === Reporting Functions ===
def create_markdown_report(story: str, visualizations: List[str], output_path: str):
    """Create README.md with links to PNGs."""
    with open(output_path, 'w') as f:
        f.write("# Analysis Report\n\n")
        f.write(story)
        f.write("\n## Visualizations\n")
        for viz in visualizations:
            if os.path.exists(viz):
                # Use relative path for visualization links
                relative_path = os.path.relpath(viz, start=os.path.dirname(output_path))
                f.write(f"![{viz}]({relative_path})\n")

# === Narrative Generation with AI Proxy ===
def generate_story(summary: Dict, visualizations: List[str]) -> str:
    """
    Generate a comprehensive narrative using the AI Proxy API.
    """
    summary_str = f"Summary: {summary['summary_stats']}\nMissing: {summary['missing_values']}"
    prompt = f"""
    Given the following data analysis results:
    {summary_str}
    And the visualizations generated:
    - Outlier Detection
    - Correlation Heatmap
    - PCA Clustering
    - Time Series Analysis (if present)
    - Geographic Analysis (if present)

    Provide a comprehensive narrative with key findings and insights.
    """
    headers = {"Authorization": f"Bearer {AI_PROXY_TOKEN}", "Content-Type": "application/json"}
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],"temperature": 0.0, "top_p": 1.0, "max_tokens": 1500},
        headers=headers,
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Failed to generate story. Error {response.status_code}: {response.text}"

# === Main Execution ===
# Main Execution
def main(input_file: str, output_folder: str = None):
    """
    Main function to process the dataset, perform analyses, and generate a report.

    Args:
        input_file (str): Path to the input dataset file.
        output_folder (str, optional): Directory to save outputs. Defaults to current directory.
    """
    df = load_data(input_file)

    # Set output folder to current directory if not provided
    if output_folder is None:
        output_folder = os.getcwd()

    os.makedirs(output_folder, exist_ok=True)

    # Generate summary for AI Proxy story generation
    summary = summarize_data(df)

    # Iterative analysis of the dataset
    insights = iterative_agentic_analysis(df, summary)

    # Suggest visualizations based on the analysis summary
    visualizations = suggest_visualizations(df, summary)

    # Correlation heatmap
    heatmap_output = correlation_heatmap(df, f"{output_folder}/correlation_heatmap.png")
    if heatmap_output:
        visualizations.append(heatmap_output)

    # Outlier detection and visualization
    outlier_output = detect_and_plot_outliers(df, f"{output_folder}/outliers.png")
    if outlier_output:
        visualizations.append(outlier_output)

    # Identify potential time series column
    time_series_column = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or any(keyword in col.lower() for keyword in ["date", "time", "timestamp"]):
            time_series_column = col
            break

    # Time series analysis if a suitable column is found
    if time_series_column:
        print(f"Time series column detected: {time_series_column}")
        time_series_output = time_series_analysis(df, time_series_column, f"{output_folder}/time_series.png")
        if time_series_output:
            visualizations.append(time_series_output)
    else:
        print("No suitable time series column found. Skipping time series analysis.")

    # Auto-analysis for regression/feature importance
    auto_analysis_output = auto_select_analysis(df, f"{output_folder}/auto_analysis")
    if auto_analysis_output:
        visualizations.append(auto_analysis_output)

    # Generate markdown report with insights and visualizations
    story = generate_story(summary, visualizations)
    create_markdown_report(story, visualizations, f"{output_folder}/README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Analysis Pipeline")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_dir", nargs="?", default=os.getcwd(), help="Directory to save the output files (default is current directory)")

    args = parser.parse_args()

    main(args.input_file, args.output_dir)
