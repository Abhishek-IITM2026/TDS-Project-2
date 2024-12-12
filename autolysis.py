import os
import sys
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
import chardet
import openai
from io import StringIO
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from joblib.externals.loky.backend.context import set_start_method
import requests

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

# Check and install dependencies
required_packages = [
    ("pandas", None),
    ("seaborn", None),
    ("matplotlib", None),
    ("scikit-learn", None),
    ("requests", None),
    ("chardet", None),
    ("joblib", ["externals.loky.backend.context"]),
    ("warnings", None),
    ("numpy", None)
]
for package, submodules in required_packages:
    install_package(package, submodules)

# Set up environment and validate dependencies
try:
    AI_PROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
except KeyError:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

openai.api_key = AI_PROXY_TOKEN
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=Warning, module="subprocess")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")

# Suppress joblib warnings and set start method
set_start_method("loky", force=True)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*subprocess.*", category=Warning)

# Function to detect encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# Function to load a dataset
def load_dataset(file_path):
    """
    Loads a dataset from the specified file path with robust error handling for encoding issues.
    """
    encoding = detect_encoding(file_path)
    try:
        data = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    return data

# Basic statistical analysis
def basic_analysis(data):
    """
    Perform basic statistical analysis and return results.
    """
    try:
        summary = data.describe(include="all").transpose()
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        numeric_data = data.select_dtypes(include=["number"])
        correlation_matrix = numeric_data.corr()
        outliers = {}

        for column in numeric_data.columns:
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = numeric_data[(numeric_data[column] < (Q1 - 1.5 * IQR)) |
                                         (numeric_data[column] > (Q3 + 1.5 * IQR))][column].count()
            outliers[column] = outlier_count

        categorical_data = data.select_dtypes(include=["object", "category"])
        category_analysis = {col: data[col].value_counts(normalize=True) * 100 
                             for col in categorical_data.columns}

        return {
            "Summary Statistics": summary,
            "Missing Values": missing_values,
            "Missing Percentage": missing_percentage,
            "Correlation Matrix": correlation_matrix,
            "Outliers": outliers,
            "Category Analysis": category_analysis
        }
    except Exception as e:
        print(f"Error while performing advanced analysis: {e}")
        sys.exit(1)

# Generate visualizations
def generate_visualizations(data, output_dir):
    """
    Generate visualizations for the dataset.
    """
    visualizations = []
    try:
        numeric_data = data.select_dtypes(include=["number"])

        if not numeric_data.empty:
            # Correlation Heatmap
            heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
            plt.figure(figsize=(14, 12))
            corr = numeric_data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", mask=mask,
                        linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                        annot_kws={"size": 10, "weight": 'bold'})
            plt.title("Correlation Heatmap", fontsize=18, weight='bold')
            plt.tight_layout()
            plt.savefig(heatmap_file)
            plt.close()
            visualizations.append(heatmap_file)

            # PCA for dimensionality reduction
            pca = PCA(n_components=2)
            data_scaled = StandardScaler().fit_transform(data.select_dtypes(include=["float64", "int64"]))
            pca_components = pca.fit_transform(data_scaled)

            pca_plot = os.path.join(output_dir, "pca_scatter.png")
            plt.scatter(pca_components[:, 0], pca_components[:, 1], alpha=0.5, c="blue")
            plt.title("PCA Scatter Plot")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.tight_layout()
            plt.savefig(pca_plot)
            plt.close()
            visualizations.append(pca_plot)

            # KMeans Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(data_scaled)
            data["Cluster"] = clusters
            clustering_plot = os.path.join(output_dir, "kmeans_clustering.png")
            sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=data["Cluster"], palette="viridis")
            plt.title("KMeans Clustering")
            plt.tight_layout()
            plt.savefig(clustering_plot)
            plt.close()
            visualizations.append(clustering_plot)

        # Box plot
        boxplot_file = os.path.join(output_dir, "boxplot.png")
        sns.boxplot(data=numeric_data)
        plt.title("Boxplot of Numeric Columns")
        plt.tight_layout()
        plt.savefig(boxplot_file)
        plt.close()
        visualizations.append(boxplot_file)

    except Exception as e:
        print(f"Error in generating visualizations: {e}")
    return visualizations

# Query OpenAI for insights
def query_llm(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a highly skilled data scientist summarizing insights from datasets."},
                {"role": "user", "content": prompt},
            ]
        )
        return response["choices"][0]["message"]["content"]
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

# Generate the narrative for the dataset
def narrate_story(data, summary, missing_values, visuals):
    columns_info = json.dumps({col: str(dtype) for col, dtype in data.dtypes.items()}, indent=2)
    summary_info = summary.to_string()
    missing_info = missing_values.to_string()
    visuals_info = "\n".join(visuals)

    prompt = (
        "You are a highly skilled data scientist tasked with creating a comprehensive analysis report for the dataset."
        "Your goal is to present findings in a clear, engaging, and insightful manner. Use the information below to write a detailed README.md file."
        "The report should include sections like overview, insights, trends, notable statistics, and potential applications.\n\n"
        "Here is the context:\n\n"
        f"### Dataset Information\n"
        f"The dataset contains the following columns and data types:\n{columns_info}\n\n"
        f"### Summary Statistics\n"
        f"{summary_info}\n\n"
        f"### Missing Values\n"
        f"{missing_info}\n\n"
        f"### Visualizations\n"
        f"The following visualizations were generated:\n{visuals_info}\n\n"
        "Conclude with recommendations for further analysis and actionable insights."
    )
    return query_llm(prompt)

# Write the analysis and visualizations to a README file
def write_readme(story, visuals, output_dir):
    try:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("# Analysis Report\n\n")
            f.write(story)
            f.write("\n\n## Visualizations\n")
            for visual in visuals:
                f.write(f"![{os.path.basename(visual)}]({os.path.basename(visual)})\n")
        print(f"README.md saved in {output_dir}")
    except Exception as e:
        print(f"Error writing README.md: {e}")

# Main function to execute the analysis and generate the report
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    if file_path is None:
        print(f"Error: File '{filename}' not found in the current directory or its subdirectories.")
        sys.exit(1)

    print(f"File found at: {file_path}")
    output_dir = os.path.dirname(file_path)
    data = load_dataset(file_path)

    summary, missing_values = basic_analysis(data)
    visuals = generate_visualizations(data, output_dir)

    story = narrate_story(data, summary, missing_values, visuals)
    if story:
        write_readme(story, visuals, output_dir)
    else:
        print("Error: Could not generate the analysis report.")

if __name__ == "__main__":
    main()
