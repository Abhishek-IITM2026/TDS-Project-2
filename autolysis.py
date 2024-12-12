# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai==0.28",
#   "chardet",
#   "scipy",
# ]
# ///

import sys
import os
import pandas as pd
import seaborn as sns
import openai
import json
from io import StringIO
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
from scipy import stats

# Check for the presence of the AIPROXY_TOKEN environment variable.
if "AIPROXY_TOKEN" not in os.environ:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
openai.api_key = AIPROXY_TOKEN
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

# Function to search for a file in the current directory or its subdirectories.
def find_file_in_subdirectories(filename, start_dir="."):
    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Load dataset with robust encoding handling.
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("Encoding issue. Attempting conversion...")
        try:
            with open(file_path, "rb") as file:
                raw_data = file.read().decode("latin1")
            data = pd.read_csv(StringIO(raw_data))
        except Exception as e:
            print(f"Error while converting file encoding: {e}")
            sys.exit(1)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

# Perform basic statistical analysis on the dataset.
def basic_analysis(data):
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

        analysis_results = {
            "Summary Statistics": summary,
            "Missing Values": missing_values,
            "Missing Percentage": missing_percentage,
            "Correlation Matrix": correlation_matrix,
            "Outliers": outliers,
            "Category Analysis": category_analysis
        }

        return analysis_results
    except Exception as e:
        print(f"Error while performing advanced analysis: {e}")
        sys.exit(1)

# Generate visualizations and save them as files.
def generate_visualizations(data, output_dir):
    visualizations = []
    try:
        numeric_data = data.select_dtypes(include=["number"])

        if not numeric_data.empty:
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

            if numeric_data.shape[1] <= 10:  
                scatter_matrix_file = os.path.join(output_dir, "scatter_matrix.png")
                scatter_matrix(numeric_data, figsize=(16, 16), diagonal="hist", alpha=0.9)
                plt.suptitle("Pairwise Relationships - Scatter Plot Matrix", fontsize=18, weight='bold')
                plt.subplots_adjust(top=0.93)  
                plt.tight_layout()
                plt.savefig(scatter_matrix_file)
                plt.close()
                visualizations.append(scatter_matrix_file)

            combined_boxplot_file = os.path.join(output_dir, "combined_boxplot.png")
            plt.figure(figsize=(16, 8))
            sns.boxplot(data=numeric_data, orient="h", palette="Set3", linewidth=2)
            plt.title("Box Plot - All Numeric Columns", fontsize=18, weight='bold')
            plt.xlabel("Values", fontsize=14)
            plt.ylabel("Columns", fontsize=14)
            plt.grid(True, axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(combined_boxplot_file)
            plt.close()
            visualizations.append(combined_boxplot_file)

        datetime_cols = data.select_dtypes(include=["datetime", "datetimetz"])
        if not datetime_cols.empty:
            for col in datetime_cols.columns:
                time_series_file = os.path.join(output_dir, f"time_series_{col}.png")
                plt.figure(figsize=(14, 7))
                for num_col in numeric_data.columns:
                    plt.plot(data[col], numeric_data[num_col], label=num_col, marker='o', markersize=5, linestyle='-', alpha=0.7)
                    moving_avg = numeric_data[num_col].rolling(window=7).mean()
                    plt.plot(data[col], moving_avg, label=f'{num_col} (Moving Average)', linestyle='--', linewidth=2)
                plt.title(f"Time Series - {col}", fontsize=18, weight='bold')
                plt.xlabel(col, fontsize=14)
                plt.ylabel("Values", fontsize=14)
                plt.legend(title="Variables", fontsize=12, loc='upper left')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
                plt.tight_layout()
                plt.savefig(time_series_file)
                plt.close()
                visualizations.append(time_series_file)
    except Exception as e:
        print(f"Error in generating visualizations: {e}")
    return visualizations

# Query OpenAI for insights.
def query_llm(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly skilled data scientist summarizing insights from datasets.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"]
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

# Generate the narrative for the dataset.
def narrate_story(data, analysis_results, visuals):
    try:
        columns_info = json.dumps({col: str(dtype) for col, dtype in data.dtypes.items()}, indent=2)
        summary_info = analysis_results["Summary Statistics"].to_string()
        missing_info = analysis_results["Missing Values"].to_string()
        visuals_info = "\n".join(visuals)

        prompt = (
            "Summarize the dataset findings using the following context:"
            "\n\n### Dataset Information:\n"
            f"{columns_info}\n\n"
            f"### Summary Statistics:\n{summary_info}\n\n"
            f"### Missing Values:\n{missing_info}\n\n"
            f"### Visualizations:\n{visuals_info}\n\n"
            "Provide a concise summary with trends and recommendations."
        )
        return query_llm(prompt)
    except Exception as e:
        print(f"Error in generating story: {e}")
        return None

# Write the analysis and visualizations to a README file.
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

# Main function to execute the analysis and generate the report.
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    file_path = find_file_in_subdirectories(filename)
    if file_path is None:
        print(f"Error: File '{filename}' not found in the current directory or its subdirectories.")
        sys.exit(1)

    print(f"File found at: {file_path}")
    output_dir = os.path.dirname(file_path)
    data = load_dataset(file_path)

    analysis_results = basic_analysis(data)
    visuals = generate_visualizations(data, output_dir)

    story = narrate_story(data, analysis_results, visuals)
    if story:
        write_readme(story, visuals, output_dir)
    else:
        print("Error: Could not generate the analysis report.")

if __name__ == "__main__":
    main()
