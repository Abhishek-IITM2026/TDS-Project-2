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
import logging
import pandas as pd
import seaborn as sns
import openai
import json
from io import StringIO
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure that OpenAI API key is set correctly
if "AIPROXY_TOKEN" not in os.environ:
    logging.error("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
openai.api_key = AIPROXY_TOKEN
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

def find_file_in_subdirectories(filename, start_dir="."):
    """Recursively search for a file in the specified directory and subdirectories."""
    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_dataset(file_path):
    """Load the dataset and handle encoding issues."""
    try:
        data = pd.read_csv(file_path)
    except UnicodeDecodeError:
        logging.warning("Encoding issue detected. Attempting to convert to UTF-8...")
        try:
            with open(file_path, "rb") as file:
                raw_data = file.read().decode("latin1")  # Handle encoding issues
            data = pd.read_csv(StringIO(raw_data))
        except Exception as e:
            logging.error(f"Error in converting file to UTF-8: {e}")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)
    logging.info(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

def basic_analysis(data):
    """Perform advanced statistical analysis on the dataset."""
    try:
        # Summary statistics for all columns
        summary = data.describe(include="all").transpose()
        # Missing values analysis
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        # Correlation matrix for numeric columns
        numeric_data = data.select_dtypes(include=["number"])
        correlation_matrix = numeric_data.corr()
        # Outlier detection using IQR
        outliers = {}
        for column in numeric_data.columns:
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = numeric_data[(numeric_data[column] < (Q1 - 1.5 * IQR)) | 
                                         (numeric_data[column] > (Q3 + 1.5 * IQR))][column].count()
            outliers[column] = outlier_count
        # Categorical column analysis
        categorical_data = data.select_dtypes(include=["object", "category"])
        category_analysis = {col: data[col].value_counts(normalize=True) * 100 
                             for col in categorical_data.columns}
        # Prepare the analysis results
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
        logging.error(f"Error while performing advanced analysis: {e}")
        sys.exit(1)

def generate_visualizations(data, output_dir):
    """Generate visualizations to help with data interpretation."""
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
            # Scatter Plot Matrix (if there are fewer than 10 numeric columns)
            if numeric_data.shape[1] <= 10:  
                scatter_matrix_file = os.path.join(output_dir, "scatter_matrix.png")
                scatter_matrix(numeric_data, figsize=(16, 16), diagonal="hist", alpha=0.9)
                plt.suptitle("Pairwise Relationships - Scatter Plot Matrix", fontsize=18, weight='bold')
                plt.subplots_adjust(top=0.93)  
                plt.tight_layout()
                plt.savefig(scatter_matrix_file)
                plt.close()
                visualizations.append(scatter_matrix_file)
        # Box Plot for Outlier Detection
        boxplot_file = os.path.join(output_dir, "boxplot_numeric.png")
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=numeric_data, orient="h", palette="Set3")
        plt.title("Boxplot for Numeric Data", fontsize=16)
        plt.tight_layout()
        plt.savefig(boxplot_file)
        plt.close()
        visualizations.append(boxplot_file)
    except Exception as e:
        logging.error(f"Error in generating visualizations: {e}")
    return visualizations

def query_llm(prompt, max_tokens=1500):
    """Query OpenAI's LLM to generate an insightful report with a focus on token efficiency."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a highly skilled data scientist summarizing insights from datasets."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens  # Limit the tokens used in the response
        )
        return response["choices"][0]["message"]["content"]
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    return None


def narrate_story(data, summary, missing_values, visuals):
    """Generate a detailed narrative for the dataset analysis."""
    try:
        columns_info = json.dumps({col: str(dtype) for col, dtype in data.dtypes.items()}, indent=2)
        summary_info = summary.to_string()
        missing_info = missing_values.to_string()
        visuals_info = "\n".join(visuals)
        prompt = (
            "You are a highly skilled data scientist tasked with creating a comprehensive analysis report for the dataset. "
            "Your goal is to present findings in a clear, engaging, and insightful manner. Use the information below to write a detailed README.md file. "
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
    except Exception as e:
        logging.error(f"Error in generating story: {e}")
        return None

def write_readme(story, visuals, output_dir):
    """Write the generated analysis and visualizations to a README file."""
    try:
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("# Analysis Report\n\n")
            f.write(story)
            f.write("\n\n## Visualizations\n")
            for visual in visuals:
                f.write(f"![{os.path.basename(visual)}]({os.path.basename(visual)})\n")
        logging.info(f"README.md saved in {output_dir}")
    except Exception as e:
        logging.error(f"Error writing README.md: {e}")

def main():
    """Main function to execute the analysis and generate the report."""
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    filename = sys.argv[1]
    file_path = find_file_in_sub
    file_path = find_file_in_subdirectories(filename)
    if file_path is None:
        logging.error(f"Error: File '{filename}' not found in the current directory or its subdirectories.")
        sys.exit(1)
    logging.info(f"File found at: {file_path}")
    output_dir = os.path.dirname(file_path)
    data = load_dataset(file_path)
    analysis_results = basic_analysis(data)
    visuals = generate_visualizations(data, output_dir)
    story = narrate_story(data, analysis_results["Summary Statistics"], analysis_results["Missing Values"], visuals)
    if story:
        write_readme(story, visuals, output_dir)
    else:
        logging.error("Error: Could not generate the analysis report.")

if __name__ == "__main__":
    main()
