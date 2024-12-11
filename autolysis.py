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

def construct_dynamic_prompt(data, summary, missing_values, outliers, correlation_matrix, visuals):
    """Construct a dynamic LLM prompt based on dataset characteristics."""
    num_rows, num_cols = data.shape
    missing_percentages = (missing_values / len(data)) * 100
    high_missing_columns = missing_percentages[missing_percentages > 20].index.tolist()

    # Detect high correlations
    high_correlation_pairs = [
        (i, j, correlation_matrix.loc[i, j])
        for i in correlation_matrix.index
        for j in correlation_matrix.columns
        if i != j and abs(correlation_matrix.loc[i, j]) > 0.8
    ]

    # Summarize outliers
    high_outlier_columns = [col for col, count in outliers.items() if count > 10]

    prompt = (
        f"The dataset contains {num_rows} rows and {num_cols} columns.\n\n"
        f"### Missing Values\n"
        f"Columns with more than 20% missing values: {', '.join(high_missing_columns) if high_missing_columns else 'None'}.\n\n"
        f"### Correlation Insights\n"
        f"There {'are' if high_correlation_pairs else 'are no'} highly correlated variable pairs.\n"
    )
    if high_correlation_pairs:
        for pair in high_correlation_pairs[:5]:  # Limit to top 5 pairs for clarity
            prompt += f"- {pair[0]} and {pair[1]}: correlation = {pair[2]:.2f}\n"

    prompt += (
        f"\n### Outliers\n"
        f"Columns with significant outliers: {', '.join(high_outlier_columns) if high_outlier_columns else 'None'}.\n\n"
        f"### Generated Visualizations\n"
        f"{len(visuals)} visualizations were created, including a heatmap and boxplots.\n\n"
        "Use this information to write a comprehensive analysis report highlighting insights, trends, and recommendations."
    )
    return prompt
def agentic_workflow(data, summary, missing_values, outliers, correlation_matrix, visuals):
    """Execute a multi-step interaction with the LLM."""
    # Step 1: Generate a high-level summary
    initial_prompt = construct_dynamic_prompt(data, summary, missing_values, outliers, correlation_matrix, visuals)
    high_level_insights = query_llm(initial_prompt)
    logging.info("Generated high-level insights.")

    # Step 2: Focused analysis on missing values
    if not missing_values.empty and missing_values.sum() > 0:
        missing_prompt = (
            f"Analyze the following missing value information in detail and suggest imputation strategies:\n"
            f"{missing_values.to_string()}"
        )
        missing_analysis = query_llm(missing_prompt)
        high_level_insights += f"\n\n### Missing Values Analysis\n{missing_analysis}"

    # Step 3: Focused analysis on outliers
    if outliers:
        outlier_prompt = (
            f"Analyze the detected outliers in the numeric columns. Suggest methods to handle them:\n"
            f"{json.dumps(outliers, indent=2)}"
        )
        outlier_analysis = query_llm(outlier_prompt)
        high_level_insights += f"\n\n### Outlier Analysis\n{outlier_analysis}"

    # Combine all insights
    return high_level_insights

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

def query_llm(prompt):
    """Query OpenAI's LLM to generate an insightful report based on the analysis."""
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
    file_path = find_file_in_subdirectories(filename)
    if not file_path:
        logging.error(f"File '{filename}' not found.")
        sys.exit(1)

    # Load and analyze the dataset
    data = load_dataset(file_path)
    analysis_results = basic_analysis(data)
    summary = analysis_results["Summary Statistics"]
    missing_values = analysis_results["Missing Values"]
    outliers = analysis_results["Outliers"]
    correlation_matrix = analysis_results["Correlation Matrix"]

    # Generate visualizations
    output_dir = os.path.dirname(file_path)
    visuals = generate_visualizations(data, output_dir)

    # Execute agentic workflow
    final_insights = agentic_workflow(data, summary, missing_values, outliers, correlation_matrix, visuals)

    # Write the analysis to a README
    write_readme(final_insights, visuals, output_dir)


if __name__ == "__main__":
    main()
