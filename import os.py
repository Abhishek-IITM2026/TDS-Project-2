import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import json

# Ensure AIPROXY_TOKEN is set in the environment variables
if "AIPROXY_TOKEN" not in os.environ:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

AIPROXY_TOKEN = os.environ["eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDE4MzhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.AFKvBEmzZ37v6spjQbLVhW4WtaWij6GUZKQDZxt1Ayc"]
openai.api_key = AIPROXY_TOKEN

def load_dataset(file_path):
    """Load CSV file into a Pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def basic_analysis(data):
    """Perform basic statistical analysis."""
    summary = data.describe(include='all').transpose()
    missing_values = data.isnull().sum()
    return summary, missing_values

def generate_visualizations(data):
    """Generate visualizations from the dataset."""
    visualizations = []
    if data.select_dtypes(include=["float64", "int64"]).shape[1] > 0:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        heatmap_file = "correlation_heatmap.png"
        plt.savefig(heatmap_file)
        plt.close()
        visualizations.append(heatmap_file)

    # Distribution plots for numerical columns
    num_cols = data.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribution of {col}")
        dist_file = f"distribution_{col}.png"
        plt.savefig(dist_file)
        plt.close()
        visualizations.append(dist_file)
    
    return visualizations

def query_llm(prompt):
    """Query GPT-4o-Mini through OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None

def narrate_story(data, summary, missing_values, visuals):
    """Generate a narrative from the analysis."""
    columns_info = json.dumps(data.dtypes.to_dict())
    summary_info = summary.to_string()
    missing_info = missing_values.to_string()
    visuals_info = "\n".join(visuals)
    
    prompt = (
        f"The dataset has the following columns and types:\n{columns_info}\n\n"
        f"Summary statistics:\n{summary_info}\n\n"
        f"Missing values:\n{missing_info}\n\n"
        f"The following charts were generated:\n{visuals_info}\n\n"
        "Write a story about this data. Mention key insights, findings, and potential implications."
    )
    return query_llm(prompt)

def write_readme(story, visuals):
    """Write the story and visuals to README.md."""
    with open("README.md", "w") as f:
        f.write("# Analysis Report\n\n")
        f.write(story)
        f.write("\n\n## Visualizations\n")
        for visual in visuals:
            f.write(f"![{visual}]({visual})\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = load_dataset(file_path)
    
    summary, missing_values = basic_analysis(data)
    visuals = generate_visualizations(data)
    
    story = narrate_story(data, summary, missing_values, visuals)
    if story:
        write_readme(story, visuals)
        print("Analysis completed. README.md and visualizations generated.")
    else:
        print("Error generating story from LLM.")

if __name__ == "__main__":
    main()
