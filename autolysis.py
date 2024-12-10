import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import json
from io import StringIO

if "AIPROXY_TOKEN" not in os.environ:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
openai.api_key = AIPROXY_TOKEN
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
    except UnicodeDecodeError:
        print("Encoding issue. Converting to UTF-8...")
        try:
            # Read the file in binary mode and decode to UTF-8
            with open(file_path, "rb") as file:
                raw_data = file.read().decode("latin1")  # Replace with correct source encoding if known
            data = pd.read_csv(StringIO(raw_data))
        except Exception as e:
            print(f"Error in converting file to UTF-8: {e}")
            sys.exit(1)
    print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data

def basic_analysis(data):
    try:
        summary = data.describe(include="all").transpose()
        missing_values = data.isnull().sum()
        return summary, missing_values
    except Exception as e:
        print(f"Error while performing basic analysis: {e}")
        sys.exit(1)

def generate_visualizations(data):
    visualizations = []
    try:
        numeric_data = data.select_dtypes(include=["number"])
        if not numeric_data.empty:
            plt.figure(figsize=(10, 8))
            corr = numeric_data.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            heatmap_file = "correlation_heatmap.png"
            plt.savefig(heatmap_file)
            plt.close()
            visualizations.append(heatmap_file)
            for col in numeric_data.columns:
                plt.figure(figsize=(6, 4))
                sns.histplot(numeric_data[col], kde=True)
                plt.title(f"Distribution of {col}")
                dist_file = f"distribution_{col}.png"
                plt.savefig(dist_file)
                plt.close()
                visualizations.append(dist_file)
        else:
            print("No numeric data available for correlation or distribution plots.")
    except Exception as e:
        print(f"Error in generating visualizations: {e}")
    return visualizations


def query_llm(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are the best data analyst summarizing insights from datasets.",
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


def narrate_story(data, summary, missing_values, visuals):
    try:
        # Convert dtypes to a JSON-serializable format
        columns_info = json.dumps({col: str(dtype) for col, dtype in data.dtypes.items()}, indent=2)
        summary_info = summary.to_string()
        missing_info = missing_values.to_string()
        visuals_info = "\n".join(visuals)

        prompt = (
            "You are a best skilled data scientist with the best vocabulary tasked with creating a very much interesting and attractive comprehensive analysis report story for the dataset."
            "The goal is to present findings in a clear, engaging, attractive, creative, professional and insightful way. Use the information below to "
            "write a very much detailed README.md file. Include sections for an overview, insights from the data, any trends or patterns, "
            "notable statistics, potential applications, and challenges. Conclude with next steps for deeper analysis. It should elaborately explain very much clearly so that every person can understand.\n\n"
            "Here is the context:\n\n"
            f"### Dataset Information\n"
            f"The dataset contains the following columns and data types:\n{columns_info}\n\n"
            f"### Summary Statistics\n"
            f"{summary_info}\n\n"
            f"### Missing Values\n"
            f"{missing_info}\n\n"
            f"### Visualizations\n"
            f"The following visualizations were generated:\n{visuals_info}\n\n"
            "Use this information to create a story highlighting the key insights, any challenges in the data, "
            "and actionable recommendations. Make the README informative, professional, and easy to understand."
        )
        return query_llm(prompt)
    except Exception as e:
        print(f"Error in generating story: {e}")
        return None


def write_readme(story, visuals):
    try:
        with open("README.md", "w") as f:
            f.write("# Analysis Report\n\n")
            f.write(story)
            f.write("\n\n## Visualizations\n")
            for visual in visuals:
                f.write(f"![{visual}]({visual})\n")
    except Exception as e:
        print(f"Error writing README.md: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
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