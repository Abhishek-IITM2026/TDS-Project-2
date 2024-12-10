import os
import sys
import subprocess
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

def find_file_in_subdirectories(filename, start_dir="."):
    """Recursively search for a file in the specified directory and subdirectories."""
    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def install_requirements(start_dir="."):
    """Install libraries from requirements.txt found in any subdirectory."""
    requirements_file = find_file_in_subdirectories("requirements.txt", start_dir)
    if requirements_file:
        print(f"Found 'requirements.txt' at {requirements_file}. Installing required libraries...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Python libraries installation successful!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing libraries: {e}")
            sys.exit(1)
    else:
        print("No 'requirements.txt' file found.")

def install_astral():
    """Install Astral using the curl command."""
    try:
        print("Installing Astral tool using the curl command...")
        subprocess.check_call(['curl', '-LsSf', 'https://astral.sh/uv/install.sh', '|', 'sh'])
        print("Astral installation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Astral: {e}")
        sys.exit(1)

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

def generate_visualizations(data, output_dir):
    visualizations = []
    try:
        numeric_data = data.select_dtypes(include=["number"])
        if not numeric_data.empty:
            # Generate correlation heatmap
            plt.figure(figsize=(10, 8))
            corr = numeric_data.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_file)
            plt.close()
            visualizations.append(heatmap_file)

            # Generate distribution plot for the first numeric column
            first_col = numeric_data.columns[0]
            plt.figure(figsize=(6, 4))
            sns.histplot(numeric_data[first_col], kde=True)
            plt.title(f"Distribution of {first_col}")
            dist_file = os.path.join(output_dir, f"distribution_{first_col}.png")
            plt.savefig(dist_file)
            plt.close()
            visualizations.append(dist_file)

            # Generate box plot for numeric data
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=numeric_data, orient="h")
            plt.title("Box Plot of Numeric Data")
            boxplot_file = os.path.join(output_dir, "boxplot_numeric_data.png")
            plt.savefig(boxplot_file)
            plt.close()
            visualizations.append(boxplot_file)
        else:
            print("No numeric data available for visualizations.")
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

def main():
    # Install required libraries from requirements.txt if the file exists
    install_requirements()

    # Install Astral tool using curl
    install_astral()

    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]

    # Search for the file in subdirectories
    file_path = find_file_in_subdirectories(filename)
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
        print("Analysis completed. README.md and visualizations generated.")
    else:
        print("Error generating story from LLM.")

if __name__ == "__main__":
    main()
