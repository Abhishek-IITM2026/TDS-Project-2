import os
import sys
import subprocess

def ensure_pip_installed():
    """Ensure that pip is installed in the environment."""
    try:
        import pip  # Check if pip is available
    except ImportError:
        print("pip not found. Attempting to install pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            print("pip successfully installed and upgraded.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install pip. Error: {e}")
            sys.exit(1)

def install_package(package_name, submodules=None):
    """Installs a Python package if not already installed."""
    try:
        __import__(package_name)
        if submodules:
            for submodule in submodules:
                __import__(f"{package_name}.{submodule}")
    except ImportError:
        print(f"Package {package_name} or submodule {submodules} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Successfully installed {package_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}. Error: {e}")
            sys.exit(1)

# Ensure pip is available
ensure_pip_installed()

# Example usage
required_packages = [
    ("pandas", None),
    ("seaborn", None),
    ("matplotlib", None),
    ("scikit-learn", None),
    ("requests", None),
    ("chardet", None),
    ("numpy", None),
    ("joblib", ["externals.loky.backend.context"]),
]

for package, submodules in required_packages:
    install_package(package, submodules)

# Validate and retrieve the AI Proxy Token
try:
    AI_PROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
except KeyError:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# Validate command-line arguments
if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' does not exist.")
    sys.exit(1)

# Import dependencies after ensuring they are installed
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from joblib.externals.loky.backend.context import set_start_method
import chardet
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=Warning, module="subprocess")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")

# Suppress joblib warnings and set start method
set_start_method("loky", force=True)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*subprocess.*", category=Warning)

# Additional suppression of subprocess warnings globally
os.environ["PYTHONWARNINGS"] = "ignore"

# Function to detect encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

# Detect the encoding of the CSV file
encoding = detect_encoding(csv_file)

# Load the dataset using the detected encoding
try:
    data = pd.read_csv(csv_file, encoding=encoding)
except Exception as e:
    print(f"Error loading CSV file: {e}")
    sys.exit(1)

# Generic analysis
summary_stats = data.describe(include="all").transpose()
missing_values = data.isnull().sum()
correlation_matrix = data.corr(numeric_only=True)

# Ensure the directory exists for saving the README.md and PNG files (use current working directory)
readme_dir = os.getcwd()

# Visualization - Save correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
correlation_plot = os.path.join(readme_dir, "correlation_matrix.png")
plt.savefig(correlation_plot)
plt.close()
print(f"Saved correlation heatmap as {correlation_plot}")

# Impute missing values (replace NaNs with the mean of the column)
imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data.select_dtypes(include=["float64", "int64"])))

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Get feature names for PCA components
features = data.select_dtypes(include=["float64", "int64"]).columns

# PCA Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, c="blue", label="Data Points")
plt.title("PCA Scatter Plot")
plt.xlabel(f"Principal Component 1 ({features[0]})")
plt.ylabel(f"Principal Component 2 ({features[1]})")
plt.legend()
plt.tight_layout()
pca_plot = os.path.join(readme_dir, "pca_scatter.png")
plt.savefig(pca_plot)
plt.close()
print(f"Saved PCA scatter plot as {pca_plot}")

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(data_scaled)

# KMeans Clustering Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data["Cluster"], palette="viridis", legend="full")
plt.title("KMeans Clustering")
plt.xlabel(f"Principal Component 1 ({features[0]})")
plt.ylabel(f"Principal Component 2 ({features[1]})")
plt.legend(title="Clusters")
plt.tight_layout()
clustering_plot = os.path.join(readme_dir, "kmeans_clustering.png")
plt.savefig(clustering_plot)
plt.close()
print(f"Saved KMeans clustering plot as {clustering_plot}")

# Outlier Detection
distances = kmeans.transform(data_scaled).min(axis=1)
threshold = distances.mean() + 3 * distances.std()
outliers = distances > threshold

plt.figure(figsize=(8, 6))
# Plot non-outliers
plt.scatter(
    np.where(~outliers)[0], distances[~outliers],
    c='blue', label="Non-Outliers"
)
# Plot outliers
plt.scatter(
    np.where(outliers)[0], distances[outliers],
    c='red', label="Outliers"
)

# Add the threshold line
plt.axhline(y=threshold, color="green", linestyle="--", label="Threshold")
plt.title("Outlier Detection")
plt.xlabel("Data Point Index")
plt.ylabel("Distance to Closest Cluster")
plt.legend()
plt.tight_layout()

outliers_plot = os.path.join(readme_dir, "outliers.png")
plt.savefig(outliers_plot)
plt.close()
print(f"Saved outliers plot as {outliers_plot}")

# Use AI Proxy for story generation
import requests

def generate_story(analysis_summary, charts):
    prompt = f"""
    Analyze the data and narrate a story:
    - Dataset summary: {analysis_summary}
    - Generated charts: {charts}
    - Provide insights into:
      1. What the data reveals
      2. The analysis performed
      3. Key findings and implications
      4. Conclusion
    """
    headers = {
        "Authorization": f"Bearer {AI_PROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500
    }
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error communicating with AI Proxy: {response.status_code} - {response.text}")
        sys.exit(1)

# Prepare inputs for story generation
analysis_summary = {
    "Summary Statistics": summary_stats.to_dict(),
    "Missing Values": missing_values.to_dict()
}
charts = {
    "Correlation Heatmap": correlation_plot,
    "PCA Scatter Plot": pca_plot,
    "KMeans Clustering": clustering_plot,
    "Outliers Plot": outliers_plot
}

# Generate and save the story
story = generate_story(analysis_summary, charts)

with open(os.path.join(readme_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write("# Analysis Report\n\n")
    f.write("## Data Analysis and Insights\n")
    f.write(story)
    f.write("\n\n### Generated Visualizations\n")
    for chart_name, chart_file in charts.items():
        f.write(f"- [{chart_name}]({chart_file})\n")

print("Analysis complete. Story saved to README.md.")
