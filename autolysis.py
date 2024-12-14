import os
import sys
import subprocess

import time

# Record the start time
start_time = time.time()


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

# Define the output directory
output_dir = os.getcwd()

# Visualization - Save correlation heatmap
correlation_plot = os.path.join(output_dir, "correlation_matrix.png")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(correlation_plot)
plt.close()

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
pca_plot = os.path.join(output_dir, "pca_scatter.png")
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, c="blue", label="Data Points")
plt.title("PCA Scatter Plot")
plt.xlabel(f"Principal Component 1 ({features[0]})")
plt.ylabel(f"Principal Component 2 ({features[1]})")
plt.legend()
plt.tight_layout()
plt.savefig(pca_plot)
plt.close()

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(data_scaled)

# KMeans Clustering Plot
clustering_plot = os.path.join(output_dir, "kmeans_clustering.png")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data["Cluster"], palette="viridis", legend="full")
plt.title("KMeans Clustering")
plt.xlabel(f"Principal Component 1 ({features[0]})")
plt.ylabel(f"Principal Component 2 ({features[1]})")
plt.legend(title="Clusters")
plt.tight_layout()
plt.savefig(clustering_plot)
plt.close()

# Outlier Detection
distances = kmeans.transform(data_scaled).min(axis=1)
threshold = distances.mean() + 3 * distances.std()
outliers = distances > threshold

outliers_plot = os.path.join(output_dir, "outliers.png")
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
plt.savefig(outliers_plot)
plt.close()

# Use AI Proxy for story generation
import requests

def generate_story(summary, visualizations):
    summary_str = f"Summary: {summary['Summary Statistics']}\nMissing: {summary['Missing Values']}"
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
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500},
        headers=headers,
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Failed to generate story. Error {response.status_code}: {response.text}"
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

readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("# Analysis Report\n\n")
    f.write("## Data Analysis and Insights\n")
    f.write(story)
    f.write("\n\n### Generated Visualizations\n")
    for chart_name, chart_file in charts.items():
        f.write(f"- [{chart_name}]({chart_file})\n")

print("Analysis complete. Story saved to README.md.")
# Record the end time
end_time = time.time()

# Calculate and display the elapsed time
elapsed_time = end_time - start_time
print(f"Script executed in {elapsed_time:.2f} seconds.")
'''
import os
import sys
import subprocess

# Function to install Python packages if not already installed
def install_package(package_name, submodules=None):
    """Installs a Python package if not already installed."""
    try:
        __import__(package_name)
        if submodules:
            for submodule in submodules:
                __import__(f"{package_name}.{submodule}")
    except ImportError:
        print(f"Installing package: {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Install required libraries
required_packages = [
    ("pandas", None),
    ("seaborn", None),
    ("matplotlib", None),
    ("scikit-learn", None),
    ("requests", None),
    ("chardet", None),
    ("numpy", None),
    ("joblib", ["externals.loky.backend.context"]),
    ("folium", None),
    ("plotly", None),
    ("Pillow", None),
    ("geopy", None),
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
    print("Usage: python autolysis.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' does not exist.")
    sys.exit(1)

# Import dependencies
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
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from PIL import Image
import requests
import plotly.express as px

# Suppress warnings
warnings.filterwarnings("ignore")
set_start_method("loky", force=True)

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']

# Function to compress and save plot
def save_compressed_plot(fig, filename, quality=70):
    # Save the figure without using 'optimize' argument
    fig.savefig(filename, dpi=150, bbox_inches='tight', format="png")
    img = Image.open(filename)  # Open the saved image using Pillow
    img.save(filename, "PNG", optimize=True, quality=quality)  # Compress the image
    plt.close(fig)  # Close the plot to free up memory

# Function to load data
def load_data(file_path):
    encoding = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=encoding)

# Function to summarize data
def summarize_data(data):
    summary = {
        "Missing Values": data.isnull().sum().to_dict(),
        "Summary Statistics": data.describe(include="all").to_dict()
    }
    return summary

# Outlier Detection using KMeans
def detect_and_plot_outliers(data, output_file):
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Handle missing values by imputing with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(numeric_data)
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    # Apply KMeans to detect outliers
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data_scaled)
    
    # Calculate distances from the cluster centroids
    distances = kmeans.transform(data_scaled).min(axis=1)
    
    # Define the threshold for outliers (3 standard deviations from the mean distance)
    threshold = distances.mean() + 3 * distances.std()
    outliers = distances > threshold

    # Plot the non-outliers and outliers
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure for plotting
    ax.scatter(
        np.where(~outliers)[0], distances[~outliers],
        c='blue', label="Non-Outliers"
    )
    ax.scatter(
        np.where(outliers)[0], distances[outliers],
        c='red', label="Outliers"
    )
    
    ax.set_title("Outlier Detection (KMeans)", fontsize=16)
    ax.set_xlabel("Data Points", fontsize=12)
    ax.set_ylabel("Distance from Centroid", fontsize=12)
    ax.legend()
    plt.tight_layout()
    
    # Save the plot with compression
    save_compressed_plot(fig, output_file)

# Function to generate correlation heatmap
def generate_correlation_heatmap(data, output_file):
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_compressed_plot(fig, output_file)

# Function to perform PCA and KMeans clustering
def perform_pca_and_kmeans(data, output_file):
    numeric_data = data.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy="mean")
    scaled_data = StandardScaler().fit_transform(imputer.fit_transform(numeric_data))

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Visualize PCA with clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=clusters, palette="viridis", ax=ax)
    ax.set_title("PCA and KMeans Clustering")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    save_compressed_plot(fig, output_file)

# Function for time series analysis
def perform_time_series_analysis(data, output_file):
    time_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    if not time_cols:
        return False  # No time series data
    fig = px.line(data, x=time_cols[0], y=data.select_dtypes(include=[np.number]).columns[0], title="Time Series Analysis")
    fig.write_image(output_file)
    return True

# Function for geographical analysis
def perform_geographical_analysis(data, output_file):
    if {'latitude', 'longitude'}.issubset(data.columns):
        m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=4)
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in data.iterrows():
            folium.Marker([row['latitude'], row['longitude']]).add_to(marker_cluster)
        m.save(output_file)
        return True
    return False

# Function to generate a story with AI Proxy
def generate_story(summary, visualizations):
    summary_str = f"Summary: {summary['Summary Statistics']}\nMissing: {summary['Missing Values']}"
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
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500},
        headers=headers,
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Failed to generate story. Error {response.status_code}: {response.text}"

# Function to save the final report
def save_report(story, output_file):
    print(f"Saving the report to {output_file}...")

    # Check if each file exists before adding it to the report
    report_content = "# Analysis Report\n\n"
    report_content += "## Key Insights\n\n"
    report_content += story
    report_content += "\n\n## Visualizations\n"

    # Visualizations section: add only files that exist
    if os.path.exists("compressed_correlation_heatmap.png"):
        report_content += "- ![Correlation Heatmap](compressed_correlation_heatmap.png)\n"
    if os.path.exists("compressed_pca_kmeans.png"):
        report_content += "- ![PCA Clustering](compressed_pca_kmeans.png)\n"
    if os.path.exists("compressed_timeseries.png"):
        report_content += "- ![Time Series](compressed_timeseries.png)\n"
    if os.path.exists("compressed_outliers.png"):
        report_content += "- ![Outlier Detection](compressed_outliers.png)\n"
    if os.path.exists("geographic_analysis.html"):
        report_content += "- Interactive Map: geographic_analysis.html\n"

    # Write the content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("Report saved successfully!")

# Main execution
if __name__ == "__main__":
    data = load_data(csv_file)
    summary = summarize_data(data)

    # Generate the correlation heatmap
    generate_correlation_heatmap(data, "compressed_correlation_heatmap.png")

    # Generate the PCA and KMeans plot
    perform_pca_and_kmeans(data, "compressed_pca_kmeans.png")

    # Perform time series analysis
    time_series_done = perform_time_series_analysis(data, "compressed_timeseries.png")

    # Perform geographical analysis
    geo_done = perform_geographical_analysis(data, "geographic_analysis.html")

    # Generate the outlier detection plot
    detect_and_plot_outliers(data, "compressed_outliers.png")
    
    # Generate the story
    story = generate_story(summary, ["compressed_correlation_heatmap.png", "compressed_pca_kmeans.png", "compressed_outliers.png"])
    
    if story:
        save_report(story, "README.md")
        print("Analysis complete! Report saved to README.md.")
    else:
        print("Failed to generate story. Report not saved.")
'''
