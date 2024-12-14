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
