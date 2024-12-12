# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "seaborn",
#   "openai==0.28",
#   "scipy",
#   "bese64",
#   "sklearn",
#   "io",
#   "PIL",
# ]
# ///
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import httpx
import chardet
import time
import base64
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image

# Set the AIPROXY TOKEN and URL

AIPROXY_TOKEN="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDMxNTdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.jqgA4t_kF3xuPBBPlGeLSKobS64VlLoTFSdh9MUMchQ"
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

# Load the 'csv' file
def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())  # # Detect the encoding of a file that may have different or unknown character encodings from different sources or systems.
        encoding = result['encoding']   # which encoding among these('utf-8', 'ISO-8859-1','Windows-1252')
        data = pd.read_csv(file_path, encoding=encoding)   # prevent issues where characters from different languages or symbols may appear as garbage or unreadable text when reading the CSV file
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

# Perform basic analysis like summary stats, missing values, etc.

def basic_analysis(data):
    summary = data.describe(include='all').to_dict()   # compute summary statistics (like count, mean, standard deviation, min, max, etc.) for numerical columns.
    missing_values = data.isnull().sum().to_dict()  # Missing values
    column_info = data.dtypes.to_dict()  # Column types
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

# Robust outlier detection using IQR (Interquartile Range)

def outlier_detection(data):
    numeric_data = data.select_dtypes(include=np.number)  # returns columns with numeric data
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1           #Outliers is for identifying extreme values in your dataset that could be errors or just rare events.
    outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum().to_dict()
    return {"outliers": outliers}

# Correlation Matrix

def generate_correlation_matrix(data, output_dir):   # To find relationships between multiple variables
    data = data.select_dtypes(include=[np.number])
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    corr_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(corr_path)
    plt.close()
    return corr_path

    
# DBSCAN clustering (Density-Based Clustering)

def dbscan_clustering(data, output_dir):    # Identifying regions of high density
    numeric_data = data.select_dtypes(include=np.number).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    numeric_data['cluster'] = clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=numeric_data.iloc[:, 0], y=numeric_data.iloc[:, 1], hue=numeric_data['cluster'], palette="viridis")
    plt.title("DBSCAN Clustering")
    dbscan_path = os.path.join(output_dir, "dbscan_clusters.png")
    plt.savefig(dbscan_path)
    print("dbscan_clusters.png created")
    plt.close()
    return dbscan_path

# Hierarchical Clustering

def hierarchical_clustering(data, output_dir):      #group similar data points together
    
    numeric_data = data.select_dtypes(include=np.number).dropna()
    linked = linkage(numeric_data, 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title("Hierarchical Clustering Dendrogram")
    hc_path = os.path.join(output_dir, "hierarchical_clustering.png")
    plt.savefig(hc_path)
    print("hierarchical_clustering.png created")
    plt.close()
    return hc_path

# Function to convert image to Base64

def image_to_base64(image_path, save_path):    # ensure the integrity of binary data during transmission,
    with Image.open(image_path) as img:
        target_width = 800
        width, height = img.size
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)  # Maintain aspect ratio

        # Resize the image using LANCZOS filter
        resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Ensure the directory exists before saving the image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print("Save path created")

        # Save the resized image to the given file path
        resized_img.save(save_path, format="PNG")  # Save the image as PNG (or adjust format if needed)
        print("Saved")

        # Save the resized image to a BytesIO object (in-memory binary stream)
        img_byte_arr = BytesIO()
        resized_img.save(img_byte_arr, format="PNG")
        
        img_byte_arr.seek(0)  # Go to the start of the BytesIO buffer

        # Encode the binary data to Base64
        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        
    return img_base64  # Return Base64-encoded image


# Function to send the data info to the LLM and request analysis or code

def query_llm_for_analysis(prompt):
   

    # Prepare the prompt to query the LLM
    
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    payload = {
        "model": "gpt-4o-mini",  # or use the correct model
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.7
    }

    retries = 10  # Increased retries before giving up
    backoff_factor = 2  # Exponential backoff factor
    max_wait_time = 60  # Maximum wait time (1 minute) to prevent indefinite retries

    for attempt in range(retries):
        try:
            response = httpx.post(API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Exponential backoff with a cap on wait time
                wait_time = min(backoff_factor ** attempt, max_wait_time)
                print(f"Rate limit hit, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error querying the LLM: {e}")
                break
        except httpx.RequestError as e:
            print(f"Error querying the LLM: {e}")
            break

    print("Max retries reached, giving up.")
    sys.exit(1)  # Exit after retries have failed

# Save results in a Markdown README

def save_readme(content, output_dir):
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(content)
        print("Readme saved")


# Function to analyze and generate output for each file

def analyze_and_generate_output(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_data(file_path)
    print("Data loaded")
    
    # Perform basic analysis
    analysis = basic_analysis(data)
    outliers = outlier_detection(data)
    combined_analysis = {**analysis, **outliers}

    # Generate visualizations and save file paths
    image_paths = {}
    image_paths['correlation_matrix'] = generate_correlation_matrix(data, output_dir)
    image_paths['dbscan_clusters'] = dbscan_clustering(data, output_dir)
    image_paths['hierarchical_clustering'] = hierarchical_clustering(data, output_dir)
    print("Images created:\n", image_paths)

    
    images_base64, filenames = process_images(image_paths, output_dir)

    # Example output to verify
    print("Base64 Encoded Images (Keys Only for Verification):")
    print("keys:",list(images_base64.keys()))  # To show the keys, without the lengthy Base64 data

    print("Resized Image Filenames for LLM Analysis:")
    print("filenames after resize:",filenames)  # Send these filenames to LLM for analysis

    # Send data to LLM for analysis and suggestions
    data_info = {
        "filename": file_path,
        "summary": combined_analysis["summary"],
        "missing_values": combined_analysis["missing_values"],
        "outliers": combined_analysis["outliers"]
    }
    
    prompt = (
        "You are a creative storyteller. "
        "Craft a compelling narrative based on this dataset analysis:\n\n"
        f"Data Summary: {data_info['summary']}\n\n"
        f"Missing Values: {data_info['missing_values']}\n\n"
        f"Outlier Analysis: {data_info['outliers']}\n\n"
        "Create a narrative covering these points:\n"
        f"Correlation matrix:{filenames[0]},\n"
        f"DBSCAN Clusters: {filenames[1]},\n"
        f"Hierarchical Clustering: {filenames[2]}\n"
    )
    narrative = query_llm_for_analysis(prompt)
   # print(f"\nLLM Narrative:\n{narrative}")

    # Save the narrative to a README file
    save_readme(narrative, output_dir)

def resize_image(input_path, output_path, size=(300, 300)):   # reduce the size of the images 
    """Resize an image to the specified size."""
    with Image.open(input_path) as img:
        img = img.resize(size)
        img.save(output_path)
    print(f"Image saved to {output_path}")

def image_to_base64(image_path):
    """Convert an image to a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_images(image_paths, output_dir, resize_size=(300, 300)):
    """Resize images, convert to Base64"""
    images_base64 = {}
    filenames = []

    for description, path in image_paths.items():
        # Step 1: Resize the image
        resized_image_path = os.path.join(output_dir, f"resized_{description}.png")
        resize_image(path, resized_image_path, size=resize_size)
        
        # Step 2: Convert the resized image to Base64
        base64_data = image_to_base64(resized_image_path)
        images_base64[description] = base64_data
        
        # Step 3: Collect the filename (without the lengthy Base64 data)
        filenames.append(resized_image_path)
    
    return images_base64, filenames



# Main execution function
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()
