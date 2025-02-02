import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textstat

def topsis(decision_matrix, weights, impacts):
    """
    Perform TOPSIS analysis on the given decision matrix.
    
    Parameters:
    decision_matrix: DataFrame with alternatives as rows and criteria as columns
    weights: List of weights for each criterion (must sum to 1)
    impacts: List of impacts (+1 for positive impact, -1 for negative impact)
    
    Returns:
    DataFrame with original data, TOPSIS scores and rankings
    """
    # Normalize the decision matrix
    normalized_matrix = decision_matrix.copy()
    for column in decision_matrix.columns:
        normalized_matrix[column] = decision_matrix[column] / np.sqrt((decision_matrix[column]**2).sum())
    
    # Weight normalization
    weighted_matrix = normalized_matrix * weights
    
    # Determine ideal best and worst values considering impacts
    ideal_best = []
    ideal_worst = []
    
    for col, impact in zip(weighted_matrix.columns, impacts):
        if impact == 1:  # Positive impact (higher is better)
            ideal_best.append(weighted_matrix[col].max())
            ideal_worst.append(weighted_matrix[col].min())
        else:  # Negative impact (lower is better)
            ideal_best.append(weighted_matrix[col].min())
            ideal_worst.append(weighted_matrix[col].max())
    
    # Calculate distances
    d_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    
    # Calculate TOPSIS scores
    topsis_scores = d_worst / (d_best + d_worst)
    
    # Add scores and ranks to original data
    result = decision_matrix.copy()
    result['TOPSIS Score'] = topsis_scores
    result['Rank'] = result['TOPSIS Score'].rank(ascending=False).astype(int)
    
    return result.sort_values('Rank')

# Text for summarization
text = """
The Industrial Revolution, which began in the 18th century, transformed societies from agrarian economies into industrial powerhouses. 
The introduction of steam engines, mechanized factories, and textile innovations accelerated economic growth and urbanization. 
Factories increased efficiency but led to poor labor conditions. Urbanization caused overcrowding, sanitation issues, and diseases. 
The invention of the telegraph improved communication, while electricity later revolutionized industry. Despite economic benefits, 
it led to environmental pollution. Today, we see similar transformations through technology, automation, and AI, shaping modern societies.
"""

reference_summary = """
The Industrial Revolution shifted agrarian societies into industrial economies, boosting urbanization and technological progress but causing poor labor conditions and environmental concerns. 
"""

# Models for comparison
models = {
    "BART": "facebook/bart-large-cnn",
    "T5": "t5-small",
    "Pegasus": "google/pegasus-xsum",
    "LED": "allenai/led-large-16384"
}

# Load semantic similarity model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Store results
results = []

for name, model_path in models.items():
    summarizer = pipeline("summarization", model=model_path)
    
    try:
        start = time.time()
        summary = summarizer(text, max_length=100, min_length=50, do_sample=False)[0]["summary_text"]
        inference_time = time.time() - start
        
        compression = len(summary) / len(text)
        readability = textstat.flesch_reading_ease(summary)
        similarity = cosine_similarity(
            semantic_model.encode([reference_summary]),
            semantic_model.encode([summary])
        )[0][0]

        results.append([name, compression, readability, similarity, inference_time])

    except Exception as e:
        print(f"Error summarizing with model {name}: {e}")
        results.append([name, None, None, None, None])

# Create DataFrame with only 4 metrics
df = pd.DataFrame(results, columns=["Model", "Compression", "Readability", "Similarity", "Time"])

# Define weights and impacts
weights = [0.3, 0.3, 0.3, 0.1]  # Weights for each criterion
impacts = [1, 1, 1, -1]  # Impacts for each metric: Compression and Time (negative), Readability and Similarity (positive)

# Perform TOPSIS analysis manually
ranked_results = topsis(df.set_index('Model'), weights, impacts)

# Display results
print("\nTOPSIS Results (Ranked Models):")
print(ranked_results)

# Enhanced Simple Visualizations

# 1. Bar Plot for Each Metric (Comparison for each model)
plt.figure(figsize=(10, 6))
df.set_index('Model')[['Compression', 'Readability', 'Similarity', 'Time']].plot(kind='bar', figsize=(12, 6), colormap='Set2')
plt.title('Model Comparison for Each Metric', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Metric Scores', fontsize=14)
plt.tight_layout()
plt.show()

# 2. Bar Plot for TOPSIS Score Comparison
plt.figure(figsize=(8, 6))
sns.barplot(x=ranked_results.index, y=ranked_results['TOPSIS Score'], palette='viridis')
plt.title('TOPSIS Score Comparison by Model', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('TOPSIS Score', fontsize=14)
plt.tight_layout()
plt.show()

# 3. Box Plot for Model Rankings
plt.figure(figsize=(8, 6))
sns.boxplot(x=ranked_results['Rank'], palette='viridis')
plt.title('Model Rank Distribution', fontsize=16)
plt.xlabel('Rank', fontsize=14)
plt.tight_layout()
plt.show()

# 4. Heatmap for Model-Score Relationships (Metric vs Score)
plt.figure(figsize=(10, 6))
sns.heatmap(df.set_index('Model')[['Compression', 'Readability', 'Similarity', 'Time']], annot=True, cmap='Blues', linewidths=0.5)
plt.title('Heatmap of Model Scores for Each Metric', fontsize=16)
plt.ylabel('Model', fontsize=14)
plt.xlabel('Metrics', fontsize=14)
plt.tight_layout()
plt.show()
