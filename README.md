# Summarization Model Evaluation Using TOPSIS
This project evaluates different text summarization models using the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) method. The evaluation is based on four metrics: Compression, Readability, Similarity, and Inference Time. The results are ranked based on these metrics, and visualized with simple plots for better comparison.

## Project Overview
The project compares four text summarization models:

### BART: A sequence-to-sequence model fine-tuned for summarization.
### T5: A text-to-text transformer model for various NLP tasks, including summarization.
### Pegasus: A transformer-based model for abstractive summarization.
### LED: A model designed for long-document summarization.

## The evaluation metrics used are:

### Compression: The ratio of the summary length to the original text length.
### Readability: The Flesch Reading Ease score, measuring how easy the text is to read.
### Similarity: The cosine similarity between the summary and a reference summary.
### Inference Time: The time taken for the model to generate a summary.
Requirements
To run the project, you'll need to install the following Python packages:

```bash
pip install pandas numpy matplotlib seaborn transformers sentence-transformers textstat scikit-learn
```
## Script Overview
### Data Preparation:

A long piece of text (Industrial Revolution summary) is used for evaluation.
A reference summary is also provided for calculating similarity.
### Model Setup:

Summarization models (BART, T5, Pegasus, LED) are loaded using Hugging Face's transformers library.
### Metrics Calculation:

Compression is calculated as the ratio of the summary length to the original text.
Readability is measured using the textstat library.
Similarity is computed using sentence-transformers to compare the model's summary with the reference summary.
Inference time is recorded for each model.
### TOPSIS Evaluation:

The TOPSIS method is used to rank the models based on the computed metrics. Weights and impacts are defined for each metric:
Compression: Positive impact (shorter summaries are better).
Readability: Positive impact (higher readability is better).
Similarity: Positive impact (higher similarity is better).
Inference Time: Negative impact (faster inference is better).
### Visualization:

Several plots are generated to visualize the results:
Bar Plot: Comparing each model across all metrics.
TOPSIS Score Comparison: Bar plot of TOPSIS scores for each model.
Box Plot: Distribution of model ranks.
Heatmap: Heatmap of model scores for each metric.
### Usage
Run the Python script to evaluate the models and generate visualizations:

```bash
python analysis.py
```
### The script will output the following:

#### TOPSIS Results: A ranked list of models based on the TOPSIS score.

#### Visualizations: Several plots will be shown, including bar plots, heatmaps, and box plots.

### Example Output
The output will look like the following:

#### TOPSIS Results (Ranked Models):

![Screenshot 2025-02-02 113859](https://github.com/user-attachments/assets/f9ad108f-09c9-48ec-a4da-189a1f5e5993)

#### Bar Plots:

Comparison of models for Compression, Readability, Similarity, and Inference Time.
![Screenshot 2025-02-02 113953](https://github.com/user-attachments/assets/58b02937-bed2-48b8-b60b-bbe7662f97ac)

TOPSIS Score comparison for each model.
![Screenshot 2025-02-02 114007](https://github.com/user-attachments/assets/ae9906eb-6056-4ae1-ad7e-7ece67968bad)

#### Heatmap:

Visualization of model performance across the metrics.
![Screenshot 2025-02-02 114039](https://github.com/user-attachments/assets/95dec690-288d-4c7d-925c-e0bf5dcab6dc)

## Conclusion
This project helps in comparing and ranking different text summarization models based on multiple evaluation metrics using the TOPSIS method. The results and visualizations can guide decision-making when choosing a summarization model based on specific needs (e.g., speed vs. quality).
