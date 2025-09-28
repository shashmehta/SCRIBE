# Plan for Classifying Alpha, Beta, Gamma Pancreatic Cells in Humans and Mice using cellxGene API

#%% 0 Resources to understand concepts.
# - cellxGene Census API documentation: https://cellxgene-census.readthedocs.io/en/latest/
# - Single cell sequencing: https://www.fluigent.com/resources-support/expertise/application-notes/drop-seq-method/
# - Random Forest Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#%% 1. Import necessary libraries
#    - cellxgene API client
#    - pandas, numpy
#    - scikit-learn for classification
#    - matplotlib/seaborn for visualization

import cellxgene_census
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#%% 2. Connect to cellxGene API and search for relevant datasets
print("Connecting to Cellxgene Census...")
census = cellxgene_census.open_soma()
print("Searching for pancreas datasets...")

# Define cell types of interest
cell_types = ["pancreatic A cell",     # Alpha cells (produce glucagon)
    "type B pancreatic cell",# Beta cells (produce insulin)
    "pancreatic D cell",     # Delta cells (produce somatostatin)
]
cell_type_filter = " or ".join([f"cell_type == '{ct}'" for ct in cell_types])

# Updated query with cell type filtering, gene expression features, and protein-coding RNA filter
human_datasets = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter=f"(tissue_general == 'pancreas' and is_primary_data == True and ({cell_type_filter}))",
    obs_column_names=["cell_type"],
    var_column_names=["feature_name", "feature_biotype"],
)


#%% 3. Download and load datasets
#    - Extract expression matrices and cegll metadata
print("Processing human datasets...")
all_cell_data = []

for cell_type in cell_types:
    mask = human_datasets.obs['cell_type'] == cell_type # returns True if obs["cell_type"] has specific value.
    subset = human_datasets[mask][:10000]
    all_cell_data.append(subset)

combined_human_data = sc.concat(all_cell_data)
print(f"Total cells collected: {len(combined_human_data)}")


#%% 4. Split data into training and test sets
#    - Ensure balanced representation of cell types and species

# Get class frequencies
print("Class frequencies:")
cell_type_counts = combined_human_data.obs['cell_type'].value_counts()
print(cell_type_counts)

# Prepare X (features) and y (labels)
X = combined_human_data.X
y = combined_human_data.obs['cell_type']

# The following would calculate balanced class weights manually:
# total_samples = len(y)
# n_classes = len(cell_types)
# class_weights = {}
# for cell_type in cell_types:
#     samples_in_class = sum(y == cell_type)
#     weight = total_samples / (n_classes * samples_in_class)
#     class_weights[cell_type] = weight

# Instead, use 'balanced' which automatically handles class weights
class_weights = 'balanced'

# Split the data while maintaining class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

#%% 6. Train a classifier (e.g., Random Forest, SVM)
#    - Use gene expression profiles to classify cell types
# Initialize and train Random Forest Classifier with class weights
print("Training Random Forest Classifier...")
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    class_weight=class_weights,
    random_state=42,
    n_jobs=-1
)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Print evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# 7. Evaluate classifier performance
#    - Accuracy, confusion matrix, classification report

# 8. Visualize results
#    - Plot confusion matrix, feature importance, etc.

# 9. Save model and results
#    - Export trained model and evaluation metrics

# 10. (Optional) Test cross-species generalization
#     - Train on one species, test on the other
# %%
