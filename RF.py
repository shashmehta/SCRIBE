# 🎯 Classify Pancreatic Cell Types Using Cellxgene Census API

# Step 1: Install these if needed:
# pip install cellxgene-census scanpy scikit-learn anndata matplotlib seaborn

import cellxgene_census
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 2: Open the Cellxgene Census Dataset
print("Connecting to Cellxgene Census...")
census = cellxgene_census.open_soma()

# Step 3: Load Human Pancreas Cells
print("Loading human pancreas cells...")
adata = cellxgene_census.get_anndata(
    census=census,
    organism="Homo sapiens",
    obs_value_filter="tissue_general == 'pancreas' and is_primary_data == True",
    measurement_name="RNA"
)

# Step 4: Fix gene names
print("Fixing gene names...")
adata.var_names = adata.var['feature_name']

# Step 5: Filter for target pancreatic endocrine cell types
target_cell_types = [
    'type B pancreatic cell',        # likely beta cell
    'pancreatic A cell',             # alpha cell
    'pancreatic D cell'              # delta cell
]

print("Filtering for target cell types...")
adata = adata[adata.obs['cell_type'].isin(target_cell_types)].copy()

# Optional: Rename cell types for readability
label_map = {
    'type B pancreatic cell': 'beta',
    'pancreatic A cell': 'alpha',
    'pancreatic D cell': 'delta'
}
adata.obs['cell_type'] = adata.obs['cell_type'].map(label_map)

# Step 6: Select marker genes (if available)
marker_genes = ['INS', 'GCG', 'SST', 'MAFA', 'ARX']
genes_present = [gene for gene in marker_genes if gene in adata.var_names]

if not genes_present:
    raise ValueError("None of the marker genes were found in the dataset.")

print(f"Using marker genes: {genes_present}")
adata = adata[:, adata.var_names.isin(genes_present)].copy()

# Step 7: Check final dataset shape
print(f"Final dataset shape: {adata.shape}")
if adata.shape[0] == 0 or adata.shape[1] == 0:
    raise ValueError("Filtered dataset is empty. Double-check cell types and gene names.")

# Step 8: Preprocess the data
print("Normalizing and log-transforming gene expression values...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Step 9: Prepare training and test data
print("Preparing training and test sets...")
X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
y = adata.obs['cell_type'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Step 10: Train the Random Forest model
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 11: Evaluate the model
print("Evaluating model performance...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 12: Visualize gene importance
print("Plotting gene importances...")
importances = clf.feature_importances_
genes = adata.var_names

sns.barplot(x=importances, y=genes)
plt.title("Important Genes for Classifying Pancreatic Cells")
plt.xlabel("Importance")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()
