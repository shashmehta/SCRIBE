import scanpy as sc

DATASET = "pancreas.h5ad" #Change this to your dataset path

adata = sc.read_h5ad(DATASET)

print(adata)
import scanpy as sc

# Load the data
adata = sc.read_h5ad("pancreas.h5ad")

# Preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)   # Normalize expression values
sc.pp.log1p(adata)                             # Log-transform
sc.pp.highly_variable_genes(adata, n_top_genes=1000)
adata = adata[:, adata.var.highly_variable]    # Keep top 1000 genes

# PCA to reduce dimensionality
sc.pp.pca(adata)

# Compute distances
sc.pp.neighbors(adata)

# Compute clustering
sc.tl.dendrogram(adata, groupby='cell_type') 

#plot dendrogram out
sc.pl.dendrogram(adata, groupby='cell_type')
print(adata.obs.columns)