import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pathlib import Path

    return Path, plt, sns


@app.cell
def _(mo):
    mo.md("""
    # SCRIBE — Batch Correction Explorer

    Inspect gene expression distributions before and after ComBat batch correction,
    and browse existing UMAP / PCA plots.
    """)
    return


@app.cell
def _(mo):
    from scribe import cache

    if cache.is_cache_stale():
        mo.output.append(mo.md("**Cache is stale — rebuilding from h5ad files...**"))
        cache.build_cache()
        mo.output.append(mo.md("Cache rebuilt."))

    gene_list = cache.get_gene_list()
    obs = cache.load_obs_metadata()
    return cache, gene_list, obs


@app.cell
def _(gene_list, mo):
    from scribe.batch import DEFAULT_HOUSEKEEPING_GENES, CANDIDATE_HOUSEKEEPING_GENES

    available_hk = [g for g in DEFAULT_HOUSEKEEPING_GENES if g in gene_list]

    gene_selector = mo.ui.multiselect(
        options=sorted(gene_list),
        label="Select genes (max 10)",
        max_selections=10,
    )

    hk_button = mo.ui.button(
        label=f"Show HK Genes ({len(available_hk)})",
        value=False,
        on_click=lambda _v: True,
    )

    mo.vstack([
        mo.md("### Gene Distribution Viewer"),
        mo.md("Select genes to compare their expression distributions across datasets, before and after batch correction."),
        mo.hstack([gene_selector, hk_button], justify="start", gap=1),
        mo.md(f"*HK genes available: {', '.join(available_hk)}*"),
    ])
    return available_hk, gene_selector, hk_button


@app.cell
def _(available_hk, cache, gene_selector, hk_button, mo, obs, plt, sns):
    _COLORS = {"GSE154778": "#e41a1c", "GSE162708": "#377eb8", "GSE165399": "#4daf4a"}

    # Use HK genes if button was clicked, otherwise use multiselect
    if hk_button.value:
        selected = available_hk
    else:
        selected = gene_selector.value or []

    if not selected:
        _output = mo.md("*Select one or more genes above to see KDE plots.*")
    else:
        figures = []
        # Load only the selected genes from cache (columnar read)
        uncorr_df = cache.load_gene_expression(list(selected), corrected=False)
        corr_df = cache.load_gene_expression(list(selected), corrected=True)

        for gene in selected:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)

            for ds in ["GSE154778", "GSE162708", "GSE165399"]:
                mask = obs["dataset"] == ds
                sns.kdeplot(uncorr_df.loc[mask, gene], ax=ax1, label=ds,
                            color=_COLORS[ds], fill=True, alpha=0.15)
                sns.kdeplot(corr_df.loc[mask, gene], ax=ax2, label=ds,
                            color=_COLORS[ds], fill=True, alpha=0.15)

            ax1.set_title(f"{gene} — Uncorrected")
            ax2.set_title(f"{gene} — ComBat Corrected")
            ax1.set_xlabel("Expression")
            ax2.set_xlabel("Expression")
            ax1.legend(fontsize=8)
            ax2.legend(fontsize=8)
            plt.tight_layout()
            plt.close(fig)
            figures.append(mo.as_html(fig))

        _output = mo.vstack(figures)

    _output
    return


@app.cell
def _(available_hk, gene_list, mo):
    # Default target genes: a small curated set of well-known markers if present,
    # otherwise fall back to a slice of the gene list (user is expected to edit).
    _default_target_candidates = [
        "INS", "GCG", "SST", "PPY", "KRT19", "REG1A", "PRSS1", "CFTR",
        "CD8A", "CD4", "CD68", "VIM", "EPCAM", "MKI67", "TP53", "KRAS",
    ]
    _default_targets = [g for g in _default_target_candidates if g in gene_list and g not in available_hk]
    if not _default_targets:
        _default_targets = [g for g in gene_list[:10] if g not in available_hk]

    _target_options = sorted([g for g in gene_list if g not in available_hk])

    target_selector = mo.ui.multiselect(
        options=_target_options,
        value=_default_targets,
        label="Target genes (up to 40)",
        max_selections=40,
    )

    mo.vstack([
        mo.md("### HK ↔ Gene Relationship Preservation"),
        mo.md(
            "For each housekeeping × target gene pair, computes the Pearson "
            "correlation *within each batch*, averages across batches, and "
            "compares the before vs after batch correction. Pairs near the "
            "y=x diagonal in the right panel are biologically preserved."
        ),
        target_selector,
    ])
    return (target_selector,)


@app.cell
def _(available_hk, cache, mo, obs, target_selector):
    from scribe.plotting import plot_hk_gene_relationship

    _targets = list(target_selector.value or [])

    if len(available_hk) < 1 or len(_targets) < 1:
        _hk_rel_output = mo.md(
            "*Need at least one housekeeping gene (auto-loaded) and one target "
            "gene selected above.*"
        )
    else:
        _needed = list(dict.fromkeys(list(available_hk) + _targets))
        _uncorr = cache.load_gene_expression(_needed, corrected=False)
        _corr = cache.load_gene_expression(_needed, corrected=True)

        _fig = plot_hk_gene_relationship(
            uncorr_expr=_uncorr,
            corr_expr=_corr,
            obs=obs,
            hk_genes=list(available_hk),
            target_genes=_targets,
            batch_key="dataset",
            save_path=None,
        )
        _hk_rel_output = mo.as_html(_fig)

    _hk_rel_output
    return


@app.cell
def _(Path, mo):
    plot_dir = Path("output/plots")

    # Map friendly display names to file paths
    _PLOT_NAMES = {
        "UMAP: Malignant cells (before vs after correction)": "malignant_uncorrected_vs_corrected.png",
        "UMAP: Normal cells (before vs after correction)": "normal_uncorrected_vs_corrected.png",
        "PCA: Housekeeping genes (before vs after correction)": "hk_pca_uncorrected_vs_corrected.png",
        "HK Analysis: Housekeeping gene PCA": "hk_analysis/housekeeping_pca.png",
        "HK Analysis: HK ↔ Gene Relationship (before vs after)": "hk_analysis/hk_gene_relationship.png",
    }

    # Only include plots that actually exist on disk
    plot_options = {
        name: str(plot_dir / fname)
        for name, fname in _PLOT_NAMES.items()
        if (plot_dir / fname).exists()
    }

    gallery_dropdown = mo.ui.dropdown(
        options=list(plot_options.keys()),
        label="Select plot",
        searchable=True,
    )

    mo.vstack([
        mo.md("### Plot Gallery"),
        mo.md("Browse existing UMAP, PCA, and housekeeping gene analysis plots."),
        gallery_dropdown,
    ])
    return gallery_dropdown, plot_options


@app.cell
def _(Path, gallery_dropdown, mo, plot_options):
    import base64 as _b64

    if gallery_dropdown.value and gallery_dropdown.value in plot_options:
        _path = Path(plot_options[gallery_dropdown.value])
        _data = _b64.b64encode(_path.read_bytes()).decode()
        _img_html = f'<img src="data:image/png;base64,{_data}" style="max-width:100%;" />'
        _result = mo.vstack([
            mo.md(f"**{gallery_dropdown.value}**"),
            mo.Html(_img_html),
        ])
    else:
        _result = mo.md("*Select a plot from the dropdown above.*")

    _result
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    *SCRIBE Batch Correction Explorer — data from Google Drive*
    """)
    return


if __name__ == "__main__":
    app.run()
