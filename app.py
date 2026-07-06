import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pathlib import Path

    def fig_to_html(fig):
        """Render a matplotlib figure as a PNG <img> that keeps its aspect ratio.

        mo.as_html() emits inline SVG that gets stretched to 100% width; embedding
        a PNG with height:auto preserves the figure's true proportions.
        """
        import io
        import base64

        _buf = io.BytesIO()
        fig.savefig(_buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        _data = base64.b64encode(_buf.getvalue()).decode()
        return mo.Html(
            f'<img src="data:image/png;base64,{_data}" '
            f'style="max-width:100%;height:auto;display:block;margin:0 auto;"/>'
        )

    return Path, fig_to_html, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md("""
    # SCRIBE — Single-Cell RNA Interpretable Biomarker Explorer

    **SCRIBE** is a machine learning pipeline for identifying biomarkers and therapeutic
    targets from single-cell RNA sequencing (scRNA-seq) data. Using an explainable
    Random Forest classifier on pancreatic cancer datasets, it ranks genes that
    discriminate between normal, precancerous, and malignant cell states — and
    applies batch correction to disentangle technical from biological variation.

    | Dataset | Cells | Condition |
    |---|---|---|
    | GSE154778 | 14,924 | Malignant (PDAC — primary & metastatic) |
    | GSE162708 | 21,938 | Normal + Malignant (neuroendocrine tumor) |
    | GSE165399 | 9,071 | Normal + Precancerous + Malignant |

    **45,933 cells × 3,004 genes** after batch-aware highly variable gene selection.
    Batch effects were corrected using **ComBat** and **Harmony**, and a three-class
    Random Forest (normal / precancerous / malignant) was trained to identify the most
    discriminating genes per condition.
    """)
    return


@app.cell
def _(mo):
    import base64
    import os as _os
    from pathlib import Path as _Path

    # Resolve relative to the app dir so it works regardless of marimo's CWD.
    _app_dir = _Path(_os.environ.get("SCRIBE_APP_DIR", _os.getcwd()))
    _poster = _app_dir / "web" / "assets" / "poster.jpg"
    if _poster.exists():
        _data = base64.b64encode(_poster.read_bytes()).decode()
        _poster_out = mo.Html(
            f'<img src="data:image/jpeg;base64,{_data}" '
            f'style="width:100%;border-radius:8px;margin:1rem 0;" '
            f'alt="SCRIBE research poster"/>'
        )
    else:
        _poster_out = mo.md(f"*Poster not found — path: `{_poster}`*")
    _poster_out
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Interactive Batch Correction Analysis

    Explore the batch correction results below. Use the **UMAP viewer** to see how
    ComBat and Harmony reduce dataset-driven clustering, the **HK Gene PCA** to
    assess correction quality on housekeeping genes, and the **Plot Gallery** to
    browse pre-rendered analysis figures.
    """)
    return


@app.cell
def _(mo):
    import os
    from scribe import cache

    if not os.environ.get("SCRIBE_READ_ONLY"):
        if cache.is_cache_stale():
            mo.output.append(mo.md("**Cache is stale — rebuilding from h5ad files...**"))
            cache.build_cache()
            mo.output.append(mo.md("Cache rebuilt."))

    gene_list = cache.get_gene_list()
    obs = cache.load_obs_metadata()
    harmony_available = cache.has_harmony_cache()

    if not harmony_available:
        mo.output.append(
            mo.md(
                "> **Note:** Harmony cache not found. Run "
                "`scribe correct-zarr --data output/processed/combined_processed.zarr "
                "--method harmony --source-h5ad output/processed/combined_processed.h5ad "
                "--to-h5ad` to enable the Harmony comparison. "
                "Showing 2-panel (Uncorrected | ComBat) mode."
            )
        )
    return cache, gene_list, harmony_available, obs


# ── Gene Distribution Viewer ─────────────────────────────────────────────────


@app.cell
def _(cache, gene_list, mo):
    from scribe.batch import DEFAULT_HOUSEKEEPING_GENES

    available_hk = [g for g in DEFAULT_HOUSEKEEPING_GENES if g in gene_list]

    # The full per-gene expression matrices are Drive-only and absent from the
    # deployment. When they're missing, restrict the selector to the bundled
    # housekeeping genes so selecting a gene never triggers a missing-file error.
    _full_expr = cache.has_full_expression_cache()
    _options = sorted(gene_list) if _full_expr else available_hk

    gene_selector = mo.ui.multiselect(
        options=_options,
        label="Select genes (max 10)",
        max_selections=10,
    )

    hk_button = mo.ui.button(
        label=f"Show HK Genes ({len(available_hk)})",
        value=False,
        on_click=lambda _v: True,
    )

    _note = (
        f"*HK genes available: {', '.join(available_hk)}*"
        if _full_expr
        else (
            "*This deployment bundles expression data for the "
            f"{len(available_hk)} housekeeping genes only "
            f"({', '.join(available_hk)}); the full matrix is too large to host. "
            "Select from these to compare distributions across datasets.*"
        )
    )

    mo.vstack([
        mo.md("### Gene Distribution Viewer"),
        mo.md(
            "Select genes to compare expression distributions across datasets. "
            "Harmony panel shows gene expression reconstructed from the top-N "
            "Harmony-corrected PCs (lossy — captures ~50% of variance)."
        ),
        mo.hstack([gene_selector, hk_button], justify="start", gap=1),
        mo.md(_note),
    ])
    return available_hk, gene_selector, hk_button


@app.cell
def _(available_hk, cache, fig_to_html, gene_selector, harmony_available, hk_button, mo, obs, plt, sns):
    _COLORS = {"GSE154778": "#e41a1c", "GSE162708": "#377eb8", "GSE165399": "#4daf4a"}

    # Use HK genes if button was clicked, otherwise use multiselect
    if hk_button.value:
        _selected = available_hk
    else:
        _selected = gene_selector.value or []

    if not _selected:
        _output = mo.md("*Select one or more genes above to see KDE plots.*")
    else:
        _figures = []
        # Use the small HK-specific parquets when all selected genes are HK genes
        _all_hk = all(g in available_hk for g in _selected)
        if _all_hk:
            _uncorr_df = cache.load_hk_expression(method="uncorrected")[list(_selected)]
            _combat_df = cache.load_hk_expression(method="combat")[list(_selected)]
            _harmony_df = (
                cache.load_hk_expression(method="harmony")[list(_selected)]
                if harmony_available else None
            )
        else:
            _uncorr_df = cache.load_gene_expression(list(_selected), method="uncorrected")
            _combat_df = cache.load_gene_expression(list(_selected), method="combat")
            _harmony_df = (
                cache.load_gene_expression(list(_selected), method="harmony")
                if harmony_available else None
            )

        _n_panels = 3 if harmony_available else 2
        _width = 6 * _n_panels

        for _gene in _selected:
            _fig, _axes = plt.subplots(1, _n_panels, figsize=(_width, 3.5), sharey=True)
            _axes = list(_axes)

            for _ds in ["GSE154778", "GSE162708", "GSE165399"]:
                _mask = obs["dataset"] == _ds
                sns.kdeplot(_uncorr_df.loc[_mask, _gene], ax=_axes[0], label=_ds,
                            color=_COLORS[_ds], fill=True, alpha=0.15)
                sns.kdeplot(_combat_df.loc[_mask, _gene], ax=_axes[1], label=_ds,
                            color=_COLORS[_ds], fill=True, alpha=0.15)
                if harmony_available:
                    sns.kdeplot(_harmony_df.loc[_mask, _gene], ax=_axes[2], label=_ds,
                                color=_COLORS[_ds], fill=True, alpha=0.15)

            _axes[0].set_title(f"{_gene} — Uncorrected")
            _axes[1].set_title(f"{_gene} — ComBat")
            if harmony_available:
                _axes[2].set_title(f"{_gene} — Harmony (reconstructed)")
            for _ax in _axes:
                _ax.set_xlabel("Expression")
                _ax.legend(fontsize=8)
            plt.tight_layout()
            _figures.append(fig_to_html(_fig))

        _output = mo.vstack(_figures)

    _output
    return


# ── Interactive UMAP Viewer ──────────────────────────────────────────────────


@app.cell
def _(cache, harmony_available, mo):
    # Load UMAP coordinates
    umap_uncorr = cache.load_umap_coords(method="uncorrected")
    umap_combat = cache.load_umap_coords(method="combat")
    umap_harmony = (
        cache.load_umap_coords(method="harmony") if harmony_available else None
    )

    # Load per-method obs (leiden/phase differ across methods)
    combat_obs = cache.load_method_obs("combat")
    harmony_obs = cache.load_method_obs("harmony") if harmony_available else None

    umap_annotation = mo.ui.dropdown(
        options=["None", "condition", "dataset", "leiden", "cell cycle phase", "sample"],
        value="None",
        label="Color by",
    )

    mo.vstack([
        mo.md("### Interactive UMAP Viewer"),
        mo.md(
            "Side-by-side UMAP projections before and after batch correction. "
            "Harmony's UMAP is computed directly from its corrected PC embedding "
            "— its native output — so no reconstruction is involved."
        ),
        umap_annotation,
    ])
    return (
        combat_obs,
        harmony_obs,
        umap_annotation,
        umap_combat,
        umap_harmony,
        umap_uncorr,
    )


@app.cell
def _(
    combat_obs,
    fig_to_html,
    harmony_available,
    harmony_obs,
    mo,
    np,
    obs,
    plt,
    umap_annotation,
    umap_combat,
    umap_harmony,
    umap_uncorr,
):
    import base64 as _b64
    import os as _umap_os
    from pathlib import Path as _UmapPath

    _annotation = umap_annotation.value
    _app_dir = _UmapPath(_umap_os.environ.get("SCRIBE_APP_DIR", _umap_os.getcwd()))
    _default_png = _app_dir / "web" / "plots" / "umap_default.png"

    def _render_umap():
        def _get_palette(categories):
            cats = sorted(categories.unique())
            _n = len(cats)
            if _n <= 10:
                _colors = plt.cm.tab10(np.linspace(0, 1, 10))[:_n]
            elif _n <= 20:
                _colors = plt.cm.tab20(np.linspace(0, 1, 20))[:_n]
            else:
                _colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, _n))
            return {cat: _colors[i] for i, cat in enumerate(cats)}

        _COL_MAP = {
            "condition": "condition",
            "dataset": "dataset",
            "leiden": "leiden",
            "cell cycle phase": "phase",
            "sample": "sample",
        }
        _col = _COL_MAP.get(_annotation)

        _n_panels = 3 if harmony_available else 2
        _width = 8 * _n_panels
        _fig, _axes = plt.subplots(1, _n_panels, figsize=(_width, 6))
        _axes = list(_axes)

        _rng = np.random.RandomState(42)
        _idx = _rng.permutation(len(umap_uncorr))

        _panels = [
            ("Uncorrected", umap_uncorr, obs),
            ("ComBat", umap_combat, combat_obs),
        ]
        if harmony_available:
            _panels.append(("Harmony", umap_harmony, harmony_obs))

        if _col is None:
            for _ax, (_title, _coords, _) in zip(_axes, _panels):
                _ax.scatter(
                    _coords["UMAP1"].values[_idx], _coords["UMAP2"].values[_idx],
                    c="#cccccc", s=1, alpha=0.3, rasterized=True,
                )
        else:
            _palette = _get_palette(obs[_col])

            for _ax, (_title, _coords, _method_obs) in zip(_axes, _panels):
                if _col in ("leiden", "phase"):
                    _labels = _method_obs[_col].values
                else:
                    _labels = obs[_col].values

                for _cat in sorted(obs[_col].unique()):
                    _m = _labels[_idx] == _cat
                    _ax.scatter(
                        _coords["UMAP1"].values[_idx][_m],
                        _coords["UMAP2"].values[_idx][_m],
                        c=[_palette[_cat]], s=1, alpha=0.5, label=_cat, rasterized=True,
                    )

            _n_cats = obs[_col].nunique()
            for _ax in _axes:
                if _n_cats <= 10:
                    _ax.legend(fontsize=8, markerscale=5, loc="best", frameon=True)
                else:
                    _ax.legend(
                        fontsize=6, markerscale=4, loc="center left",
                        bbox_to_anchor=(1.01, 0.5), frameon=True, ncol=1,
                    )

        for _ax, (_title, _, _) in zip(_axes, _panels):
            _ax.set_title(_title, fontsize=13)
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.set_xlabel("UMAP1")
            _ax.set_ylabel("UMAP2")

        _fig.suptitle(
            f"UMAP — colored by {_annotation}" if _col else "UMAP",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        return fig_to_html(_fig)

    if _annotation == "None" and _default_png.exists():
        _umap_data = _b64.b64encode(_default_png.read_bytes()).decode()
        _viz = mo.Html(f'<img src="data:image/png;base64,{_umap_data}" style="width:100%;"/>')
    else:
        _viz = _render_umap()

    _viz
    return


# ── Interactive HK PCA Viewer ────────────────────────────────────────────────


@app.cell
def _(cache, harmony_available, mo):
    pca_uncorr, var_uncorr = cache.load_hk_pca(method="uncorrected")
    pca_combat, var_combat = cache.load_hk_pca(method="combat")
    if harmony_available:
        pca_harmony, var_harmony = cache.load_hk_pca(method="harmony")
    else:
        pca_harmony, var_harmony = None, None

    hk_genes_used = cache.get_hk_genes_from_cache()

    pca_annotation = mo.ui.dropdown(
        options=["None", "condition", "dataset", "leiden", "cell cycle phase", "sample"],
        value="None",
        label="Color by",
    )

    mo.vstack([
        mo.md("### Housekeeping Gene PCA"),
        mo.md(
            f"PCA on **{len(hk_genes_used)} housekeeping genes** "
            f"({', '.join(hk_genes_used)}). "
            "Harmony panel uses HK genes reconstructed from corrected PCs (lossy)."
        ),
        pca_annotation,
    ])
    return (
        hk_genes_used,
        pca_annotation,
        pca_combat,
        pca_harmony,
        pca_uncorr,
        var_combat,
        var_harmony,
        var_uncorr,
    )


@app.cell
def _(
    combat_obs,
    fig_to_html,
    harmony_available,
    harmony_obs,
    mo,
    np,
    obs,
    pca_annotation,
    pca_combat,
    pca_harmony,
    pca_uncorr,
    plt,
    var_combat,
    var_harmony,
    var_uncorr,
):
    _COL_MAP = {
        "condition": "condition",
        "dataset": "dataset",
        "leiden": "leiden",
        "cell cycle phase": "phase",
        "sample": "sample",
    }

    _annotation = pca_annotation.value

    def _render_hk_pca():
        def _get_palette(categories):
            cats = sorted(categories.unique())
            _n = len(cats)
            if _n <= 10:
                _colors = plt.cm.tab10(np.linspace(0, 1, 10))[:_n]
            elif _n <= 20:
                _colors = plt.cm.tab20(np.linspace(0, 1, 20))[:_n]
            else:
                _colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, _n))
            return {cat: _colors[i] for i, cat in enumerate(cats)}

        _col = _COL_MAP.get(_annotation)

        _n_panels = 3 if harmony_available else 2
        _width = 8 * _n_panels
        _fig, _axes = plt.subplots(1, _n_panels, figsize=(_width, 6))
        _axes = list(_axes)

        _rng = np.random.RandomState(42)
        _idx = _rng.permutation(len(pca_uncorr))

        _panels = [
            ("Uncorrected", pca_uncorr, var_uncorr, obs),
            ("ComBat", pca_combat, var_combat, combat_obs),
        ]
        if harmony_available:
            _panels.append(("Harmony (reconstructed)", pca_harmony, var_harmony, harmony_obs))

        if _col is None:
            for _ax, (_title, _coords, _var, _) in zip(_axes, _panels):
                _ax.scatter(
                    _coords["PC1"].values[_idx], _coords["PC2"].values[_idx],
                    c="#cccccc", s=1, alpha=0.3, rasterized=True,
                )
        else:
            _palette = _get_palette(obs[_col])

            for _ax, (_title, _coords, _var, _method_obs) in zip(_axes, _panels):
                if _col in ("leiden", "phase"):
                    _labels = _method_obs[_col].values
                else:
                    _labels = obs[_col].values

                for _cat in sorted(obs[_col].unique()):
                    _m = _labels[_idx] == _cat
                    _ax.scatter(
                        _coords["PC1"].values[_idx][_m],
                        _coords["PC2"].values[_idx][_m],
                        c=[_palette[_cat]], s=1, alpha=0.5, label=_cat, rasterized=True,
                    )

            _n_cats = obs[_col].nunique()
            for _ax in _axes:
                if _n_cats <= 10:
                    _ax.legend(fontsize=8, markerscale=5, loc="best", frameon=True)
                else:
                    _ax.legend(
                        fontsize=6, markerscale=4, loc="center left",
                        bbox_to_anchor=(1.01, 0.5), frameon=True, ncol=1,
                    )

        for _ax, (_title, _, _var, _) in zip(_axes, _panels):
            _ax.set_title(_title, fontsize=13)
            _ax.set_xlabel(f"PC1 ({_var[0]*100:.1f}%)")
            _ax.set_ylabel(f"PC2 ({_var[1]*100:.1f}%)")

        _fig.suptitle(
            f"HK Gene PCA — colored by {_annotation}" if _COL_MAP.get(_annotation) else "HK Gene PCA",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        return fig_to_html(_fig)

    if _annotation == "None":
        _viz = mo.lazy(_render_hk_pca, show_loading_indicator=True)
    else:
        _viz = _render_hk_pca()

    _viz
    return


# ── Plot Gallery ─────────────────────────────────────────────────────────────


@app.cell
def _(Path, mo):
    from scribe import paths as _paths
    plot_dir = _paths.get_plots_dir()

    _PLOT_NAMES = {
        "HK Analysis: Violin plots": "hk_analysis/housekeeping_violin.png",
        "HK Analysis: Heatmap": "hk_analysis/housekeeping_heatmap.png",
        "HK Analysis: Housekeeping PCA": "hk_analysis/housekeeping_pca.png",
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
        mo.md("Browse existing housekeeping gene analysis plots."),
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
    *SCRIBE — Single-Cell RNA Interpretable Biomarker Explorer*
    """)
    return


if __name__ == "__main__":
    app.run()
