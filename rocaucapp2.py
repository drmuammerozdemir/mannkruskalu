import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kruskal, shapiro, levene, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
import pyreadstat
import tempfile
import itertools
import math
from io import BytesIO

def format_p(p):
    try:
        if pd.isna(p):
            return ""
        return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")
    except Exception:
        return ""

# p-etiketi: "<0,001" ise eşittir yok, diğerlerinde "p = x,xxx"
def format_p_label(p):
    s = format_p(p)
    if not s:
        return ""
    return f"p {s}" if s.startswith("<") else f"p = {s}"

# ---- Helpers ----
def fig_to_bytes(fig, fmt):
    """Legacy helper (kept for compatibility). Saves current fig as-is."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf

def export_fig_bytes(fig, fmt, width_px=None, height_px=None, dpi=300):
    """
    Export the given matplotlib figure to bytes with custom pixel size and dpi.
    We temporarily change the figure size (inches = pixels / dpi) and restore it.
    - For PNG/JPG: uses dpi and target size.
    - For PDF: vector; uses size in inches (dpi ignored by PDF backends).
    """
    orig_size_in = fig.get_size_inches()
    try:
        if width_px and height_px and dpi and width_px > 0 and height_px > 0 and dpi > 0:
            target_w_in = width_px / dpi
            target_h_in = height_px / dpi
            fig.set_size_inches(target_w_in, target_h_in)

        buf = BytesIO()
        if fmt.lower() in ("png", "jpg", "jpeg"):
            fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        elif fmt.lower() == "pdf":
            fig.savefig(buf, format=fmt, bbox_inches="tight")
        else:
            fig.savefig(buf, format=fmt, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()
    finally:
        fig.set_size_inches(orig_size_in)

# ---- UI ----
st.set_page_config(page_title="Biomarker Analysis Dashboard", layout="wide")
st.title("🔬 Biomarker Analysis Dashboard (.csv, .sav)")

uploaded_file = st.file_uploader("Upload CSV or SPSS (.sav)", type=["csv", "sav"])

if uploaded_file:
    # ---- Load data ----
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".sav"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        df, meta = pyreadstat.read_sav(tmp_file_path)

    st.subheader("Edit Your Data (Optional)")
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    # ---- Sidebar controls ----
    st.sidebar.header("Options")
    analysis_type = st.sidebar.radio("Select Analysis", ["Statistical Plots", "Summary Table"])

    palette_options = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
    palette_choice = st.sidebar.selectbox("Color Palette", options=palette_options)

    # ---- Post-hoc ve düzeltme yöntemi seçimi ----
    posthoc_test_choice = st.sidebar.selectbox(
        "Select Post-hoc Test",
        options=["Dunn"]  # İleride diğer testler eklenebilir
    )

    p_adjust_methods = [
        "bonferroni", "holm", "holm-sidak", "sidak",
        "fdr_bh", "fdr_by", "hochberg", "hommel"
    ]
    p_adjust_choice = st.sidebar.selectbox(
        "Select p-value Adjustment Method",
        options=p_adjust_methods,
        index=0  # Varsayılan Bonferroni
    )

    group_var = st.sidebar.selectbox("Select Group Variable (categorical)", options=df.columns)
    test_vars = st.sidebar.multiselect("Select Test Variables (numeric)", options=df.columns)

    test_choice = st.sidebar.radio("Select Test", ["Auto", "Mann-Whitney U", "Kruskal-Wallis"])
    show_nonsignificant = st.sidebar.checkbox("Show Non-Significant p-values (p > 0.05)", value=False)

    # ---- Export settings (resolution & DPI) ----
    with st.sidebar.expander("Export settings (resolution)"):
        export_width_px = st.number_input("Width (pixels)", min_value=200, value=1600, step=50)
        export_height_px = st.number_input("Height (pixels)", min_value=200, value=1200, step=50)
        export_dpi = st.number_input("DPI", min_value=72, value=300, step=1,
                                     help="For PNG/JPG. PDF ignores DPI (vector).")

    # ---- Data types ----
    df[group_var] = df[group_var].astype(str)
    group_labels = sorted(df[group_var].dropna().unique())

    # ---- Summary Table ----
    if analysis_type == "Summary Table":
        rows = []
        for var in test_vars:
            df_clean = df[[group_var, var]].dropna()
            df_clean = df_clean[df_clean[group_var].notna()]
            group_labels = sorted(df_clean[group_var].unique())
            group_data = [df_clean[df_clean[group_var] == grp][var] for grp in group_labels]

            # Per-group summary (median, min-max)
            summaries = []
            for g in group_data:
                med = np.median(g)
                gmin = np.min(g)
                gmax = np.max(g)
                summaries.append(f"{med:.2f} ({gmin:.2f}-{gmax:.2f})")

            # Auto/Manual test selection
            if test_choice == "Auto":
                test_to_use = "Mann-Whitney U" if len(group_labels) == 2 else "Kruskal-Wallis"
            else:
                test_to_use = test_choice

            if test_to_use == "Mann-Whitney U" and len(group_labels) == 2:
                u_stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative="two-sided")
            else:
                h_stat, p_value = kruskal(*group_data)

            if not show_nonsignificant and p_value > 0.05:
                p_display = ""
            else:
                p_display = format_p(p_value)  # <-- formatlı yaz

            rows.append([var] + summaries + [p_display])

        columns = ["Parameter"] + group_labels + ["p-value"]
        summary_df = pd.DataFrame(rows, columns=columns)
        st.dataframe(
            summary_df.style.format(na_rep="").set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]}]
            )
        )

    # ---- Statistical Plots ----
    if analysis_type == "Statistical Plots":
        title_input = st.text_input("Figure Title", value="Statistical Comparison")
        xaxis_label = st.text_input("X Axis Label", value=group_var)
        yaxis_label = st.text_input("Y Axis Label", value="Value")
        custom_labels_input = st.text_input("Custom X Axis Group Labels (comma-separated)", value="")
        subplot_titles_default = ", ".join(test_vars)
        subplot_titles_input = st.text_area(
            "Subfigure Titles (comma-separated, leave empty for automatic)",
            value=subplot_titles_default
        )

        custom_labels = [s.strip() for s in custom_labels_input.split(",")] if custom_labels_input.strip() else None
        subplot_titles = [s.strip() for s in subplot_titles_input.split(",")] if subplot_titles_input.strip() else test_vars

        ncols = math.ceil(math.sqrt(len(test_vars))) if len(test_vars) > 0 else 1
        nrows = math.ceil(len(test_vars) / ncols) if len(test_vars) > 0 else 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.array(axes).reshape(-1)
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        for idx, test_var in enumerate(test_vars):
            df_clean = df[[group_var, test_var]].dropna()
            df_clean = df_clean[df_clean[group_var].notna()]
            group_labels = sorted(df_clean[group_var].unique())
            group_data = [df_clean[df_clean[group_var] == grp][test_var] for grp in group_labels]

            ax = axes[idx]
            sns.stripplot(
                data=df_clean,
                x=group_var,
                y=test_var,
                jitter=True,
                ax=ax,
                palette=palette_choice,
                order=group_labels
            )

            # Median line + min-max whiskers
            for i, group in enumerate(group_labels):
                group_values = df_clean[df_clean[group_var] == group][test_var]
                center_value = group_values.median()
                ymin = group_values.min()
                ymax = group_values.max()
                ax.hlines(y=center_value, xmin=i - 0.2, xmax=i + 0.2, colors="k", linewidth=2)
                ax.vlines(x=i, ymin=ymin, ymax=ymax, colors="k", linewidth=2)
                ax.hlines(y=ymin, xmin=i - 0.1, xmax=i + 0.1, colors="k", linewidth=2)
                ax.hlines(y=ymax, xmin=i - 0.1, xmax=i + 0.1, colors="k", linewidth=2)

            # Decide test
            if test_choice == "Auto":
                test_to_use = "Mann-Whitney U" if len(group_labels) == 2 else "Kruskal-Wallis"
            else:
                test_to_use = test_choice

            # p-values + brackets
            if test_to_use == "Mann-Whitney U" and len(group_labels) == 2:
                u_stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative="two-sided")
                ymax = df_clean[test_var].max()
                ypos = ymax + 0.1 * (ymax if ymax != 0 else 1)
                ax.plot([0, 0, 1, 1], [ypos, ypos * 1.05, ypos * 1.05, ypos], lw=1.5, c="k")
                if show_nonsignificant or p_value <= 0.05:
                    ax.text(0.5, ypos + 0.07*ypos, format_p_label(p_value), ha='center', va='bottom')  # <-- düzeltildi
            else:
                h_stat, p_value = kruskal(*group_data)
                if posthoc_test_choice == "Dunn":
                    posthoc = sp.posthoc_dunn(
                        df_clean,
                        val_col=test_var,
                        group_col=group_var,
                        p_adjust=p_adjust_choice
                    )
                    pairs = list(itertools.combinations(posthoc.columns, 2))
                    datamax = df_clean[test_var].max()
                    datamin = df_clean[test_var].min()
                    rng = max(datamax - datamin, 1e-9)
                    ypos = datamax + 0.05 * rng
                    spacing = 0.1 * rng
                    visible = 0
                    for (g1, g2) in pairs:
                        pval = float(posthoc.loc[g1, g2])
                        if not show_nonsignificant and pval > 0.05:
                            continue
                        x1 = group_labels.index(g1)
                        x2 = group_labels.index(g2)
                        y = ypos + visible * spacing
                        ax.plot([x1, x1, x2, x2], [y, y + 0.3 * spacing, y + 0.3 * spacing, y], lw=1.5, c="k")
                        label = format_p_label(pval)  # <-- düzeltildi
                        if label:
                            ax.text((x1 + x2) / 2, y + spacing * 0.35, label, ha='center', va='bottom')
                        visible += 1

            # Axis labels & titles
            ax.set_xticks(range(len(group_labels)))
            ax.set_xticklabels(custom_labels if custom_labels else group_labels)
            subtitle = subplot_titles[idx] if idx < len(subplot_titles) else test_var
            ax.set_title(subtitle)
            ax.set_xlabel(xaxis_label)
            ax.set_ylabel(yaxis_label)

        # Turn off unused axes
        for ax in axes[len(test_vars):]:
            ax.axis("off")

        fig.suptitle(title_input, fontsize=16)
        st.pyplot(fig)

        # ---- Downloads with resolution ----
        png_bytes = export_fig_bytes(fig, "png", export_width_px, export_height_px, export_dpi)
        jpg_bytes = export_fig_bytes(fig, "jpg", export_width_px, export_height_px, export_dpi)
        pdf_bytes = export_fig_bytes(fig, "pdf", export_width_px, export_height_px, export_dpi)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download PNG",
                data=png_bytes,
                file_name=f"figure_{export_width_px}x{export_height_px}_{export_dpi}dpi.png"
            )
        with col2:
            st.download_button(
                "Download JPG",
                data=jpg_bytes,
                file_name=f"figure_{export_width_px}x{export_height_px}_{export_dpi}dpi.jpg"
            )
        with col3:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"figure_{export_width_px}x{export_height_px}.pdf"
            )
