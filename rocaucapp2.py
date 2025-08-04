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


def fig_to_bytes(fig, fmt):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight')
    buf.seek(0)
    return buf


st.set_page_config(page_title="Biomarker Analysis Dashboard", layout="wide")
st.title('🔬 Biomarker Analysis Dashboard (.csv, .sav)')

uploaded_file = st.file_uploader("Upload CSV or SPSS (.sav)", type=["csv", "sav"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".sav"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        df, meta = pyreadstat.read_sav(tmp_file_path)

    st.subheader("Edit Your Data (Optional)")
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    st.sidebar.header("Options")

    analysis_type = st.sidebar.radio("Select Analysis", ["Statistical Plots", "Summary Table"])

    palette_options = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
    palette_choice = st.sidebar.selectbox("Color Palette", options=palette_options)

    group_var = st.sidebar.selectbox("Select Group Variable (categorical)", options=df.columns)
    test_vars = st.sidebar.multiselect("Select Test Variables (numeric)", options=df.columns)

    test_choice = st.sidebar.radio("Select Test", ["Auto", "Mann-Whitney U", "Kruskal-Wallis"])
    show_nonsignificant = st.sidebar.checkbox("Show Non-Significant p-values (p > 0.05)", value=False)

    df[group_var] = df[group_var].astype(str)
    group_labels = sorted(df[group_var].dropna().unique())

    if analysis_type == "Summary Table":
        rows = []
        for var in test_vars:
            df_clean = df[[group_var, var]].dropna()
            df_clean = df_clean[df_clean[group_var].notna()]
            group_labels = sorted(df_clean[group_var].unique())
            group_data = [df_clean[df_clean[group_var] == grp][var] for grp in group_labels]
            summaries = []
            for g in group_data:
                med = np.median(g)
                gmin = np.min(g)
                gmax = np.max(g)
                summaries.append(f"{med:.2f} ({gmin:.2f}-{gmax:.2f})")
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
                p_display = f"{p_value:.3f}"
            rows.append([var] + summaries + [p_display])
        columns = ["Parameter"] + group_labels + ["p-value"]
        summary_df = pd.DataFrame(rows, columns=columns)
        st.dataframe(summary_df.style.format(na_rep="").set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]))

    if analysis_type == "Statistical Plots":
        title_input = st.text_input("Figure Title", value="Statistical Comparison")
        xaxis_label = st.text_input("X Axis Label", value=group_var)
        yaxis_label = st.text_input("Y Axis Label", value="Value")
        custom_labels_input = st.text_input("Custom X Axis Group Labels (comma-separated)", value="")
        subplot_titles_default = ", ".join(test_vars)
        subplot_titles_input = st.text_area("Subfigure Titles (comma-separated, leave empty for automatic)", value=subplot_titles_default)

        if custom_labels_input.strip():
            custom_labels = [label.strip() for label in custom_labels_input.split(",")]
        else:
            custom_labels = None

        if subplot_titles_input.strip():
            subplot_titles = [label.strip() for label in subplot_titles_input.split(",")]
        else:
            subplot_titles = test_vars

        ncols = math.ceil(math.sqrt(len(test_vars)))
        nrows = math.ceil(len(test_vars) / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.array(axes).reshape(-1)
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        for idx, test_var in enumerate(test_vars):
            df_clean = df[[group_var, test_var]].dropna()
            df_clean = df_clean[df_clean[group_var].notna()]
            group_labels = sorted(df_clean[group_var].unique())
            group_data = [df_clean[df_clean[group_var] == grp][test_var] for grp in group_labels]

            ax = axes[idx]
            sns.stripplot(data=df_clean, x=group_var, y=test_var, jitter=True, ax=ax, palette=palette_choice, order=group_labels)

            for i, group in enumerate(group_labels):
                group_values = df_clean[df_clean[group_var] == group][test_var]
                center_value = group_values.median()
                ymin = group_values.min()
                ymax = group_values.max()
                ax.hlines(y=center_value, xmin=i - 0.2, xmax=i + 0.2, colors='k', linewidth=2)
                ax.vlines(x=i, ymin=ymin, ymax=ymax, colors='k', linewidth=2)
                ax.hlines(y=ymin, xmin=i - 0.1, xmax=i + 0.1, colors='k', linewidth=2)
                ax.hlines(y=ymax, xmin=i - 0.1, xmax=i + 0.1, colors='k', linewidth=2)

            if test_choice == "Auto":
                test_to_use = "Mann-Whitney U" if len(group_labels) == 2 else "Kruskal-Wallis"
            else:
                test_to_use = test_choice

            if test_to_use == "Mann-Whitney U" and len(group_labels) == 2:
                u_stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative="two-sided")
                ypos = df_clean[test_var].max() * 1.1
                ax.plot([0, 0, 1, 1], [ypos, ypos + 0.05*ypos, ypos + 0.05*ypos, ypos], lw=1.5, c='k')
                if show_nonsignificant or p_value <= 0.05:
                    ax.text(0.5, ypos + 0.07*ypos, f"p = {p_value:.3f}", ha='center', va='bottom')
            elif test_to_use == "Kruskal-Wallis" or len(group_labels) > 2:
                h_stat, p_value = kruskal(*group_data)
                posthoc = sp.posthoc_dunn(df_clean, val_col=test_var, group_col=group_var, p_adjust='bonferroni')
                pairs = list(itertools.combinations(posthoc.columns, 2))
                ypos = df_clean[test_var].max() * 1.05
                spacing = (df_clean[test_var].max() - df_clean[test_var].min()) * 0.1
                visible_pairs = 0
                for (g1, g2) in pairs:
                    pval = posthoc.loc[g1, g2]
                    if not show_nonsignificant and pval > 0.05:
                        continue
                    x1 = group_labels.index(g1)
                    x2 = group_labels.index(g2)
                    y = ypos + visible_pairs * spacing
                    ax.plot([x1, x1, x2, x2], [y, y + spacing * 0.3, y + spacing * 0.3, y], lw=1.5, c='k')
                    ax.text((x1 + x2) / 2, y + spacing * 0.35, f"p = {pval:.3f}", ha='center', va='bottom')
                    visible_pairs += 1

if custom_labels:
    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels(custom_labels)
else:
    ax.set_xticks(range(len(group_labels)))
    ax.set_xticklabels(group_labels)

    subtitle = subplot_titles[idx] if idx < len(subplot_titles) else test_var
    ax.set_title(subtitle)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)

        for ax in axes[len(test_vars):]:
            ax.axis('off')

        fig.suptitle(title_input, fontsize=16)
        st.pyplot(fig)

        st.download_button("Download PNG", fig_to_bytes(fig, "png"), file_name="figure.png")
        st.download_button("Download JPG", fig_to_bytes(fig, "jpg"), file_name="figure.jpg")

        st.download_button("Download PDF", fig_to_bytes(fig, "pdf"), file_name="figure.pdf")




