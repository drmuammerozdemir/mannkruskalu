import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kruskal, shapiro, levene, mannwhitneyu, chi2_contingency, fisher_exact
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
import pyreadstat
import tempfile
import itertools
import math
from io import BytesIO

# ===================== Helpers (formatlama) =====================
def format_p(p):
    try:
        if pd.isna(p):
            return ""
        return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")
    except Exception:
        return ""

def format_p_label(p):
    s = format_p(p)
    if not s:
        return ""
    return f"p {s}" if s.startswith("<") else f"p = {s}"

def fmt_num(x, nd=2):
    try:
        return f"{x:.{nd}f}".replace(".", ",")
    except Exception:
        return ""

# ---- Eski export yardımcıları ----
def fig_to_bytes(fig, fmt):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf

def export_fig_bytes(fig, fmt, width_px=None, height_px=None, dpi=300):
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

# ===================== Binary (0/1, 1/2) yardımcıları =====================
def is_binary_like(s: pd.Series) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna().unique()
    sv = set(vals)
    return sv.issubset({0, 1}) or sv.issubset({1, 2})

def binary_summary_n_pct(s: pd.Series, present_code: int = 1) -> str:
    x = pd.to_numeric(s, errors="coerce")
    n = int(x.notna().sum())
    n1 = int((x == present_code).sum())
    pct = (n1 / n * 100) if n > 0 else 0.0
    return f"{n1} ({pct:.1f}%)".replace(".", ",")

def chi2_rx2(dsub: pd.DataFrame, group_var: str, col_name: str,
             present_code: int = 1, monte_carlo_B: int = None, seed: int = 42):
    """
    Rx2 (örn. 4x2) tablo için Pearson ki-kare p-değeri (gerekirse 2x2'de Fisher/Yates),
    opsiyonel Monte Carlo p ve Cramér's V döndürür.
    """
    g = dsub[group_var].astype(str)
    x = pd.to_numeric(dsub[col_name], errors="coerce")
    d = pd.DataFrame({group_var: g, col_name: x}).dropna()
    if d.empty:
        return np.nan, "Chi-square (Pearson)", np.nan, pd.DataFrame()

    g = d[group_var]
    x = d[col_name]

    x01 = (x == present_code).astype(int)
    tab = pd.crosstab(g, x01).reindex(columns=[0, 1], fill_value=0)
    obs = tab.values
    n = obs.sum()

    chi2, p, dof, exp = chi2_contingency(obs, correction=False)
    method = "Chi-square (Pearson)"

    # 2x2 özel durum: küçük beklenen varsa Fisher, değilse Yates
    if obs.shape == (2, 2):
        if (exp < 5).any():
            _, p = fisher_exact(obs)
            method = "Fisher exact"
        else:
            chi2, p, _, _ = chi2_contingency(obs, correction=True)
            method = "Chi-square (Yates)"

    # Rx2 ve çok seyrek tabloysa Monte Carlo p (opsiyonel)
    elif monte_carlo_B:
        rng = np.random.default_rng(seed)
        ge = 0
        chi2_obs = chi2
        x01_vals = x01.values
        g_vals = g.values
        for _ in range(int(monte_carlo_B)):
            perm = rng.permutation(x01_vals)
            t = pd.crosstab(g_vals, perm).reindex(columns=[0, 1], fill_value=0).values
            chi2_perm = chi2_contingency(t, correction=False)[0]
            ge += (chi2_perm >= chi2_obs)
        p = (ge + 1) / (int(monte_carlo_B) + 1)
        method = "Chi-square (Monte Carlo p)"

    # Cramér's V
    r, c = obs.shape
    k = min(r - 1, c - 1)
    V = np.sqrt(chi2 / (n * k)) if k > 0 and n > 0 else np.nan
    return p, method, V, tab

# ===================== UI =====================
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
    test_vars = st.sidebar.multiselect("Select Test Variables (numeric or binary)", options=df.columns)

    test_choice = st.sidebar.radio("Default omnibus test (for continuous)", ["Auto", "Mann-Whitney U", "Kruskal-Wallis"])
    show_nonsignificant = st.sidebar.checkbox("Show Non-Significant p-values (p > 0.05)", value=False)

    # ---- Export settings (resolution & DPI) ----
    with st.sidebar.expander("Export settings (resolution)"):
        export_width_px = st.number_input("Width (pixels)", min_value=200, value=1600, step=50)
        export_height_px = st.number_input("Height (pixels)", min_value=200, value=1200, step=50)
        export_dpi = st.number_input("DPI", min_value=72, value=300, step=1,
                                     help="For PNG/JPG. PDF ignores DPI (vector).")

    # ---- Binary ayarları ----
    with st.sidebar.expander("Binary settings (0/1, 1/2)"):
        present_code_ui = st.selectbox("Code that means 'present' (var=1)", options=[1, 2], index=1 if 2 in df.columns else 0)
        use_mc = st.checkbox("Use Monte Carlo p for Rx2 if sparse", value=False)
        mc_B = st.number_input("Permutations (B)", min_value=1000, value=20000, step=1000)

    # ---- Data types ----
    df[group_var] = df[group_var].astype(str)
    group_labels = sorted(df[group_var].dropna().unique())
    group_n = {g: int(df[df[group_var] == g].shape[0]) for g in group_labels}
    group_headers = [f"{g} (n={group_n[g]})" for g in group_labels]

    # ===================== SUMMARY TABLE =====================
    if analysis_type == "Summary Table":

        # --- Özet biçimleri (GLOBAL varsayılan + per-variable) ---
        AVAILABLE_SUMMARY_FORMATS = ["Median [IQR]", "Mean ± SD", "Median [Min–Max]"]
        summary_format = st.sidebar.selectbox(
            "Default summary display",
            options=AVAILABLE_SUMMARY_FORMATS,
            index=0
        )

        # Bölüm ve sıra editörü
        default_sections = "# Section 1\n" + ("\n".join(test_vars) if test_vars else "")
        sections_text = st.text_area(
            "Sections & order (use # for section headers, and 'Column|Label' for alias)",
            value=default_sections,
            height=220
        )

        # ---------- Yardımcı: seçilen formata göre grup özet stringi ----------
        def group_summary_str(arr, fmt):
            arr = np.array(arr, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return ""
            if fmt == "Median [IQR]":
                med = np.median(arr)
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                return f"{fmt_num(med)} [{fmt_num(q1)}–{fmt_num(q3)}]"
            elif fmt == "Mean ± SD":
                mean = np.mean(arr)
                sd = np.std(arr, ddof=1) if arr.size > 1 else 0.0
                return f"{fmt_num(mean)} ± {fmt_num(sd)}"
            elif fmt == "Median [Min–Max]":
                med = np.median(arr)
                mn = np.min(arr)
                mx = np.max(arr)
                return f"{fmt_num(med)} [{fmt_num(mn)}–{fmt_num(mx)}]"
            else:
                med = np.median(arr)
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                return f"{fmt_num(med)} [{fmt_num(q1)}–{fmt_num(q3)}]"

        # ---------- Parse lines (section titles & variables) ----------
        lines = [ln.strip() for ln in sections_text.splitlines() if ln.strip() != ""]
        items = []  # [{'type':'header','label':...} or {'type':'var','name':..., 'label':...}]
        for line in lines:
            if line.startswith("#"):
                items.append({"type": "header", "label": line.lstrip("#").strip()})
                continue
            if "|" in line:
                col_name, disp = [p.strip() for p in line.split("|", 1)]
            else:
                col_name, disp = line, line
            items.append({"type": "var", "name": col_name, "label": disp})

        # ---------- Sağ panel: Per-variable display + test seçimi ----------
        planned_vars = [it["name"] for it in items if it["type"] == "var" and it["name"] in df.columns]

        if "var_fmt" not in st.session_state:
            st.session_state["var_fmt"] = {}
        if "var_test" not in st.session_state:
            st.session_state["var_test"] = {}

        for v in planned_vars:
            st.session_state["var_fmt"].setdefault(v, summary_format)
            st.session_state["var_test"].setdefault(v, "Auto")

        table_col, cfg_col = st.columns([3, 1], gap="large")
        with cfg_col:
            st.markdown("**Per-variable settings**")
            for v in planned_vars:
                key_fmt = f"fmt_{hash(v)}"
                current_fmt = st.session_state["var_fmt"].get(v, summary_format)
                st.session_state["var_fmt"][v] = st.selectbox(
                    f"{v} — display", AVAILABLE_SUMMARY_FORMATS,
                    index=AVAILABLE_SUMMARY_FORMATS.index(current_fmt),
                    key=key_fmt
                )

                # Test seçenekleri, değişken tipine ve grup sayısına göre
                is_bin = is_binary_like(df[v])
                if is_bin and len(group_labels) == 2:
                    test_opts = ["Auto", "Chi-square", "Fisher exact", "Mann-Whitney U"]
                elif is_bin and len(group_labels) > 2:
                    test_opts = ["Auto", "Chi-square", "Kruskal-Wallis"]  # Fisher 2x2 içindir
                else:
                    # Sürekli değişken
                    test_opts = ["Auto", "Mann-Whitney U", "Kruskal-Wallis"]

                key_test = f"test_{hash(v)}"
                cur_test = st.session_state["var_test"].get(v, "Auto")
                if cur_test not in test_opts:
                    cur_test = "Auto"
                st.session_state["var_test"][v] = st.selectbox(
                    f"{v} — test", test_opts,
                    index=test_opts.index(cur_test),
                    key=key_test
                )

        # ---------- Üstsimge (a,b,c,...) atama için yardımcılar ----------
        supers = ['ᵃ','ᵇ','ᶜ','ᵈ','ᵉ','ᶠ','ᵍ','ʰ','ᶦ','ʲ','ᵏ','ˡ','ᵐ','ⁿ','ᵒ','ᵖ','ʳ','ˢ','ᵗ','ᵘ','ᵛ','ʷ','ˣ','ʸ','ᶻ']
        method_notes = {}  # method_label -> superscript
        def note_for(method_label: str):
            if not method_label:
                return ""
            if method_label not in method_notes:
                method_notes[method_label] = supers[len(method_notes) % len(supers)]
            return method_notes[method_label]

        # ---------- Tabloyu oluştur ----------
        rows = []
        header_idx = []  # Stil için bölüm satır indeksleri
        columns = ["Parameter"] + group_headers + ["p-value"]

        with table_col:
            for it in items:
                if it["type"] == "header":
                    rows.append([it["label"]] + [""] * (len(group_headers) + 1))
                    header_idx.append(len(rows) - 1)
                    continue

                col_name = it["name"]
                disp = it["label"]

                if col_name not in df.columns:
                    rows.append([f"[Missing: {disp}]"] + [""] * (len(group_headers) + 1))
                    continue

                dsub = df[[group_var, col_name]].dropna()
                dsub = dsub[dsub[group_var].notna()]
                grp_data = [dsub[dsub[group_var] == g][col_name].values for g in group_labels]
                is_bin = is_binary_like(dsub[col_name])

                # Özet
                if is_bin:
                    present_code = int(present_code_ui)
                    summaries = []
                    for gname in group_labels:
                        s_grp = dsub.loc[dsub[group_var] == gname, col_name]
                        summaries.append(binary_summary_n_pct(s_grp, present_code))
                else:
                    fmt_for_var = st.session_state["var_fmt"].get(col_name, summary_format)
                    summaries = [group_summary_str(arr, fmt_for_var) for arr in grp_data]

                # ---- Test seçimi (elle/Auto) ve p ----
                chosen_test = st.session_state["var_test"].get(col_name, "Auto")
                p_val = np.nan
                method_label = ""

                if is_bin:
                    # Binary veri
                    if chosen_test == "Auto" or chosen_test == "Chi-square":
                        p_val, method_label, _, _ = chi2_rx2(
                            dsub, group_var, col_name,
                            present_code=int(present_code_ui),
                            monte_carlo_B=int(mc_B) if use_mc else None
                        )
                    elif chosen_test == "Fisher exact":
                        # 2x2 şart
                        if len(group_labels) == 2:
                            # Fisher exact
                            x = pd.to_numeric(dsub[col_name], errors="coerce")
                            tab = pd.crosstab(dsub[group_var].astype(str), (x == int(present_code_ui)).astype(int)).reindex(columns=[0,1], fill_value=0).values
                            try:
                                _, p_val = fisher_exact(tab)
                                method_label = "Fisher exact"
                            except Exception:
                                p_val = np.nan
                                method_label = "Fisher exact (not applicable)"
                        else:
                            p_val = np.nan
                            method_label = "Fisher exact (not applicable)"
                    elif chosen_test == "Mann-Whitney U":
                        if len(group_labels) == 2:
                            try:
                                x0 = pd.to_numeric(grp_data[0], errors="coerce")
                                x1 = pd.to_numeric(grp_data[1], errors="coerce")
                                _, p_val = mannwhitneyu(x0, x1, alternative="two-sided")
                                method_label = "Mann–Whitney U"
                            except Exception:
                                p_val = np.nan
                                method_label = "Mann–Whitney U"
                        else:
                            p_val = np.nan
                            method_label = "Mann–Whitney U (not applicable)"
                    elif chosen_test == "Kruskal-Wallis":
                        try:
                            arrays = [pd.to_numeric(a, errors="coerce") for a in grp_data]
                            _, p_val = kruskal(*arrays)
                            method_label = "Kruskal–Wallis"
                        except Exception:
                            p_val = np.nan
                            method_label = "Kruskal–Wallis"
                else:
                    # Sürekli/ordinal veri
                    if chosen_test == "Auto":
                        if len(group_labels) == 2:
                            chosen_test_eff = "Mann–Whitney U"
                            try:
                                _, p_val = mannwhitneyu(grp_data[0], grp_data[1], alternative="two-sided")
                            except Exception:
                                p_val = np.nan
                        else:
                            chosen_test_eff = "Kruskal–Wallis"
                            try:
                                _, p_val = kruskal(*grp_data)
                            except Exception:
                                p_val = np.nan
                        method_label = chosen_test_eff
                    elif chosen_test == "Mann-Whitney U":
                        if len(group_labels) == 2:
                            try:
                                _, p_val = mannwhitneyu(grp_data[0], grp_data[1], alternative="two-sided")
                            except Exception:
                                p_val = np.nan
                            method_label = "Mann–Whitney U"
                        else:
                            p_val = np.nan
                            method_label = "Mann–Whitney U (not applicable)"
                    elif chosen_test == "Kruskal-Wallis":
                        try:
                            _, p_val = kruskal(*grp_data)
                        except Exception:
                            p_val = np.nan
                        method_label = "Kruskal–Wallis"
                    elif chosen_test == "Chi-square":
                        p_val = np.nan
                        method_label = "Chi-square (not applicable)"
                    elif chosen_test == "Fisher exact":
                        p_val = np.nan
                        method_label = "Fisher exact (not applicable)"

                # p yazdırma + üstsimge
                if not show_nonsignificant and (pd.isna(p_val) or p_val > 0.05):
                    p_disp = ""
                else:
                    p_disp = format_p(p_val)
                    if p_disp:
                        sup = note_for(method_label)
                        p_disp = p_disp + sup

                rows.append([disp] + summaries + [p_disp])

            summary_df = pd.DataFrame(rows, columns=columns)

            # ---------- STYLING (Arial, hizalama, bölüm başlığı kalın/çizgi) ----------
            def highlight_sections(row):
                if row.name in header_idx:
                    return ["font-weight: bold; border-top: 1px solid black; background-color: #f7f7f7;"] * len(row)
                return [""] * len(row)

            styler = (
                summary_df
                .style
                .format(na_rep="")
                .apply(highlight_sections, axis=1)
                .set_properties(**{"font-family": "Arial", "font-size": "12pt"})
                .set_properties(subset=pd.IndexSlice[:, ["Parameter"]], **{"text-align": "left"})
                .set_properties(subset=pd.IndexSlice[:, group_headers + ["p-value"]], **{"text-align": "right"})
            )

            st.dataframe(styler, use_container_width=True)

            # ---------- Dipnotlar ----------
            if method_notes:
                lines = []
                # Kullanıcıya anlaşılır dipnotlar
                for label, sup in method_notes.items():
                    nice = label.replace("--", "–")
                    lines.append(f"{sup} {nice}")
                st.markdown("**Notes:**  \n" + "  \n".join(lines))

    # ===================== STATISTICAL PLOTS =====================
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
            group_labels_plot = sorted(df_clean[group_var].unique())
            group_data = [df_clean[df_clean[group_var] == grp][test_var] for grp in group_labels_plot]

            ax = axes[idx]
            sns.stripplot(
                data=df_clean,
                x=group_var,
                y=test_var,
                jitter=True,
                ax=ax,
                palette=palette_choice,
                order=group_labels_plot
            )

            # Median line + min-max whiskers
            for i, group in enumerate(group_labels_plot):
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
                test_to_use = "Mann-Whitney U" if len(group_labels_plot) == 2 else "Kruskal-Wallis"
            else:
                test_to_use = test_choice

            # p-values + brackets
            if test_to_use == "Mann-Whitney U" and len(group_labels_plot) == 2:
                u_stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative="two-sided")
                ymax = df_clean[test_var].max()
                ypos = ymax + 0.1 * (ymax if ymax != 0 else 1)
                ax.plot([0, 0, 1, 1], [ypos, ypos * 1.05, ypos * 1.05, ypos], lw=1.5, c="k")
                if show_nonsignificant or p_value <= 0.05:
                    ax.text(0.5, ypos + 0.07*ypos, format_p_label(p_value), ha='center', va='bottom')
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
                        x1 = group_labels_plot.index(g1)
                        x2 = group_labels_plot.index(g2)
                        y = ypos + visible * spacing
                        ax.plot([x1, x1, x2, x2], [y, y + 0.3 * spacing, y + 0.3 * spacing, y], lw=1.5, c="k")
                        label = format_p_label(pval)
                        if label:
                            ax.text((x1 + x2) / 2, y + spacing * 0.35, label, ha='center', va='bottom')
                        visible += 1

            # Axis labels & titles
            ax.set_xticks(range(len(group_labels_plot)))
            ax.set_xticklabels(custom_labels if custom_labels else group_labels_plot)
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
