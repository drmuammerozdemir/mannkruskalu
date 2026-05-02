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
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from statsmodels.stats.multitest import multipletests


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


# ===================== Yeni Helperlar: CI ve LOO =====================
def get_mean_diff_ci(data1, data2, alpha=0.05):
    """İki grup arasındaki ortalama farkı için %95 GA hesaplar (Welch's t-interval)."""
    try:
        d1 = DescrStatsW(data1)
        d2 = DescrStatsW(data2)
        cm = CompareMeans(d1, d2)
        lower, upper = cm.tconfint_diff(alpha=alpha, usevar='unequal')
        return f"[{lower:.2f}, {upper:.2f}]"
    except:
        return "-"


def run_loo_sensitivity(dsub, group_var, col_name, original_p, test_type, present_code=1):
    """
    Leave-One-Out (LOO) Analizi:
    Verisetindeki Minimum ve Maksimum değerleri çıkararak testi tekrarlar.
    Eğer p-değeri anlamlılık sınırını (0.05) değiştiriyorsa 'Sensitive' döner.
    """
    if pd.isna(original_p):
        return "-"

    # Sadece anlamlı veya sınırdaki sonuçlar için analiz yapalım (Performans için)
    # Ancak kullanıcı her şeyi görmek isterse buradaki if kaldırılabilir.
    threshold = 0.05
    is_sig = original_p < threshold

    # Hangi testi kullanacağız?
    def run_test(subset):
        grps = [subset[subset[group_var] == g][col_name] for g in subset[group_var].unique()]
        if len(grps) < 2: return np.nan

        try:
            if test_type == "Mann-Whitney U":
                _, p = mannwhitneyu(grps[0], grps[1], alternative="two-sided")
            elif test_type == "Kruskal-Wallis":
                _, p = kruskal(*grps)
            elif "Chi" in test_type or "Fisher" in test_type:
                # Kategorik için LOO çok anlamlı olmayabilir ama Min/Max yerine rastgele drop denenebilir
                # Şimdilik kategorik için LOO pas geçiyoruz.
                return original_p
            else:  # Auto (Varsayalım Mann-Whitney)
                _, p = mannwhitneyu(grps[0], grps[1], alternative="two-sided")
            return p
        except:
            return original_p

    # Veriyi sırala
    d_sorted = dsub.sort_values(by=col_name)

    # 1. Minimumu çıkar
    d_no_min = d_sorted.iloc[1:]
    p_no_min = run_test(d_no_min)

    # 2. Maksimumu çıkar
    d_no_max = d_sorted.iloc[:-1]
    p_no_max = run_test(d_no_max)

    # Karşılaştırma
    status = []

    # Min atılınca durum değişti mi?
    if not pd.isna(p_no_min):
        if is_sig and p_no_min >= threshold:
            status.append("Sensitive (Min)")
        elif not is_sig and p_no_min < threshold:
            status.append("Gain Sig (Min)")

    # Max atılınca durum değişti mi?
    if not pd.isna(p_no_max):
        if is_sig and p_no_max >= threshold:
            status.append("Sensitive (Max)")
        elif not is_sig and p_no_max < threshold:
            status.append("Gain Sig (Max)")

    if not status:
        return "Robust"
    return ", ".join(list(set(status)))


def calculate_effect_sizes(data1, data2):
    """Cohen's d ve Hedges' g hesaplar."""
    try:
        n1, n2 = len(data1), len(data2)
        if n1 < 2 or n2 < 2: return "-", "-"

        m1, m2 = np.mean(data1), np.mean(data2)
        v1, v2 = np.var(data1, ddof=1), np.var(data2, ddof=1)

        # Pooled Standard Deviation (Havuzlanmış Standart Sapma)
        pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))

        if pooled_sd == 0: return "-", "-"

        # Cohen's d
        d = (m1 - m2) / pooled_sd

        # Hedges' g (Küçük örneklem düzeltmesi)
        # J düzeltme faktörü
        df = n1 + n2 - 2
        j = 1 - (3 / (4 * df - 1))
        g = d * j

        return f"{d:.2f}", f"{g:.2f}"
    except:
        return "-", "-"


def calculate_kruskal_effect_sizes(data_groups):
    """
    Kruskal-Wallis için Eta-Squared ve Epsilon-Squared hesaplar.
    (3+ grup karşılaştırmaları için)
    """
    try:
        # Boş değerleri (NaN) temizle
        clean_groups = [g[~np.isnan(g)] for g in data_groups]

        # n (toplam örneklem) ve k (grup sayısı)
        n = sum(len(g) for g in clean_groups)
        k = len(clean_groups)

        if k < 3:
            return "-", "-"  # 2 grup için d/g kullanılır

        # H istatistiğini hesapla
        H, _ = kruskal(*clean_groups)

        # 1. Eta Squared (Klasik)
        # Formül: (H - k + 1) / (n - k)
        eta_sq = (H - k + 1) / (n - k)

        # 2. Epsilon Squared (Daha az yanlı/biased olduğu kabul edilir)
        # Formül: H / ((n^2 - 1) / (n + 1))
        epsilon_sq = H * (n + 1) / (n ** 2 - 1)

        # Değerleri 0-1 arasına sabitle (Matematiksel sapmalar için)
        eta_sq = max(0, min(1, eta_sq))
        epsilon_sq = max(0, min(1, epsilon_sq))

        return f"{eta_sq:.3f}", f"{epsilon_sq:.3f}"
    except:
        return "-", "-"


# ===================== UI =====================
st.set_page_config(page_title="Biomarker Analysis Dashboard", layout="wide")
st.title("🔬 Biomarker Analysis Dashboard (.csv, .sav)")

uploaded_file = st.file_uploader("Upload CSV, Excel or SPSS", type=["csv", "sav", "xls", "xlsx"])

if uploaded_file:
    # ---- Load data ----
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".sav"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        df, meta = pyreadstat.read_sav(tmp_file_path)
    # --- YENİ EKLENEN KISIM (Excel Desteği) ---
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)

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

    test_choice = st.sidebar.radio("Default omnibus test (for continuous)",
                                   ["Auto", "Mann-Whitney U", "Kruskal-Wallis"])
    show_nonsignificant = st.sidebar.checkbox("Show Non-Significant p-values (p > 0.05)", value=False)

    # ---- Export settings (resolution & DPI) ----
    with st.sidebar.expander("Export settings (resolution)"):
        export_width_px = st.number_input("Width (pixels)", min_value=200, value=1600, step=50)
        export_height_px = st.number_input("Height (pixels)", min_value=200, value=1200, step=50)
        export_dpi = st.number_input("DPI", min_value=72, value=300, step=1,
                                     help="For PNG/JPG. PDF ignores DPI (vector).")

    # ---- Binary ayarları ----
    with st.sidebar.expander("Binary settings (0/1, 1/2)"):
        present_code_ui = st.selectbox("Code that means 'present' (var=1)", options=[1, 2],
                                       index=1 if 2 in df.columns else 0)
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
        supers = ['ᵃ', 'ᵇ', 'ᶜ', 'ᵈ', 'ᵉ', 'ᶠ', 'ᵍ', 'ʰ', 'ᶦ', 'ʲ', 'ᵏ', 'ˡ', 'ᵐ', 'ⁿ', 'ᵒ', 'ᵖ', 'ʳ', 'ˢ', 'ᵗ', 'ᵘ',
                  'ᵛ', 'ʷ', 'ˣ', 'ʸ', 'ᶻ']
        method_notes = {}  # method_label -> superscript


        def note_for(method_label: str):
            if not method_label:
                return ""
            if method_label not in method_notes:
                method_notes[method_label] = supers[len(method_notes) % len(supers)]
            return method_notes[method_label]


        # ---------- Tabloyu oluştur ----------
        rows = []
        header_idx = []
        all_p_values = []  # Düzeltme için p değerlerini toplayacağız

        # SÜTUNLARI GÜNCELLE: q-value ve h eklendi
        columns = ["Parameter"] + group_headers + ["p-value", "q-value (FDR)", "h", "95% CI (Diff)", "Cohen's d",
                                                   "Hedges' g", "Eta-sq", "Epsilon-sq", "LOO Analysis"]

        # Önce p değerlerini toplamak için ilk döngü
        temp_data = []
        for it in items:
            if it["type"] == "header":
                temp_data.append({"type": "header", "label": it["label"]})
                continue

            col_name = it["name"]
            disp = it["label"]
            if col_name not in df.columns:
                temp_data.append({"type": "missing", "disp": disp})
                continue

            dsub = df[[group_var, col_name]].dropna()
            if not is_binary_like(dsub[col_name]):
                dsub[col_name] = pd.to_numeric(dsub[col_name].astype(str).str.replace(',', '.'), errors='coerce')
                dsub = dsub.dropna()

            grp_data = [dsub[dsub[group_var] == g][col_name].values for g in group_labels]
            is_bin = is_binary_like(dsub[col_name])

            # Test hesaplama
            p_val = np.nan
            chosen_test = st.session_state["var_test"].get(col_name, "Auto")
            method_label = ""

            try:
                if is_bin:
                    p_val, method_label, _, _ = chi2_rx2(dsub, group_var, col_name, present_code=int(present_code_ui),
                                                         monte_carlo_B=int(mc_B) if use_mc else None)
                else:
                    if chosen_test == "Auto":
                        test_to_use = "Mann-Whitney U" if len(group_labels) == 2 else "Kruskal-Wallis"
                    else:
                        test_to_use = chosen_test

                    if test_to_use == "Mann-Whitney U":
                        _, p_val = mannwhitneyu(grp_data[0], grp_data[1], alternative="two-sided")
                        method_label = "Mann–Whitney U"
                    else:
                        _, p_val = kruskal(*grp_data)
                        method_label = "Kruskal–Wallis"
            except:
                p_val = np.nan

            temp_data.append({
                "type": "var", "col_name": col_name, "disp": disp,
                "p_val": p_val, "method_label": method_label, "grp_data": grp_data,
                "is_bin": is_bin, "dsub": dsub
            })
            if not np.isnan(p_val):
                all_p_values.append(p_val)

        # Benjamini-Hochberg Düzeltmesi (FDR)
        q_values_map = {}
        if all_p_values:
            rejected, q_vals, _, _ = multipletests(all_p_values, alpha=0.05, method='fdr_bh')
            for p, q, rej in zip(all_p_values, q_vals, rejected):
                q_values_map[p] = (q, rej)

        # Tablo satırlarını oluşturma
        for entry in temp_data:
            if entry["type"] == "header":
                rows.append([entry["label"]] + [""] * (len(columns) - 1))
                header_idx.append(len(rows) - 1)
                continue
            if entry["type"] == "missing":
                rows.append([f"[Missing: {entry['disp']}]"] + [""] * (len(columns) - 1))
                continue

            # Özetler
            if entry["is_bin"]:
                summaries = [binary_summary_n_pct(entry["dsub"].loc[entry["dsub"][group_var] == g, entry["col_name"]],
                                                  int(present_code_ui)) for g in group_labels]
            else:
                fmt = st.session_state["var_fmt"].get(entry["col_name"], summary_format)
                summaries = [group_summary_str(arr, fmt) for arr in entry["grp_data"]]

            # p, q ve h değerleri
            p_disp, q_disp, h_disp = "", "", ""
            if not np.isnan(entry["p_val"]):
                if show_nonsignificant or entry["p_val"] <= 0.05:
                    sup = note_for(entry["method_label"])
                    p_disp = format_p(entry["p_val"]) + sup

                    if entry["p_val"] in q_values_map:
                        q_val, is_rej = q_values_map[entry["p_val"]]
                        q_disp = format_p(q_val)
                        h_disp = "*" if is_rej else "ns"

            # Etki büyüklükleri ve LOO (Sizin mevcut fonksiyonlarınızla)
            ci, d, g, eta, eps, loo = "-", "-", "-", "-", "-", "-"
            if not entry["is_bin"]:
                if len(group_labels) == 2:
                    ci = get_mean_diff_ci(entry["grp_data"][0], entry["grp_data"][1])
                    d, g = calculate_effect_sizes(entry["grp_data"][0], entry["grp_data"][1])
                else:
                    eta, eps = calculate_kruskal_effect_sizes(entry["grp_data"])
                loo = run_loo_sensitivity(entry["dsub"], group_var, entry["col_name"], entry["p_val"],
                                          entry["method_label"])

            rows.append([entry["disp"]] + summaries + [p_disp, q_disp, h_disp, ci, d, g, eta, eps, loo])

        summary_df = pd.DataFrame(rows, columns=columns)


            # ---------- STYLING ----------
        def highlight_sections(row):
            if row.name in header_idx:
                return ["font-weight: bold; border-top: 1px solid black; background-color: #f7f7f7;"] * len(row)
            return [""] * len(row)


        def highlight_sensitive(val):
            color = 'red' if 'Sensitive' in str(val) else 'black'
            return f'color: {color}'


        # Hizalama: Tüm istatistik sütunlarını sağa yasla
        cols_to_align_right = group_headers + ["p-value", "q-value (FDR)", "h", "95% CI (Diff)", "Cohen's d",
                                               "Hedges' g", "Eta-sq", "Epsilon-sq", "LOO Analysis"]

        styler = (
            summary_df
            .style
            .format(na_rep="")
            .apply(highlight_sections, axis=1)
            .map(highlight_sensitive, subset=["LOO Analysis"])
            .set_properties(**{"font-family": "Arial", "font-size": "12pt"})
            .set_properties(subset=pd.IndexSlice[:, ["Parameter"]], **{"text-align": "left"})
            .set_properties(subset=pd.IndexSlice[:, cols_to_align_right], **{"text-align": "right"})
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
        subplot_titles = [s.strip() for s in
                          subplot_titles_input.split(",")] if subplot_titles_input.strip() else test_vars

        ncols = math.ceil(math.sqrt(len(test_vars))) if len(test_vars) > 0 else 1
        nrows = math.ceil(len(test_vars) / ncols) if len(test_vars) > 0 else 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.array(axes).reshape(-1)
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # ... (kodun üst kısımları aynı) ...

        for idx, test_var in enumerate(test_vars):
            # --- DÜZELTME BAŞLANGICI ---

            # 1. Önce veriyi temizle: Virgülleri noktaya çevir (Türkiye formatı sorunu için)
            # ve zorla sayısala çevir (pd.to_numeric). Sayı olmayanlar NaN olur.
            numeric_series = pd.to_numeric(
                df[test_var].astype(str).str.replace(',', '.'),
                errors='coerce'
            )

            # 2. Geçici bir DataFrame oluştur ve NaN olanları at
            df_clean = pd.DataFrame({
                group_var: df[group_var],
                test_var: numeric_series
            }).dropna()

            # 3. Eğer veri tamamen boşaldıysa (hiç sayı yoksa) bu değişkeni atla
            if df_clean.empty:
                st.warning(
                    f"Variable '{test_var}' could not be converted to numeric (check for text values). Skipping.")
                continue

            # --- DÜZELTME BİTİŞİ ---

            # Buradan sonrası sizin orijinal kodunuzla aynı mantıkta devam eder
            # Ancak df_clean artık garanti olarak sayısal veri içerir.

            df_clean = df_clean[df_clean[group_var].notna()]
            group_labels_plot = sorted(df_clean[group_var].unique())
            group_data = [df_clean[df_clean[group_var] == grp][test_var] for grp in group_labels_plot]

            ax = axes[idx]
            # ... (kalan grafik çizim kodları aynı)
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
                    ax.text(0.5, ypos + 0.07 * ypos, format_p_label(p_value), ha='center', va='bottom')
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






