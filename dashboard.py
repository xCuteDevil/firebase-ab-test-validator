import streamlit as st
import pandas as pd
import plotly.express as px
import os
from scipy.stats import ttest_ind, mannwhitneyu
import math
from PIL import Image
import base64
from io import BytesIO
from typing import Tuple, Optional
import difflib


def get_base64_image(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def load_srm_report(path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    if not os.path.exists(path):
        return None, f"Soubor '{path}' nebyl nalezen."
    
    df = pd.read_csv(path)
    required_columns = {"experiment_number", "first_seen", "p99_seen", "counts"}
    missing = required_columns - set(df.columns)
    if missing:
        return None, f"Chybějící sloupce v SRM_Report.csv: {missing}"

    df["experiment_number"] = df["experiment_number"].apply(lambda x: str(int(x)) if pd.notnull(x) else "")
    df["start_date"] = pd.to_datetime(df["first_seen"])
    df["end_date"] = pd.to_datetime(df["p99_seen"])
    df["total_users"] = df["counts"].apply(lambda counts: sum(int(x.strip()) for x in counts.split(",")) if isinstance(counts, str) else 0)
    
    return df, None

def winsorize_series(series: pd.Series, lower_pct: float, upper_pct: float):
    lower = series.quantile(lower_pct / 100)
    upper = series.quantile(1 - upper_pct / 100)
    return series.clip(lower=lower, upper=upper)

st.set_page_config(page_title="Experiment Dashboard", layout="wide")

# --- Cesty ---
REPORTS_DIR = "Reports"
SRM_REPORT_PATH = "SRM_Report.csv"

# --- Slovník názvů experimentů ---
experiment_names = {
    "40": "Intro and Tutorial",
    "42": "Hero Gem Offers",
    "44": "Stage Rewards Hero Materials (1)",
    "45": "Stage Rewards Hero Materials (2)",
    "46": "AA Test"
}

SRM_REPORT_PATH = "SRM_Report.csv"
srm_df, srm_error = load_srm_report(SRM_REPORT_PATH)

# --- Sidebar ---
with st.sidebar:
    logo_path = "logo2.png"
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
        base64_logo = get_base64_image(logo_img)
        st.markdown(
            f"""
            <div style="text-align: center; padding-bottom: 1rem;">
                <img src="data:image/png;base64,{base64_logo}" style="width: 75%; max-width: 160px;" />
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Soubor 'logo2.png' nebyl nalezen.")
    st.markdown("---")
    selected_game = st.selectbox("Vyber hru:", ["Hexapolis", "Age of Tanks", "Tanks Arena", "Spacehex", "Mecha Fortress"])
    st.markdown("---")
    selected_test = st.selectbox(
        "Statistický test pro výpočet p-value:",
        ["Welchův t-test", "Mann–Whitney U test"]
    )
    st.markdown("---")
    filter_positive_revenue = st.sidebar.checkbox("Zobrazit pouze hráče s revenue > 0", value=False)
    winsor_pct = st.slider("Winsorizovat hodnoty na percentily:", min_value=0.0, max_value=3.0, value=0.0, step=0.05,
                        help="Odstraní extrémní hodnoty mimo zadané spodní a horní percentily (např. 5% a 95%). 0 = žádné ořezání.")

    selected_mode = st.radio("Typ metriky:", ["Revenue", "Revenue per User"])
    selected_metric = st.selectbox("Zobrazit metriku:", ["Total Revenue (IAP + Ad)", "Ad Revenue", "IAP Revenue"])
    st.markdown("---")

    if not os.path.exists(REPORTS_DIR):
        st.error("Složka 'Reports/' neexistuje.")
        st.stop()

    experiment_ids = sorted([f for f in os.listdir(REPORTS_DIR) if os.path.isdir(os.path.join(REPORTS_DIR, f))])
    selected_experiment = st.selectbox("Vyber A/B test ID:", experiment_ids)

# --- Metadata z SRM ---
exp_row = srm_df[srm_df["experiment_number"] == selected_experiment]
experiment_title = experiment_names.get(selected_experiment, "Neznámý název")

# --- Titulek a metriky ---
st.title(f"#{selected_experiment} — {experiment_title}")
if not exp_row.empty:
    start = exp_row.iloc[0]["start_date"].date()
    end = exp_row.iloc[0]["end_date"].date()
    total_users = int(exp_row.iloc[0]["total_users"])
    duration = (end - start).days + 1
    st.info(f"Experiment probíhal od **{start}** do **{end}** po dobu **{duration} dní**.")
else:
    st.warning("Nebyla nalezena metadata o trvání experimentu.")
    total_users = None
    duration = None

col1, col2, col3 = st.columns(3)
col1.metric("👥 Počet uživatelů", f"{total_users:,}" if total_users else "—")
variant_count = len(exp_row.iloc[0]["counts"].split(",")) if not exp_row.empty and "counts" in exp_row.columns else "—"
col2.metric("🧫 Varianty", variant_count)
col3.metric("🕒 Doba trvání", f"{duration} dní" if duration else "—")

st.markdown("---")

# --- Načtení dat ---
data = []
cumulative_by_day = []
experiment_path = os.path.join(REPORTS_DIR, selected_experiment)

if os.path.exists(experiment_path):
    files = sorted([f for f in os.listdir(experiment_path) if f.endswith(".csv")])
    for d_index, f in enumerate(files):
        df = pd.read_csv(os.path.join(experiment_path, f)).fillna(0)
        df["den"] = d_index
        df["datum"] = pd.to_datetime(exp_row.iloc[0]["start_date"]) + pd.to_timedelta(d_index, unit="d")
        df["total_revenue"] = df.get("total_ad_revenue", 0) + df.get("total_iap_revenue", 0)

        # Urči metriku bez winsorizace – ta bude aplikována globálně později
        if selected_metric == "Total Revenue (IAP + Ad)":
            df["selected_value"] = df["total_revenue"]
        elif selected_metric == "Ad Revenue":
            df["selected_value"] = df.get("total_ad_revenue", 0)
        elif selected_metric == "IAP Revenue":
            df["selected_value"] = df.get("total_iap_revenue", 0)
        else:
            df["selected_value"] = df["total_revenue"]

        cumulative_by_day.append(df[["experiment_group", "user_pseudo_id", "selected_value", "den", "datum"]])
        data.append(df)
else:
    st.warning("Data pro tento experiment nejsou dostupná.")


# --- Cumulative graf ---
if cumulative_by_day:
    full_daily = pd.concat(cumulative_by_day, ignore_index=True)
    if winsor_pct > 0:
        full_daily["selected_value"] = winsorize_series(full_daily["selected_value"], winsor_pct, winsor_pct)

    full_daily.sort_values(by=["experiment_group", "den"], inplace=True)

    user_counts = full_daily.groupby(["den", "experiment_group"])['user_pseudo_id'].nunique().reset_index(name="new_users")
    user_counts["cumulative_users"] = user_counts.groupby("experiment_group")["new_users"].cumsum()

    daily_sum = full_daily.groupby(["experiment_group", "den"]).agg(revenue_sum=('selected_value', 'sum')).reset_index()
    daily_sum["cumulative_revenue"] = daily_sum.groupby("experiment_group")["revenue_sum"].cumsum()
    daily_sum = pd.merge(daily_sum, user_counts, on=["den", "experiment_group"], how="left")
    daily_sum["value"] = daily_sum["cumulative_revenue"] / daily_sum["cumulative_users"] if selected_mode == "Revenue per User" else daily_sum["cumulative_revenue"]
    # Přiřaď každému "den" jeho odpovídající datum
    den_datum_map = full_daily.groupby("den")["datum"].first().reset_index()
    daily_sum = pd.merge(daily_sum, den_datum_map, on="den", how="left")

    # Vytvoření popisku pro osu X ve formátu "D1 – 2025-01-23"
    daily_sum["x_label"] = daily_sum["den"].apply(lambda d: f"D{d}") + " – " + daily_sum["datum"].dt.strftime("%Y-%m-%d")

    y_label = f"Cumulative {selected_metric}" + (" per User" if selected_mode == "Revenue per User" else "")
    color_map = {0: "#9E9E9E", 1: "#4285F4", 2: "#FF8F00", 3: "#EC407A", 4: "#00B8D4"}
    
    # Vytvoření custom popisků pro osu X
    interval = max(1, len(daily_sum["den"].unique()) // 20)  # max ~15 popisků
    tick_df = daily_sum[daily_sum["den"] % interval == 0]
    tickvals = tick_df["den"]
    ticktext = tick_df["den"].apply(lambda d: f"D{d}") + "<br>" + tick_df["datum"].dt.strftime("%b %d")

    fig = px.line(
        daily_sum,
        x="den",
        y="value",
        color="experiment_group",
        labels={"den": "Den", "value": y_label, "experiment_group": "Varianta"},
        color_discrete_map=color_map,
        height=450
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=45
    )

    st.subheader(f"📈 {y_label} by Variant")
    st.plotly_chart(fig, use_container_width=True)

# --- Souhrnná tabulka + Boxplot + Histogram ---
if data:
    # Urči správný sloupec podle vybrané metriky
    metric_column = {
        "Total Revenue (IAP + Ad)": "total_revenue",
        "Ad Revenue": "total_ad_revenue",
        "IAP Revenue": "total_iap_revenue"
    }.get(selected_metric, "total_revenue")

    full_data = pd.concat(data, ignore_index=True).fillna(0)
    if filter_positive_revenue:
        full_data = full_data[full_data[metric_column] > 0]
    full_data["total_revenue"] = full_data.get("total_ad_revenue", 0) + full_data.get("total_iap_revenue", 0)

    # Aplikace winsorizace na hlavní metriku (pro grafy)
    if winsor_pct > 0:
        full_data[metric_column] = winsorize_series(full_data[metric_column], winsor_pct, winsor_pct)



    st.subheader("📋 Souhrnná tabulka variant")
    summary = full_data.groupby("experiment_group").agg(
        Users=('user_pseudo_id', 'nunique'),
        Total_Revenue=('total_revenue', 'sum'),
        Ad_Revenue=('total_ad_revenue', 'sum'),
        IAP_Revenue=('total_iap_revenue', 'sum'),
        Revenue_per_User=(metric_column, 'mean'),
        Std_Dev=(metric_column, 'std')
    )

    # Extra statistiky – zvlášť
    medians = full_data.groupby("experiment_group")[metric_column].median().rename("Median")
    q1 = full_data.groupby("experiment_group")[metric_column].quantile(0.25).rename("Q1")
    q3 = full_data.groupby("experiment_group")[metric_column].quantile(0.75).rename("Q3")

    # Spoj dohromady
    summary = summary.join([medians, q1, q3])


    baseline_group = sorted(summary.index)[0]
    baseline_mean = summary.loc[baseline_group, "Revenue_per_User"]
    summary["Rozdíl od baseline"] = summary["Revenue_per_User"].apply(lambda x: "0.00%" if x == baseline_mean else f"{((x - baseline_mean) / baseline_mean) * 100:+.2f}%")

    baseline_raw = full_data[full_data["experiment_group"] == baseline_group][metric_column]

    p_values = []
    for group in summary.index:
        if group == baseline_group:
            p_values.append("—")
        else:
            test_raw = full_data[full_data["experiment_group"] == group][metric_column]

            # Aplikuj winsorizaci, pokud je nastavena
            if winsor_pct > 0:
                baseline_data = winsorize_series(baseline_raw, winsor_pct, winsor_pct)
                test_data = winsorize_series(test_raw, winsor_pct, winsor_pct)
            else:
                baseline_data = baseline_raw
                test_data = test_raw

            # Vyhodnoť statistický test
            if selected_test == "Welchův t-test":
                _, p = ttest_ind(baseline_data, test_data, equal_var=False)
            elif selected_test == "Mann–Whitney U test":
                _, p = mannwhitneyu(baseline_data, test_data, alternative="two-sided")
            else:
                p = float("nan")

            p_values.append(f"{p:.3f}" if p >= 0.0001 else "<0.0001")

    summary["P-value"] = p_values

    summary = summary.reset_index().rename(columns={
        "experiment_group": "Varianta",
        "Users": "Počet uživatelů",
        "Total_Revenue": "Total revenue",
        "Ad_Revenue": "Ad revenue",
        "IAP_Revenue": "IAP revenue",
        "Revenue_per_User": "Revenue / user",
        "Std_Dev": "Standardní odchylka"
    })
    summary["Revenue / user"] = summary["Revenue / user"].apply(lambda x: f"${x:.2f}")
    summary["Total revenue"] = summary["Total revenue"].apply(lambda x: f"${x:,.2f}")
    summary["Ad revenue"] = summary["Ad revenue"].apply(lambda x: f"${x:,.2f}")
    summary["IAP revenue"] = summary["IAP revenue"].apply(lambda x: f"${x:,.2f}")
    summary["Standardní odchylka"] = summary["Standardní odchylka"].apply(lambda x: f"${x:.2f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # --- Boxplot ---
    st.subheader("🎯 Rozdělení výnosů mezi variantami (Boxplot)")

    # Definice sloupce podle metriky
    boxplot_col = {
        "Total Revenue (IAP + Ad)": "total_revenue",
        "Ad Revenue": "total_ad_revenue",
        "IAP Revenue": "total_iap_revenue"
    }.get(selected_metric, "total_revenue")

    with st.expander("⚙️ Nastavení boxplotu"):
        col1, col2, col3 = st.columns(3)
        use_log_y = col1.checkbox("Použít logaritmickou osu Y", value=False)
        min_y = col2.number_input("Min Y", value=0.0, format="%.4f", step=0.01, key="box_min_y")
        max_y = col3.number_input("Max Y", value=round(float(full_data[boxplot_col].max()), 4), format="%.4f", step=0.01, key="box_max_y")

    # Pojistka na 0 u log
    if use_log_y and min_y <= 0:
        st.warning("⚠️ Pro logaritmickou osu Y musí být Min Y > 0. Nastavuji automaticky na 0.01.")
        min_y = 0.01

    # Připrav data a filtruj podle rozsahu
    box_df = full_data[["experiment_group", boxplot_col]].copy()
    box_df = box_df.rename(columns={"experiment_group": "Varianta", boxplot_col: "Revenue per user"})
    box_df = box_df[(box_df["Revenue per user"] >= min_y) & (box_df["Revenue per user"] <= max_y)]

    fig_box = px.box(
        box_df,
        x="Varianta",
        y="Revenue per user",
        points="all",
        color="Varianta",
        color_discrete_map=color_map,
        height=500
    )

    # Nastavení osy
    if use_log_y:
        fig_box.update_yaxes(type="log")
        if max_y > min_y:
            fig_box.update_yaxes(range=[math.log10(min_y), math.log10(max_y)])
    else:
        if max_y > min_y:
            fig_box.update_yaxes(range=[min_y, max_y])
        elif min_y:
            fig_box.update_yaxes(range=[min_y, None])
        elif max_y:
            fig_box.update_yaxes(range=[None, max_y])

    st.plotly_chart(fig_box, use_container_width=True)


    # --- Histogram ---
    st.subheader("📊 Histogram výnosů podle variant")

    # Výběr správného sloupce
    metric_column = {
        "Total Revenue (IAP + Ad)": "total_revenue",
        "Ad Revenue": "total_ad_revenue",
        "IAP Revenue": "total_iap_revenue"
    }.get(selected_metric, "total_revenue")

    hist_df = full_data[["experiment_group", "user_pseudo_id", metric_column]].copy()
    hist_df = hist_df.rename(columns={metric_column: "value"})

    with st.expander("⚙️ Nastavení histogramu"):
        col1, col2, col3, col4 = st.columns(4)
        use_log_y = col1.checkbox("Použít logaritmickou osu Y", value=False, key="hist_log_y")

        default_min = float(hist_df["value"].min())
        default_max = float(hist_df["value"].max())

        min_value = col2.number_input("Minimální hodnota", value=0.0, step=1.0, format="%.2f", key="hist_min_value")
        max_value = col3.number_input("Maximální hodnota", value=round(default_max, 2), step=1.0, format="%.2f", key="hist_max_value")

        bin_count = col4.number_input("Počet binů", value=100, min_value=5, max_value=5000, step=10, key="hist_bins")

    # Filtrování podle hodnot
    hist_df = hist_df[(hist_df["value"] >= min_value) & (hist_df["value"] <= max_value)]

    if hist_df.empty:
        st.warning("⚠️ Po aplikaci filtrů nezůstala žádná data pro histogram.")
    else:
        fig_hist = px.histogram(
            hist_df,
            x="value",
            color="experiment_group",
            nbins=bin_count,
            barmode="overlay",
            opacity=0.6,
            log_y=use_log_y,
            labels={"value": f"{selected_metric} per User", "experiment_group": "Varianta"},
            color_discrete_map=color_map,
            height=400
        )
        fig_hist.update_layout(xaxis_title=f"{selected_metric} per User")
        st.plotly_chart(fig_hist, use_container_width=True)

