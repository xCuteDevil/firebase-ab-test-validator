import streamlit as st
import pandas as pd
import plotly.express as px
import os
from scipy.stats import ttest_ind

# --- Konfigurace stránky ---
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

# --- Načti SRM report ---
if os.path.exists(SRM_REPORT_PATH):
    srm_df = pd.read_csv(SRM_REPORT_PATH)
    required_columns = {"experiment_number", "first_seen", "p99_seen", "counts"}
    if not required_columns.issubset(srm_df.columns):
        st.error(f"Chybějící sloupce v SRM_Report.csv: {required_columns - set(srm_df.columns)}")
        st.stop()
    srm_df["experiment_number"] = srm_df["experiment_number"].apply(lambda x: str(int(x)) if pd.notnull(x) else "")
    srm_df["start_date"] = pd.to_datetime(srm_df["first_seen"])
    srm_df["end_date"] = pd.to_datetime(srm_df["p99_seen"])
    srm_df["total_users"] = srm_df["counts"].apply(lambda counts: sum(int(x.strip()) for x in counts.split(",")) if isinstance(counts, str) else 0)
else:
    st.error("Soubor 'SRM_Report.csv' nebyl nalezen.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title("🫪 A/B Testy")
    selected_view = st.radio("Zobrazit", ["Přehled", "Výnosy", "Retence"])
    st.markdown("---")
    selected_metric = st.selectbox("Zobrazit metriku:", ["Total Revenue", "Ad Revenue", "IAP Revenue"])
    st.markdown("---")
    if not os.path.exists(REPORTS_DIR):
        st.error("Složka 'Reports/' neexistuje.")
        st.stop()
    experiment_ids = sorted([f for f in os.listdir(REPORTS_DIR) if os.path.isdir(os.path.join(REPORTS_DIR, f))])
    selected_experiment = st.selectbox("Vyber experiment ID:", experiment_ids)


# --- Metadata z SRM ---
exp_row = srm_df[srm_df["experiment_number"] == selected_experiment]
experiment_title = experiment_names.get(selected_experiment, "Neznámý název")

# --- Hlavní nadpis ---
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

# --- Přehled metrik ---
col1, col2, col3 = st.columns(3)
col1.metric("👥 Počet uživatelů", f"{total_users:,}" if total_users else "—")
variant_count = len(exp_row.iloc[0]["counts"].split(",")) if not exp_row.empty and "counts" in exp_row.columns else "—"
col2.metric("🧫 Varianty", variant_count)
col3.metric("🕒 Doba trvání", f"{duration} dní" if duration else "—")

st.markdown("---")

# --- Načtení dat a agregace ---
data = []
cumulative_by_day = []
experiment_path = os.path.join(REPORTS_DIR, selected_experiment)

if os.path.exists(experiment_path):
    files = sorted([f for f in os.listdir(experiment_path) if f.endswith(".csv")])
    for d_index, f in enumerate(files):
        df = pd.read_csv(os.path.join(experiment_path, f)).fillna(0)
        df["den"] = d_index
        if "total_revenue" not in df.columns:
            df["total_revenue"] = df.get("total_ad_revenue", 0) + df.get("total_iap_revenue", 0)
        cumulative_by_day.append(df[["experiment_group", "total_revenue", "den"]])
        data.append(df)
else:
    st.warning("Data pro tento experiment nejsou dostupná.")

# --- Graf kumulativního total revenue ---
if cumulative_by_day:
    full_daily = pd.concat(cumulative_by_day, ignore_index=True)
    full_daily.sort_values(by=["experiment_group", "den"], inplace=True)
    full_daily["cumulative_total_revenue"] = full_daily.groupby("experiment_group")["total_revenue"].cumsum()

    fig = px.line(
        full_daily,
        x="den",
        y="cumulative_total_revenue",
        color="experiment_group",
        labels={
            "den": "Den",
            "cumulative_total_revenue": "Cumulative Total Revenue",
            "experiment_group": "Varianta"
        },
        height=450
    )
    st.subheader("📈 Cumulative Total Revenue by Variant")
    st.plotly_chart(fig, use_container_width=True)

# --- Souhrnná tabulka variant ---
if data:
    full_data = pd.concat(data, ignore_index=True).fillna(0)
    if 'total_revenue' not in full_data.columns:
        full_data['total_revenue'] = full_data.get("total_ad_revenue", 0) + full_data.get("total_iap_revenue", 0)

    st.subheader("📋 Souhrnná tabulka variant")
    summary = full_data.groupby("experiment_group").agg(
        Users=('user_pseudo_id', 'nunique'),
        Total_Revenue=('total_revenue', 'sum'),
        Revenue_per_User=('total_revenue', 'mean'),
        Std_Dev=('total_revenue', 'std')
    )
    baseline_group = sorted(summary.index)[0]
    baseline_mean = summary.loc[baseline_group, "Revenue_per_User"]
    summary["Rozdíl od baseline"] = summary["Revenue_per_User"].apply(
        lambda x: "0.00%" if x == baseline_mean else f"{((x - baseline_mean) / baseline_mean) * 100:+.2f}%")

    p_values = []
    baseline_data = full_data[full_data["experiment_group"] == baseline_group]["total_revenue"]
    for group in summary.index:
        if group == baseline_group:
            p_values.append("—")
        else:
            test_data = full_data[full_data["experiment_group"] == group]["total_revenue"]
            _, p = ttest_ind(baseline_data, test_data, equal_var=False)
            p_values.append(f"{p:.2f}" if p >= 0.0001 else "<0.0001")
    summary["P-value"] = p_values

    summary = summary.reset_index()
    summary.rename(columns={
        "experiment_group": "Varianta",
        "Users": "Počet uživatelů",
        "Total_Revenue": "Total revenue",
        "Revenue_per_User": "Revenue / user",
        "Std_Dev": "Standardní odchylka"
    }, inplace=True)

    summary["Revenue / user"] = summary["Revenue / user"].apply(lambda x: f"${x:.2f}")
    summary["Total revenue"] = summary["Total revenue"].apply(lambda x: f"${x:,.2f}")
    summary["Standardní odchylka"] = summary["Standardní odchylka"].apply(lambda x: f"${x:.2f}")

    st.dataframe(summary, use_container_width=True)
