"""
app.py  ─  SalesDynamics Forecast  ·  Streamlit Dashboard
Retail Store Sales Prediction using Stacking Ensemble
Run: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SalesDynamics Forecast",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "Linear Regression": "#4C72B0",
    "Random Forest":     "#55A868",
    "SVR":               "#C44E52",
    "Stacking Ensemble": "#8172B2",
}
BG = "#0E1117"

st.markdown("""
<style>
  .metric-card {background:#1a1f2b;border-radius:12px;padding:16px 20px;
               margin:4px 0;border-left:4px solid #8172B2;}
  .metric-label{font-size:.78rem;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;}
  .metric-value{font-size:1.6rem;font-weight:700;color:#f9fafb;}
  .winner-badge{display:inline-block;background:#8172B2;color:#fff;
                border-radius:20px;padding:2px 10px;font-size:.72rem;margin-left:6px;}
  .section-header{font-size:1.1rem;font-weight:600;color:#c4b5fd;
                  border-bottom:1px solid #2d3748;padding-bottom:6px;margin:16px 0 10px 0;}
  h1,h2,h3{color:#f9fafb!important;}
  .stTabs [data-baseweb="tab-list"]{gap:6px;}
  .stTabs [data-baseweb="tab"]{border-radius:8px 8px 0 0;padding:8px 18px;}
  .info-box{background:#1a1f2b;border-radius:10px;padding:14px 18px;
            border-left:4px solid #4C72B0;margin:8px 0;color:#d1d5db;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def styled_metric(label, value):
    return f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div></div>"""

def info_box(text):
    return f'<div class="info-box">{text}</div>'

def dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#1a1f2b")
    ax.tick_params(colors="#9ca3af")
    ax.spines[:].set_color("#2d3748")
    return fig, ax

def bar_chart(ax, data, col, title, ylabel):
    bars = ax.bar(data["Model"], data[col],
                  color=[COLORS[m] for m in data["Model"]],
                  edgecolor="none", width=0.55)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#f9fafb", pad=8)
    ax.set_ylabel(ylabel, color="#9ca3af")
    ax.tick_params(axis="x", rotation=15, colors="#9ca3af")
    ax.tick_params(axis="y", colors="#9ca3af")
    ax.set_facecolor("#1a1f2b"); ax.spines[:].set_color("#2d3748")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.01*max(bar.get_height(),1),
                f"{bar.get_height():.2f}",
                ha="center",va="bottom",fontsize=8,color="#f9fafb")

def prepare_data(df):
    y = df["Sales"]
    X = df.drop("Sales", axis=1)
    if "HasPromoInterval" in X.columns:
        X = X.drop("HasPromoInterval", axis=1)
    X = X.astype(float)
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING  (faster settings to reduce training time)
# ══════════════════════════════════════════════════════════════════════════════

def build_lr():
    return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

def build_rf():
    # n_estimators=100 (was 200), max_depth=8 — ~2x faster, minimal accuracy drop
    return Pipeline([("model", RandomForestRegressor(
        n_estimators=100, max_depth=8,
        min_samples_split=10, random_state=42, n_jobs=-1))])

def build_svr():
    return Pipeline([("scaler", StandardScaler()),
                     ("model", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1))])

def build_stack():
    return StackingRegressor(
        estimators=[
            ("lr",  build_lr()),
            ("rf",  build_rf()),
            ("svr", build_svr()),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=3,           # was 5 — biggest speed win
        passthrough=False,  # False = faster meta-training
        n_jobs=-1
    )


# ══════════════════════════════════════════════════════════════════════════════
# CACHED TRAINING  (runs once, cached until data changes)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = "models"

@st.cache_resource(show_spinner=False)
def train_models_cached(data_key: str, csv_bytes: bytes):
    df    = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    # Use a sample of max 20,000 rows for speed — plenty for good model performance
    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    X, y  = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": build_lr(),
        "Random Forest":     build_rf(),
        "SVR":               build_svr(),
        "Stacking Ensemble": build_stack(),
    }

    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = np.maximum(model.predict(X_test), 0)

    # Save models to disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, name.lower().replace(" ","_")+".pkl"))

    comparison = pd.DataFrame([
        {"Model": n,
         "MAE":  round(mean_absolute_error(y_test, preds[n]), 2),
         "RMSE": round(np.sqrt(mean_squared_error(y_test, preds[n])), 2),
         "R2":   round(r2_score(y_test, preds[n]), 4)}
        for n in models
    ])
    return models, preds, comparison, X_train, X_test, y_train, y_test, X.columns.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛒 <span style='color:#8172B2'>SalesDynamics</span>\nForecast", unsafe_allow_html=True)
    st.markdown("---")

    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "processed", "cleaned_sales_data.csv"
    )

    data_source = st.radio("Data Source", ["Auto-load project dataset", "Upload CSV manually"])
    uploaded = None
    if data_source == "Upload CSV manually":
        uploaded = st.file_uploader("Upload cleaned_sales_data.csv", type=["csv"])

    st.markdown("---")
    st.markdown("**🏗️ Ensemble Architecture**")
    st.markdown("""
| Layer | Model |
|---|---|
| Base 1 | Linear Regression |
| Base 2 | Random Forest |
| Base 3 | SVR |
| Meta | Ridge Regression |
""")
    st.markdown("---")
    st.markdown("**🎯 Prediction Goal**")
    st.markdown(
        "Predict **daily sales** for a retail store "
        "based on store type, promotions, holidays, "
        "competition distance and more."
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
raw_df = None

if data_source == "Auto-load project dataset":
    if os.path.exists(default_path):
        raw_df = pd.read_csv(default_path)
        st.sidebar.success(f"✅ Loaded: cleaned_sales_data.csv\n({len(raw_df):,} rows)")
    else:
        st.warning(
            "⚠️ Could not auto-find `data/processed/cleaned_sales_data.csv`.\n\n"
            "Please switch to **Upload CSV manually** in the sidebar."
        )
        st.stop()
else:
    if uploaded is None:
        st.info("👆 Upload your **cleaned_sales_data.csv** file to begin.")
        st.stop()
    raw_df = pd.read_csv(uploaded)

if "Sales" not in raw_df.columns:
    st.error("❌ Dataset must contain a **'Sales'** column as the target variable.")
    st.stop()

# Training spinner shown only on first load
data_key  = str(raw_df.shape) + str(round(raw_df["Sales"].sum(), 2))
csv_bytes = raw_df.to_csv(index=False).encode()

with st.spinner("⏳ Training models for the first time… (this only happens once)"):
    models, preds, comparison, X_train, X_test, y_train, y_test, feat_cols = \
        train_models_cached(data_key, csv_bytes)

best_model_name = comparison.loc[comparison["R2"].idxmax(), "Model"]
best_r2         = comparison.loc[comparison["R2"].idxmax(), "R2"]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛒 <span style='color:#8172B2'>SalesDynamics</span> Forecast", unsafe_allow_html=True)
st.markdown(
    "**Retail Store Sales Prediction** using a Stacking Ensemble "
    "— combining Linear Regression, Random Forest and SVR "
    "with a Ridge meta-model to forecast daily store sales."
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview",
    "📊 Data Exploration",
    "🤖 Model Comparison",
    "🔮 Predict Store Sales"
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(styled_metric("Total Records",     f"{len(raw_df):,}"),              unsafe_allow_html=True)
    c2.markdown(styled_metric("Features",          str(len(feat_cols))),              unsafe_allow_html=True)
    c3.markdown(styled_metric("Avg Daily Sales",   f"€{raw_df['Sales'].mean():,.0f}"),unsafe_allow_html=True)
    c4.markdown(styled_metric("Max Daily Sales",   f"€{raw_df['Sales'].max():,.0f}"), unsafe_allow_html=True)
    c5.markdown(styled_metric("Best Model R²",     f"{best_r2:.4f}"),                 unsafe_allow_html=True)

    st.markdown("---")

    # What are we predicting?
    st.markdown('<div class="section-header">🎯 What are we predicting?</div>', unsafe_allow_html=True)
    st.markdown(info_box(
        "We are predicting <b>daily sales (in €)</b> for individual Rossmann retail stores across Germany. "
        "Sales depend on factors like whether a store is running a <b>promotion</b>, nearby <b>competition distance</b>, "
        "<b>school/public holidays</b>, <b>store type</b>, and <b>day of week</b>. "
        "Accurate forecasts help the business manage <b>inventory, staffing, and promotions</b> more efficiently."
    ), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">📋 Key Input Features</div>', unsafe_allow_html=True)
        feature_desc = {
            "Store":               "Store ID (unique identifier)",
            "DayOfWeek":           "Day of week (1=Mon … 7=Sun)",
            "Promo":               "Whether store is running a promotion (0/1)",
            "StateHoliday":        "Public holiday flag",
            "SchoolHoliday":       "School holiday flag",
            "StoreType":           "Store category (a/b/c/d)",
            "Assortment":          "Product assortment level (a/b/c)",
            "CompetitionDistance": "Distance to nearest competitor (metres)",
            "Promo2":              "Ongoing consecutive promotion (0/1)",
        }
        desc_df = pd.DataFrame(
            [{"Feature": k, "Description": v} for k, v in feature_desc.items()
             if k in feat_cols or k in raw_df.columns]
        )
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<div class="section-header">🏗️ How the Stacking Ensemble Works</div>', unsafe_allow_html=True)
        st.markdown(info_box(
            "<b>Step 1 — Base learners train on the data:</b><br>"
            "&nbsp;&nbsp;• Linear Regression — captures linear trends<br>"
            "&nbsp;&nbsp;• Random Forest — captures non-linear patterns<br>"
            "&nbsp;&nbsp;• SVR — handles outliers with kernel trick<br><br>"
            "<b>Step 2 — Base learners make predictions (cross-validated)</b><br><br>"
            "<b>Step 3 — Ridge meta-model learns from base predictions</b><br>"
            "&nbsp;&nbsp;→ Final output: best combined sales forecast"
        ), unsafe_allow_html=True)

    st.markdown('<div class="section-header">🗂️ Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(8), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 · DATA EXPLORATION
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">📊 Exploring the Sales Data</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # Sales distribution
    with col_l:
        fig, ax = dark_fig(6, 3.5)
        ax.hist(raw_df["Sales"], bins=50, color="#8172B2", edgecolor="none", alpha=0.85)
        ax.set_title("Daily Sales Distribution", color="#f9fafb", fontweight="bold")
        ax.set_xlabel("Sales (€)", color="#9ca3af")
        ax.set_ylabel("Number of Store-Days", color="#9ca3af")
        ax.axvline(raw_df["Sales"].mean(), color="#f59e0b", linestyle="--",
                   linewidth=1.5, label=f"Mean: €{raw_df['Sales'].mean():,.0f}")
        ax.legend(facecolor="#1a1f2b", labelcolor="#f9fafb", fontsize=8)
        st.pyplot(fig)
        st.caption("Most store-days cluster around the mean; a long right tail shows high-sales days (weekends/promos).")

    # Sales by DayOfWeek
    with col_r:
        if "DayOfWeek" in raw_df.columns:
            day_avg = raw_df.groupby("DayOfWeek")["Sales"].mean()
            day_labels = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
            fig, ax = dark_fig(6, 3.5)
            bars = ax.bar([day_labels.get(d, str(d)) for d in day_avg.index],
                          day_avg.values, color="#4C72B0", edgecolor="none")
            ax.set_title("Average Sales by Day of Week", color="#f9fafb", fontweight="bold")
            ax.set_xlabel("Day", color="#9ca3af")
            ax.set_ylabel("Avg Sales (€)", color="#9ca3af")
            st.pyplot(fig)
            st.caption("Sales peak mid-week and on Saturdays; Sundays show near-zero (stores closed).")

    col_l2, col_r2 = st.columns(2)

    # Promo vs No Promo
    with col_l2:
        if "Promo" in raw_df.columns:
            promo_avg = raw_df.groupby("Promo")["Sales"].mean()
            fig, ax = dark_fig(5, 3.5)
            bars = ax.bar(["No Promo", "Promo"], promo_avg.values,
                          color=["#C44E52", "#55A868"], edgecolor="none", width=0.5)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                        f"€{bar.get_height():,.0f}", ha="center", va="bottom",
                        fontsize=10, color="#f9fafb", fontweight="bold")
            ax.set_title("Avg Sales: Promo vs No Promo", color="#f9fafb", fontweight="bold")
            ax.set_ylabel("Avg Sales (€)", color="#9ca3af")
            st.pyplot(fig)
            st.caption("Promotions clearly boost daily sales — a key feature for the model.")

    # Competition Distance vs Sales
    with col_r2:
        if "CompetitionDistance" in raw_df.columns:
            sample = raw_df[raw_df["Sales"] > 0].sample(min(3000, len(raw_df)), random_state=42)
            fig, ax = dark_fig(6, 3.5)
            ax.scatter(sample["CompetitionDistance"], sample["Sales"],
                       alpha=0.25, s=8, color="#4C72B0")
            ax.set_title("Competition Distance vs Sales", color="#f9fafb", fontweight="bold")
            ax.set_xlabel("Distance to Nearest Competitor (m)", color="#9ca3af")
            ax.set_ylabel("Sales (€)", color="#9ca3af")
            st.pyplot(fig)
            st.caption("Stores farther from competitors tend to have slightly higher sales.")

    # Correlation heatmap
    st.markdown('<div class="section-header">🔗 Feature Correlation with Sales</div>', unsafe_allow_html=True)
    numeric_df = raw_df.select_dtypes(include=np.number).fillna(0)
    top_cols   = numeric_df.corr()["Sales"].abs().sort_values(ascending=False).head(12).index.tolist()
    corr       = numeric_df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor(BG)
    sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                linewidths=0.4, linecolor="#0E1117",
                cbar_kws={"shrink": 0.7}, annot_kws={"size": 8})
    ax.set_facecolor(BG)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.set_title("Top 12 Features — Correlation Matrix", color="#f9fafb",
                 fontweight="bold", pad=10)
    st.pyplot(fig)
    st.caption("Customers and Promo show the strongest positive correlation with Sales.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 · MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🤖 How Each Model Performs on Store Sales Prediction</div>',
                unsafe_allow_html=True)

    # Highlight best
    def highlight_best(s):
        styles = []
        for v in s:
            if   s.name == "R2"            and v == s.max(): styles.append("background-color:#8172B255;font-weight:bold")
            elif s.name in ("MAE","RMSE")  and v == s.min(): styles.append("background-color:#8172B255;font-weight:bold")
            else: styles.append("")
        return styles

    st.dataframe(
        comparison.style
            .apply(highlight_best, subset=["MAE","RMSE","R2"])
            .format({"MAE":"€{:.2f}", "RMSE":"€{:.2f}", "R2":"{:.4f}"}),
        use_container_width=True, hide_index=True
    )
    st.caption("Purple highlight = best value. MAE/RMSE in € — lower is better. R² closer to 1 = better.")

    # Bar charts
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor("#1a1f2b"); ax.spines[:].set_color("#2d3748")
    bar_chart(axes[0], comparison, "MAE",  "Mean Abs Error (€) ↓",      "MAE (€)")
    bar_chart(axes[1], comparison, "RMSE", "Root Mean Sq Error (€) ↓",  "RMSE (€)")
    bar_chart(axes[2], comparison, "R2",   "R² Score ↑",                 "R²")
    plt.suptitle("Model Comparison — Retail Sales Prediction",
                 color="#f9fafb", fontweight="bold", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

    # Actual vs Predicted
    st.markdown('<div class="section-header">📈 Actual vs Predicted Sales</div>', unsafe_allow_html=True)
    sel_model = st.selectbox("Select model to inspect",
                             list(models.keys()),
                             index=list(models.keys()).index("Stacking Ensemble"))
    y_p    = preds[sel_model]
    color  = COLORS[sel_model]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(BG)

    ax = axes[0]; ax.set_facecolor("#1a1f2b"); ax.spines[:].set_color("#2d3748")
    ax.scatter(y_test.values, y_p, alpha=0.3, s=8, color=color)
    mn, mx = min(y_test.min(), y_p.min()), max(y_test.max(), y_p.max())
    ax.plot([mn, mx], [mn, mx], "w--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Sales (€)", color="#9ca3af")
    ax.set_ylabel("Predicted Sales (€)", color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    ax.legend(facecolor="#1a1f2b", labelcolor="#f9fafb", fontsize=8)
    ax.set_title(f"{sel_model}: Actual vs Predicted", color="#f9fafb", fontweight="bold")

    ax = axes[1]; ax.set_facecolor("#1a1f2b"); ax.spines[:].set_color("#2d3748")
    ax.hist(y_test.values - y_p, bins=50, color=color, edgecolor="none", alpha=0.8)
    ax.axvline(0, color="white", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (€)", color="#9ca3af")
    ax.set_ylabel("Count", color="#9ca3af"); ax.tick_params(colors="#9ca3af")
    ax.set_title("Residuals (errors centred near 0 = good)", color="#f9fafb", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig)

    # Sorted prediction
    st.markdown('<div class="section-header">📉 Sorted Sales: Actual vs Predicted</div>', unsafe_allow_html=True)
    sorted_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_p})\
                  .sort_values("Actual").reset_index(drop=True)
    idx     = np.linspace(0, len(sorted_df)-1, min(500, len(sorted_df)), dtype=int)
    plot_df = sorted_df.iloc[idx]

    fig, ax = dark_fig(13, 4)
    ax.plot(plot_df["Actual"].values,    label="Actual Sales",    color="#4C72B0", linewidth=1.5)
    ax.plot(plot_df["Predicted"].values, label="Predicted Sales", color=color,     linewidth=1.5, alpha=0.85)
    ax.legend(facecolor="#1a1f2b", labelcolor="#f9fafb")
    ax.set_xlabel("Stores sorted by actual sales", color="#9ca3af")
    ax.set_ylabel("Sales (€)", color="#9ca3af")
    ax.set_title(f"{sel_model} — How well does it follow the actual trend?",
                 color="#f9fafb", fontweight="bold")
    st.pyplot(fig)

    # RF Feature importances
    st.markdown('<div class="section-header">🔑 Which Features Drive Sales Most? (Random Forest)</div>',
                unsafe_allow_html=True)
    rf_importances = pd.Series(
        models["Random Forest"].named_steps["model"].feature_importances_,
        index=feat_cols
    ).sort_values(ascending=False).head(12)

    fig, ax = dark_fig(9, 4)
    ax.barh(rf_importances.index[::-1], rf_importances.values[::-1],
            color="#55A868", edgecolor="none")
    ax.set_xlabel("Importance Score", color="#9ca3af")
    ax.set_title("Top Features Influencing Sales (Random Forest)",
                 color="#f9fafb", fontweight="bold")
    st.pyplot(fig)
    st.caption("Higher bar = more influence on the sales prediction.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 · PREDICT STORE SALES
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🔮 Predict Daily Sales for a Store</div>',
                unsafe_allow_html=True)
    st.markdown(info_box(
        "Enter the store details below. All 4 models will predict the expected "
        "<b>daily sales in €</b> for that store on that day. "
        "The <b>Stacking Ensemble</b> gives the most accurate prediction."
    ), unsafe_allow_html=True)

    col_form, col_results = st.columns([1, 1])

    with col_form:
        st.markdown("#### 🏪 Store Details")

        user_input = {}

        # ── Smart input widgets based on column name ──────────────────────
        # Binary / small range → selectbox or slider
        # Large range → number input
        binary_cols  = {"Promo", "Promo2", "Open", "SchoolHoliday", "StateHoliday"}
        day_cols     = {"DayOfWeek"}
        month_cols   = {"Month", "PromoInterval"}
        cat_like     = {"StoreType", "Assortment"}

        for feat in feat_cols:
            if not pd.api.types.is_numeric_dtype(raw_df[feat]):
                continue
            col_min  = float(raw_df[feat].min())
            col_max  = float(raw_df[feat].max())
            col_mean = float(raw_df[feat].mean())

            if col_min == col_max:
                user_input[feat] = col_mean
                continue

            if feat in binary_cols or (col_max - col_min) <= 1:
                user_input[feat] = st.selectbox(
                    f"{feat}  (0 = No, 1 = Yes)",
                    options=[0, 1],
                    index=int(round(col_mean)),
                    key=f"sel_{feat}"
                )
            elif feat == "DayOfWeek":
                day_map = {"Monday":1,"Tuesday":2,"Wednesday":3,
                           "Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
                chosen  = st.selectbox("DayOfWeek", list(day_map.keys()), index=0, key="dow")
                user_input[feat] = day_map[chosen]
            elif feat in {"Month"}:
                month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
                chosen = st.selectbox("Month", list(month_map.keys()), index=6, key="month")
                user_input[feat] = month_map[chosen]
            elif feat == "CompetitionDistance":
                user_input[feat] = st.number_input(
                    "CompetitionDistance (metres to nearest rival store)",
                    min_value=col_min, max_value=col_max,
                    value=round(col_mean, 0), step=100.0, key=f"n_{feat}"
                )
            elif feat == "Customers":
                user_input[feat] = st.number_input(
                    "Customers (expected footfall today)",
                    min_value=col_min, max_value=col_max,
                    value=round(col_mean, 0), step=10.0, key=f"n_{feat}"
                )
            elif col_max - col_min <= 20:
                user_input[feat] = st.slider(feat, col_min, col_max,
                                             round(col_mean, 1), key=f"sl_{feat}")
            else:
                user_input[feat] = st.number_input(
                    feat, min_value=col_min, max_value=col_max,
                    value=round(col_mean, 2), key=f"n_{feat}"
                )

        predict_btn = st.button("🚀 Predict Sales", use_container_width=True, type="primary")

    with col_results:
        if predict_btn:
            input_df = pd.DataFrame([user_input])
            for col in feat_cols:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[feat_cols].astype(float)

            st.markdown("#### 💰 Predicted Daily Sales")

            pred_rows = []
            for name, model in models.items():
                val = max(0.0, float(model.predict(input_df)[0]))
                pred_rows.append({"Model": name, "Predicted Sales (€)": round(val, 2)})
            pred_df = pd.DataFrame(pred_rows)

            for _, row in pred_df.iterrows():
                badge = '<span class="winner-badge">★ Most Accurate</span>' \
                        if row["Model"] == best_model_name else ""
                st.markdown(
                    styled_metric(row["Model"] + badge,
                                  f"€ {row['Predicted Sales (€)']:,.2f}"),
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # Comparison bar
            fig, ax = dark_fig(6, 3)
            bars = ax.bar(pred_df["Model"], pred_df["Predicted Sales (€)"],
                          color=[COLORS[m] for m in pred_df["Model"]],
                          edgecolor="none", width=0.55)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                        f"€{bar.get_height():,.0f}",
                        ha="center", va="bottom", fontsize=9, color="#f9fafb")
            ax.set_ylabel("Predicted Sales (€)", color="#9ca3af")
            ax.set_title("All Models — Today's Sales Prediction",
                         color="#f9fafb", fontweight="bold")
            ax.tick_params(axis="x", rotation=15)
            st.pyplot(fig)

            # Interpretation
            stack_pred = pred_df.loc[pred_df["Model"]=="Stacking Ensemble",
                                     "Predicted Sales (€)"].values[0]
            avg_sales  = float(raw_df["Sales"].mean())
            diff_pct   = ((stack_pred - avg_sales) / avg_sales) * 100
            direction  = "above" if diff_pct > 0 else "below"

            st.markdown(info_box(
                f"📌 <b>Interpretation:</b> The Stacking Ensemble predicts <b>€{stack_pred:,.0f}</b> "
                f"in sales for this store today — that is <b>{abs(diff_pct):.1f}% {direction}</b> "
                f"the dataset average of €{avg_sales:,.0f}."
            ), unsafe_allow_html=True)

        else:
            st.markdown("")
            st.markdown(info_box(
                "👈 Fill in the store details on the left and click <b>Predict Sales</b>.<br><br>"
                "The model will estimate how much this store will sell today in €, "
                "based on its promotion status, nearby competition, holiday flags and more."
            ), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("SalesDynamics Forecast  ·  Retail Store Sales Prediction  ·  "
           "Stacking Ensemble: LR + RF + SVR → Ridge Meta-Model  ·  Group Project")