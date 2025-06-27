# ================================================================
# Customer Analytics Dashboard – ORDERS-ONLY Edition
# ================================================================
import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from datetime import timedelta
from typing import Tuple, Dict
from operator import attrgetter

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import shap

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
try:                       # lifetimes < v0.12
    from lifetimes.exceptions import ConvergenceError
except ModuleNotFoundError:
    class ConvergenceError(Exception):
        ...

# ──────────────────────────────────────────────────────────────── #
st.set_page_config(page_title="Customer Analytics MVP", layout="wide")
COLOR_SEQ = px.colors.qualitative.Set2
PX_OPTS   = dict(color_discrete_sequence=COLOR_SEQ)

# ──────────────────────────────────────────────────────────────── #
# 1 · Load a single **Orders** export
# ──────────────────────────────────────────────────────────────── #
@st.cache_data(show_spinner=False)
def load_orders(path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    orders = pd.read_csv(path)

    orders = orders.rename(columns={
        "Email":               "customer_id",
        "Name":                "order_id",
        "Created at":          "order_date",
        "Total":               "order_total",
        "Lineitem sku":        "sku",
        "Lineitem price":      "unit_price",
        "Lineitem quantity":   "qty",
        "Lineitem line_price": "line_total",
    })

    # core types ----------------------------------------------------------
    orders["order_date"]  = pd.to_datetime(orders["order_date"], errors="coerce")
    orders["order_total"] = pd.to_numeric(orders["order_total"], errors="coerce")

    # revenue at line-item level (for product RFM) -----------------------
    if "line_total" in orders.columns:
        orders["payment_value"] = pd.to_numeric(orders["line_total"], errors="coerce")
    else:
        orders["payment_value"] = (
            pd.to_numeric(orders["unit_price"], errors="coerce") *
            pd.to_numeric(orders["qty"].fillna(1), errors="coerce")
        )

    # dummy product columns so Product-RFM never crashes -----------------
    orders["product_id"]       = orders.get("sku")
    orders["product_category"] = orders.get("Product Type", "Unknown")

    before = len(orders)
    orders = orders.dropna(subset=["customer_id", "order_id", "order_date", "order_total"])
    dropped = before - len(orders)
    return orders, {"rows_final": len(orders), "rows_dropped": dropped}

# ──────────────────────────────────────────────────────────────── #
# 2 · RFM helpers (customer & product)
# ──────────────────────────────────────────────────────────────── #
def segment_map(r, f, m):
    if (r, f, m) == (4, 4, 4): return "A.CHAMPIONS"
    if r >= 3 and f >= 3:      return "B.LOYAL"
    if r >= 3:                 return "C.POTENTIAL_LOYALIST"
    if r == 4:                 return "D.RECENT_CUSTOMERS"
    if r == 3:                 return "E.PROMISING"
    if f >= 3:                 return "F.NEED_ATTENTION"
    if r == 2:                 return "G.ABOUT_TO_SLEEP"
    if r == 1 and f >= 2:      return "H.AT_RISK"
    if r == 1 and f == 1 and m >= 2: return "I.CANNOT_LOSE"
    return "J.HIBERNATING"

@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame, q: int = 4):
    snap = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_id")
          .agg(
              Recency  = ("order_date",  lambda x: (snap - x.max()).days),
              Frequency= ("order_id",   "nunique"),
              Monetary = ("order_total","sum")
          )
          .reset_index()
    )
    rfm["R"] = pd.qcut(rfm["Recency"], q, labels=list(range(q, 0, -1)), duplicates="drop").astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), q, labels=list(range(1, q + 1)), duplicates="drop").astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"],  q, labels=list(range(1, q + 1)), duplicates="drop").astype(int)
    rfm["RFM_Score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["Segment"]   = rfm.apply(lambda x: segment_map(x["R"], x["F"], x["M"]), axis=1)
    return rfm, snap

@st.cache_data(show_spinner=False)
def compute_product_rfm(df: pd.DataFrame, q: int = 4, col: str = "sku"):
    snap = df["order_date"].max() + pd.Timedelta(days=1)
    prfm = (
        df.groupby(col)
          .agg(
              Recency  = ("order_date",  lambda x: (snap - x.max()).days),
              Frequency= ("order_id",   "nunique"),
              Monetary = ("payment_value","sum")
          )
          .reset_index()
          .rename(columns={col: "product"})
    )
    def _q(series, rev=False):
        codes = pd.qcut(series, q, duplicates="drop").cat.codes
        return (codes.max() - codes + 1) if rev else (codes + 1)

    prfm["R"] = _q(prfm["Recency"],  rev=True)
    prfm["F"] = _q(prfm["Frequency"])
    prfm["M"] = _q(prfm["Monetary"])
    prfm["RFM_Score"] = prfm["R"].astype(str) + prfm["F"].astype(str) + prfm["M"].astype(str)
    return prfm, snap

# ──────────────────────────────────────────────────────────────── #
# 3 · Insight-box helper
# ──────────────────────────────────────────────────────────────── #
def insight_box(title: str, bullets: list[str], colour: str = "blue"):
    bg = {"blue": "#e8f4fd", "green": "#e9f8f1", "yellow": "#fff8db", "red": "#fdecea"}
    br = {"blue": "#90c2f1", "green": "#8dd9b6", "yellow": "#f6d36b", "red": "#f5a3a3"}
    html = (
        f'<div style="border-left:6px solid {br[colour]}; '
        f'background:{bg[colour]}; padding:0.75rem 1rem; border-radius:4px;">'
        f'<strong>{title}</strong><ul>'
        + "".join(f"<li>{b}</li>" for b in bullets) +
        "</ul></div>"
    )
    st.markdown(html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────── #
# 4 · SIDEBAR – file + global controls
# ──────────────────────────────────────────────────────────────── #
orders_file = st.sidebar.file_uploader("Shopify Orders CSV", type="csv")

if orders_file:
    df, dq = load_orders(orders_file)
    st.sidebar.success(f"Loaded {dq['rows_final']} rows "
                       f"({dq['rows_dropped']} dropped)")
else:
    demo_path = "demo_shopify_orders.csv"
    df, dq    = load_orders(demo_path)
    st.sidebar.info(f"Demo dataset · {dq['rows_final']} rows")

churn_days = st.sidebar.slider("Churn threshold (days)", 30, 180, 60, 5)
q_buckets  = st.sidebar.slider("RFM buckets", 3, 5, 4)
lifespan   = st.sidebar.slider("Avg lifespan (years)", 1, 10, 3)

page = st.sidebar.radio(
    "Page",
    [
        "Data Preview", "RFM Segmentation", "Cohort Analysis",
        "Churn Prediction", "Customer LTV", "Customer Journey",
        "Product RFM"
    ]
)

# ──────────────────────────────────────────────────────────────── #
# 0 · DATA PREVIEW
# ──────────────────────────────────────────────────────────────── #
if page == "Data Preview":
    st.title("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    with st.expander("Data-quality report", expanded=True):
        st.json(dq)

# ──────────────────────────────────────────────────────────────── #
# 1 · RFM SEGMENTATION  (overview + drill-down)
# ──────────────────────────────────────────────────────────────── #
elif page == "RFM Segmentation":
    st.title("📦 RFM Segmentation")
    rfm, snap = compute_rfm(df, q_buckets)
    st.write(f"Snapshot date: **{snap.date()}**")

    tab_over, tab_drill = st.tabs(["Overview", "Drill-down"])

    # ---------- Overview ------------------------------------------------
    with tab_over:
        seg_sum = (
            rfm.groupby("Segment")
               .agg(mean_recency=("Recency", "mean"),
                    revenue=("Monetary", "sum"),
                    customers=("customer_id", "count"))
               .reset_index()
        )
        fig = px.scatter(
            seg_sum, x="mean_recency", y="revenue", size="customers",
            color="Segment", **PX_OPTS, log_y=True, height=500,
            title="Segment landscape",
            hover_data={"mean_recency":":.1f", "revenue":":,.0f"}
        )
        fig.update_layout(
            xaxis_title="Mean Recency (days)",
            yaxis_title="Total revenue"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg_sum.sort_values("revenue", ascending=False), use_container_width=True)
        st.download_button("Download segment summary", seg_sum.to_csv(index=False), "rfm_segment_summary.csv")

        # consultant bullets ---------------------------------------------
        top_rev  = seg_sum.sort_values("revenue",   ascending=False).iloc[0]
        top_size = seg_sum.sort_values("customers", ascending=False).iloc[0]
        at_risk  = seg_sum[seg_sum["Segment"].str.contains("AT_RISK|HIBERNATING")]
        share    = top_rev["revenue"] / seg_sum["revenue"].sum()

        bullets = [
            f"🔝 **{top_rev['Segment']}** accounts for **{share:.1%}** of total revenue "
            f"(€{top_rev['revenue']:,.0f} / {top_rev['customers']} customers).",
            f"👥 Largest segment: **{top_size['Segment']}** "
            f"({top_size['customers']} customers, avg €{top_size['revenue']/top_size['customers']:,.0f})."
        ]
        if not at_risk.empty:
            bullets.append(
                f"⚠️ *At-Risk / Hibernating* customers: {at_risk['customers'].sum()} "
                f"(€{at_risk['revenue'].sum():,.0f} potential lost revenue)."
            )
        bullets += [
            "💡 **Next steps:**",
            "• Send a VIP coupon to *Champions* / *Loyal* segments.",
            "• Launch a win-back email/SMS flow for *At-Risk*.",
        ]
        insight_box("Consultant summary", bullets, "green")

    # ---------- Drill-down ---------------------------------------------
    with tab_drill:
        seg = st.selectbox("Choose segment", sorted(rfm["Segment"].unique()))
        seg_df = rfm[rfm["Segment"] == seg]
        st.dataframe(
            seg_df.sort_values("Monetary", ascending=False),
            use_container_width=True, height=450
        )
        st.download_button(
            f"Download {seg} customers",
            seg_df.to_csv(index=False),
            f"{seg.lower()}_customers.csv"
        )

# ──────────────────────────────────────────────────────────────── #
# 2 · COHORT ANALYSIS  (heatmap + counts)
# ──────────────────────────────────────────────────────────────── #
elif page == "Cohort Analysis":
    st.title("📈 Cohort Analysis")
    df["order_month"]  = df["order_date"].dt.to_period("M")
    df["cohort_month"] = df.groupby("customer_id")["order_date"].transform("min").dt.to_period("M")
    df["cohort_index"] = (df["order_month"] - df["cohort_month"]).apply(attrgetter("n")) + 1

    cohorts = df.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().reset_index()
    pivot_counts  = cohorts.pivot(index="cohort_month", columns="cohort_index", values="customer_id").fillna(0)
    pivot_percent = pivot_counts.divide(pivot_counts.iloc[:, 0], axis=0)
    pivot_percent.index = pivot_percent.index.to_timestamp()

    tab_heat, tab_counts = st.tabs(["Heatmap %", "Counts table"])

    # heatmap %
    with tab_heat:
        heat = px.imshow(
            pivot_percent, text_auto=".0%", color_continuous_scale="YlGnBu",
            aspect="auto", labels=dict(color="Retention"), height=500
        )
        heat.update_layout(
            xaxis_title="Cohort index (months)",
            yaxis_title="Cohort month"
        )
        st.plotly_chart(heat, use_container_width=True)
        st.download_button("Download retention %", pivot_percent.to_csv(), "retention_percent.csv")

    # raw counts
    with tab_counts:
        st.dataframe(pivot_counts, use_container_width=True, height=450)
        st.download_button("Download counts", pivot_counts.to_csv(), "cohort_counts.csv")

    # consultant insight
    m3 = pivot_percent[3].dropna()
    bullets = [
        f"🔎 Median 3-month retention: **{m3.median():.1%}**."
    ] if not m3.empty else [
        "🔎 Fewer than 3 cohort months available — too early to measure retention."
    ]
    if len(m3) > 1:
        delta = m3.iloc[-1] - m3.iloc[0]
        trend = "improved" if delta > 0 else "declined"
        bullets.append(
            f"📈 Retention has **{trend}** by **{delta:.1%}** from the first to the latest cohort."
        )
    bullets += [
        "💡 **Actions:**",
        "• Strengthen the first-24h onboarding sequence.",
        "• Introduce a referral incentive in month 2.",
    ]
    insight_box("Consultant note", bullets, "blue")





# ------------------------------------------------------------------ #
# 3 · CHURN PREDICTION  + KAPLAN-MEIER
# ------------------------------------------------------------------ #
elif page == "Churn Prediction":
    st.title("📉 Churn Prediction – ML & Survival")

    # build churn flag from RFM -----------------------------------------
    rfm, _ = compute_rfm(df, q_buckets)
    rfm["churned"] = (rfm["Recency"] > churn_days).astype(int)

    X = rfm[["Recency", "Frequency", "Monetary"]]
    y = rfm["churned"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.30, random_state=42
    )
    spw = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    # — fit XGBoost (cached) -------------------------------------------
    @st.cache_resource(show_spinner=False)
    def fit_xgb(Xtr, ytr, spw_):
        mdl = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=spw_
        )
        mdl.fit(Xtr, ytr)
        return mdl

    model = fit_xgb(X_tr, y_tr, spw)

    tab_ml, tab_surv = st.tabs(["ML metrics", "Kaplan-Meier"])

    # ───────────────── ML metrics tab ─────────────────────────────────
    with tab_ml:
        cv_auc = cross_val_score(
            model, X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring="roc_auc"
        )
        st.metric("CV AUC", f"{cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

        # confusion matrix ---------------------------------------------
        cm = confusion_matrix(y_te, model.predict(X_te))
        cm_fig = go.Figure(
            go.Heatmap(
                z=cm, text=cm, texttemplate="%{text}",
                colorscale="Blues", showscale=False
            )
        )
        cm_fig.update_layout(
            title="Confusion matrix",
            xaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),
            yaxis=dict(tickvals=[0, 1], ticktext=["No", "Yes"]),
            height=350
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        # classification report ----------------------------------------
        rep_df = pd.DataFrame(
            classification_report(y_te, model.predict(X_te), output_dict=True)
        ).T.round(3)
        st.dataframe(rep_df, use_container_width=True)

        # SHAP importance ---------------------------------------------
        expl      = shap.TreeExplainer(model)
        shap_vals = expl.shap_values(X)

        shap_fig = px.bar(
            x=X.columns,
            y=np.abs(shap_vals).mean(axis=0),
            **PX_OPTS,
            title="Feature importance (SHAP)",
            labels={"x": "Feature", "y": "|SHAP| mean"}
        )
        st.plotly_chart(shap_fig, use_container_width=True)

        # probabilities export ----------------------------------------
        rfm["churn_probability"] = model.predict_proba(X)[:, 1]
        st.subheader("Top-20 customers by churn probability")
        st.dataframe(
            rfm.sort_values("churn_probability", ascending=False).head(20),
            use_container_width=True
        )
        st.download_button(
            "Download churn predictions",
            rfm.to_csv(index=False),
            "churn_predictions.csv"
        )

        # interpretation insight --------------------------------------
        fnr = cm[1, 0] / cm.sum()
        insight_box(
            "Model interpretation",
            [
                f"AUC **{cv_auc.mean():.3f}** (rule-of-thumb: > 0.75 is strong).",
                f"False-negative rate **{fnr:.1%}** (churners the model misses).",
                "💡 Offer a personalised voucher to customers with probability > 0.60 "
                "— check ROI against predicted CLV."
            ],
            "yellow"
        )

    # ───────────────── Kaplan-Meier tab ───────────────────────────────
    with tab_surv:
        from lifelines import KaplanMeierFitter

        surv = (
            df.groupby("customer_id")
              .agg(first=("order_date", "min"), last=("order_date", "max"))
              .reset_index()
        )
        surv["tenure"]  = (surv["last"] - surv["first"]).dt.days
        surv["churned"] = (
            (df["order_date"].max() - surv["last"]).dt.days > churn_days
        ).astype(int)

        kmf = KaplanMeierFitter()
        kmf.fit(surv["tenure"], surv["churned"])

        km_fig = go.Figure()
        km_fig.add_scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_["KM_estimate"],
            mode="lines",
            line=dict(color=COLOR_SEQ[0])
        )
        km_fig.update_layout(
            height=400,
            xaxis_title="Tenure (days)",
            yaxis_title="Survival probability",
            title="Kaplan-Meier survival curve"
        )
        st.plotly_chart(km_fig, use_container_width=True)

        insight_box(
            "Survival insight",
            [
                f"Median customer lifetime: **{kmf.median_survival_time_:.0f} days**.",
                "Schedule a reminder / cross-sell touch-point ~15 days before this median."
            ],
            "blue"
        )

# ------------------------------------------------------------------ #
# 4 · CUSTOMER LTV  (BG/NBD + Γ-Γ with auto-retry)
# ------------------------------------------------------------------ #
elif page == "Customer LTV":
    st.title("💸 Customer LTV – Historic + Forecast")

    # lifetimes summary frame -------------------------------------------
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="customer_id",
        datetime_col="order_date",
        monetary_value_col="payment_value",
        observation_period_end=df["order_date"].max() + pd.Timedelta(days=1),
        freq="D"
    )
    summary = summary[summary["monetary_value"] > 0]   # Γ-Γ requirement

    # safe fit with penalizer-escalation --------------------------------
    @st.cache_resource(show_spinner=False)
    def fit_bgnbd_gg(summary_df, max_penalty: float = 0.02):
        penalty = 0.001
        while penalty <= max_penalty:
            try:
                bgf = BetaGeoFitter(penalizer_coef=penalty)
                bgf.fit(summary_df["frequency"],
                        summary_df["recency"],
                        summary_df["T"])
                ggf = GammaGammaFitter(penalizer_coef=penalty)
                ggf.fit(summary_df["frequency"],
                        summary_df["monetary_value"])
                return bgf, ggf, penalty        # converged ✔
            except ConvergenceError:
                penalty *= 2                    # retry
        return None, None, None                 # give up

    bgf, ggf, used_pen = fit_bgnbd_gg(summary)

    # forecast (or fallback) --------------------------------------------
    if bgf is None:
        st.warning(
            "BG/NBD did not converge – using historic CLV "
            "(frequency × monetary_value × lifespan)."
        )
        summary["pred_clv"] = (
            summary["frequency"] *
            summary["monetary_value"] *
            lifespan
        )
    else:
        t_months = lifespan * 12
        summary["pred_avg_val"] = ggf.conditional_expected_average_profit(
            summary["frequency"], summary["monetary_value"]
        )
        summary["pred_freq"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            t_months,
            summary["frequency"],
            summary["recency"],
            summary["T"]
        )
        summary["pred_clv"] = summary["pred_avg_val"] * summary["pred_freq"]
        st.success(f"BG/NBD converged (penalizer = {used_pen:.3f})")

    # show & export ------------------------------------------------------
    st.dataframe(
        summary.sort_values("pred_clv", ascending=False).head(20),
        use_container_width=True
    )
    st.download_button(
        "Download CLV forecast",
        summary.reset_index()[["customer_id", "pred_clv"]].to_csv(index=False),
        "clv_forecast.csv"
    )

    hist = px.histogram(
        summary,
        x="pred_clv",
        nbins=50,
        **PX_OPTS,
        title="Predicted CLV distribution",
        labels={"pred_clv": "Predicted CLV"}
    )
    hist.update_layout(height=400)
    st.plotly_chart(hist, use_container_width=True)

    insight_box(
        "CLV insight",
        [
            f"Top-10 customers represent €{summary['pred_clv'].nlargest(10).sum():,.0f}.",
            "👑 Unlock VIP perks when predicted CLV > €500."
        ],
        "green"
    )




# ------------------------------------------------------------------ #
# 5 · CUSTOMER JOURNEY
# ------------------------------------------------------------------ #
elif page == "Customer Journey":
    st.title("🧭 Customer Journey Timeline")

    snap = df["order_date"].max() + pd.Timedelta(days=1)
    journey = (
        df.groupby("customer_id")
          .agg(first_order=("order_date", "min"),
               last_order =("order_date", "max"),
               orders     =("order_id",   "count"),
               revenue    =("payment_value", "sum"))
          .reset_index()
    )
    journey["days_since_last"] = (snap - journey["last_order"]).dt.days
    journey["status"] = pd.cut(
        journey["days_since_last"],
        [-1, 30, 90, np.inf],
        labels=["Active", "Dormant", "Lost"]
    )
    timeline = df.merge(journey[["customer_id", "status"]], on="customer_id")

    tab_over, tab_cust = st.tabs(["Overview", "Inspect customer"])

    # -------- overview scatter -----------------------------------------
    with tab_over:
        fig = px.scatter(
            timeline,
            x="order_date",
            y="customer_id",
            color="status",
            hover_data=["order_id", "payment_value"],
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Status summary")
        status_tbl = (
            journey["status"]
                   .value_counts()
                   .rename_axis("Status")
                   .reset_index(name="Customers")
        )
        st.dataframe(status_tbl, use_container_width=True)

        # consultant insight --------------------------------------------
        active_ratio = (journey["status"] == "Active").mean()
        lost_ratio   = (journey["status"] == "Lost").mean()
        insight_box(
            "Customer-base health",
            [
                f"✅ Active (last 30 d): **{active_ratio:.1%}**.",
                f"🚪 Lost (>90 d): **{lost_ratio:.1%}**.",
                "💡 **Actions:**",
                "• Trigger a 3-touch drip for Dormant (31–90 days).",
                "• Run an exit-survey for Lost customers to discover churn reasons."
            ],
            "blue"
        )

    # -------- individual customer --------------------------------------
    with tab_cust:
        cust_sel = st.selectbox(
            "Choose customer",
            journey.sort_values("revenue", ascending=False)["customer_id"]
        )
        cust_df = timeline[timeline["customer_id"] == cust_sel]

        fig2 = px.scatter(
            cust_df,
            x="order_date",
            y="payment_value",
            color="status",
            **PX_OPTS,
            height=400,
            title=f"Order timeline – {cust_sel}"
        )
        fig2.update_layout(xaxis_title="Order date", yaxis_title="Order value")
        st.plotly_chart(fig2, use_container_width=True)

        cinfo = journey.set_index("customer_id").loc[cust_sel]
        st.markdown(
            f"""
            **First order:** {cinfo['first_order'].date()}  
            **Last order:** {cinfo['last_order'].date()}  
            **Orders:** {cinfo['orders']}  
            **Revenue:** €{cinfo['revenue']:,.0f}  
            **Days since last order:** {cinfo['days_since_last']}  
            **Status:** {cinfo['status']}
            """
        )

# ------------------------------------------------------------------ #
# 6 · PRODUCT-LEVEL RFM
# ------------------------------------------------------------------ #
elif page == "Product RFM":
    st.title("📦 Product-level RFM")

    candidates = ["product_id", "product_category", "sku"]
    available  = [c for c in candidates if c in df.columns]
    if not available:
        st.error("No product-level column found.")
        st.stop()

    entity_col = st.selectbox("Analyse by…", available, index=0)
    prfm, snap = compute_product_rfm(df, q_buckets, col=entity_col)
    st.write(f"Snapshot date: **{snap.date()}**")

    tab_sum, tab_top = st.tabs(["Summary table", "Top products"])

    # summary table ------------------------------------------------------
    with tab_sum:
        st.dataframe(
            prfm.sort_values("RFM_Score", ascending=False),
            use_container_width=True, height=500
        )
        st.download_button(
            "Download product RFM",
            prfm.to_csv(index=False),
            f"product_rfm_{entity_col}.csv"
        )

    # top-N visual -------------------------------------------------------
    with tab_top:
        top_n  = st.slider("Top N by revenue", 5, 100, 20)
        top_df = prfm.sort_values("Monetary", ascending=False).head(top_n)

        fig = px.scatter(
            top_df,
            x="Recency",
            y="Monetary",
            size="Frequency",
            hover_name="product",
            size_max=50,
            color_discrete_sequence=COLOR_SEQ,
            log_y=True,
            height=500,
            title=f"Top {top_n} products – Recency vs Revenue"
        )
        fig.update_layout(xaxis_title="Recency (days)", yaxis_title="Total revenue")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_df, use_container_width=True)

        lead = top_df.iloc[0]
        insight_box(
            "SKU opportunities",
            [
                f"⭐ Leading SKU: **{lead['product']}** – €{lead['Monetary']:,.0f} revenue.",
                "💡 Add cross-sell banners and bundled offers to lift Frequency."
            ],
            "green"
        )

# ------------------------------------------------------------------ #
# END OF FILE
# ------------------------------------------------------------------ #
