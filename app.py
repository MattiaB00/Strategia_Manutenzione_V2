import streamlit as st

import pipeline

st.set_page_config(
    page_title="Optimal Maintenance Strategy",
    page_icon="🛠️",
    layout="centered",
)

st.title("🛠️ Optimal Maintenance Strategy")
st.write(
    "This tool suggests, based on the FMEA scenario and a few economic/reliability "
    "parameters, which maintenance strategy is more convenient among **Corrective**, "
    "**Preventive** and **Predictive**, using a decision tree trained on the cost "
    "of each policy."
)


@st.cache_resource(show_spinner="Computing costs and training the models (one-time setup)...")
def load_models():
    return pipeline.build_all("tabella_weibull.csv")


data = load_models()
models = data["models"]
leaf_stats = data["leaf_stats"]

st.divider()
st.subheader("1. FMEA scenario")

col1, col2, col3 = st.columns(3)
with col1:
    detectability = st.selectbox(
        "Detectability",
        options=["HIGH", "MEDIUM", "LOW"],
        help="How easily an incoming failure can be detected in advance.",
    )
with col2:
    severity = st.selectbox(
        "Severity",
        options=["HIGH", "MEDIUM", "LOW"],
        help="Impact/cost of the failure if it occurs (mapped onto Cf).",
    )
with col3:
    occurrence = st.selectbox(
        "Occurrence",
        options=["HIGH", "MEDIUM", "LOW"],
        help="Expected failure frequency (mapped onto MTTF: HIGH = frequent failures).",
    )

st.subheader("2. Economic & reliability parameters")
st.caption(
    "These values are free and continuous: the decision tree evaluates them against "
    "its learned split thresholds, it is not limited to the discrete grid used for training."
)

c1, c2 = st.columns(2)
with c1:
    cinter = st.slider(
        "Cinter — cost of a single intervention (€)",
        min_value=500.0,
        max_value=8000.0,
        value=4250.0,
        step=50.0,
    )
    beta = st.slider(
        "Beta — Weibull shape parameter",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.05,
    )
with c2:
    csystpdm = st.slider(
        "CSystPdM — yearly cost of the monitoring system (€/year)",
        min_value=500.0,
        max_value=30000.0,
        value=15200.0,
        step=100.0,
    )
    alfa = st.slider(
        "Alfa — utilization level of the monitoring system",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

st.divider()

if st.button("🔍 Get recommended strategy", type="primary", use_container_width=True):
    result = pipeline.query_tree(
        models, leaf_stats, detectability, severity, occurrence, cinter, csystpdm, beta, alfa
    )

    if "error" in result:
        st.error(result["error"])
    else:
        colors = {
            "PREDICTIVE": "green",
            "PREVENTIVE": "orange",
            "CORRECTIVE": "red",
        }
        strategy = result["strategy"]
        color = colors.get(strategy, "blue")

        st.markdown(f"### Recommended strategy: :{color}[{strategy}]")

        st.write("Estimated probability for the selected scenario:")
        proba = result["proba"]
        for cls in ["PREDICTIVE", "PREVENTIVE", "CORRECTIVE"]:
            if cls in proba:
                st.progress(float(proba[cls]), text=f"{cls}: {proba[cls]:.0%}")

        saving_expected_pct = result.get("saving_expected_pct")
        cost_expected_pct = result.get("cost_expected_pct")
        n_samples = result.get("n_samples")

        if saving_expected_pct is not None:
            st.write("")
            st.write("Expected impact for this scenario:")
            m1, m2 = st.columns(2)
            m1.metric(
                "Expected extra savings (%)",
                f"{saving_expected_pct:.1%}",
                help="Historical average % saved vs. the corrective  "
                 baseline, weighted by the scenario's accuracy.",
            )
            m2.metric(
                "Expected extra cost (%)",
                f"{cost_expected_pct:.1%}",
                help="Historical average % extra cost vs. the true optimal strategy,"
                "weighted by the scenario's inaccuracy.",
            )
            st.caption(
                f"Based on historical combinations."
                "Both metrics are 0 when the leaf is 100% accurate."
            )

with st.expander("ℹ️ How it works / definitions"):
    st.markdown(
        """
- **Corrective**: intervene only after the failure occurs.
- **Preventive**: replace/intervene at scheduled intervals, computed by optimizing
  the trade-off between failure cost and early-intervention cost.
- **Predictive**: monitor the component's condition and intervene only when the
  data indicates an imminent risk.

For each of the 27 combinations of Detectability × Severity × Occurrence, a
decision tree (max depth 3) was trained that, given Cinter, CSystPdM, Beta and
Alfa, predicts which of the three strategies minimizes the expected total cost.
Since the tree only checks numeric thresholds, any continuous value for these
four parameters can be evaluated, not just the discrete grid used during training.

**Estimated probability**: this is exactly the training-sample class distribution
in the leaf your inputs land on — i.e. the leaf's accuracy for the recommended
class (and the complementary share for the other classes present in that leaf).

**Expected extra savings**: among the leaf's historical cases where the tree's
recommendation was the optimal strategy, this is the average % saved versus
the corrective (run-to-failure) baseline, weighted by the leaf's accuracy — i.e.
by how often that leaf is right.

**Expected extra cost**: among the leaf's historical cases where the tree's
recommendation was not the optimal strategy, this is the average extra cost
paid (as a % of the correct strategy's cost), weighted by the leaf's
inaccuracy — i.e. by how often that leaf is wrong. It is 0 whenever the leaf is
100% accurate.

Together, these two numbers turn the bare probability into an economic
statement: how much you stand to gain when the leaf is right, and how much
you risk when it's wrong.
        """
    )
