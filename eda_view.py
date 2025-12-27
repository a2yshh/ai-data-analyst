import pandas as pd
import streamlit as st


def render_eda(eda: dict):
    st.markdown("## Dataset Overview")

    c1, c2 = st.columns(2)
    c1.metric("Rows", eda["shape"]["rows"])
    c2.metric("Columns", eda["shape"]["columns"])

    # ---------- Column Types ----------
    st.markdown("### Column Types")
    types_df = (
        pd.Series(eda["column_types"])
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Type", 0: "Count"})
    )
    st.dataframe(types_df, use_container_width=True)

    # ---------- Missing Values ----------
    missing = {k: v for k, v in eda["missing_values"].items() if v > 0}
    if missing:
        st.markdown("### Missing Values")
        st.dataframe(
            pd.DataFrame.from_dict(
                missing,
                orient="index",
                columns=["Missing Count"]
            ),
            use_container_width=True
        )

    # ---------- Numeric Summary ----------
    if eda.get("numeric_summary"):
        # Build dataframe
        num_df = pd.DataFrame(eda["numeric_summary"]).T

        # Convert everything to numeric (safety)
        num_df = num_df.apply(pd.to_numeric, errors="coerce")

        # Drop rows where all stats are NaN
        num_df = num_df.dropna(how="all")

        # Drop features with zero or near-zero variance
        if "std" in num_df.columns:
            num_df = num_df[num_df["std"].fillna(0) > 0]

        # Drop constant columns (min == max)
        if {"min", "max"}.issubset(num_df.columns):
            num_df = num_df[num_df["min"] != num_df["max"]]

        if not num_df.empty:
            st.markdown("### Numeric Feature Summary")
            st.dataframe(
                num_df.style.format("{:.2f}"),
                use_container_width=True
            )
        else:
            st.info("No numeric features with meaningful variation to display.")

    # ---------- Outliers ----------
    outliers = {k: v for k, v in eda["outliers"].items() if v > 0}
    if outliers:
        st.markdown("### Potential Outliers (IQR)")
        st.dataframe(
            pd.DataFrame.from_dict(
                outliers,
                orient="index",
                columns=["Outlier Count"]
            ),
            use_container_width=True
        )

    # ---------- Correlations ----------
    if eda.get("correlations"):
        st.markdown("### Strong Correlations (>|0.6|)")
        corr_df = pd.DataFrame(eda["correlations"])

        strong = (
            corr_df.where(corr_df.abs() > 0.6)
            .stack()
            .reset_index()
        )
        strong.columns = ["Feature 1", "Feature 2", "Correlation"]
        strong = strong[strong["Feature 1"] != strong["Feature 2"]]

        if not strong.empty:
            st.dataframe(strong, use_container_width=True)
