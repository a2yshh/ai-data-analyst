import streamlit as st
from analyst import load_data, suggest_prompts, prompt_to_code, run_code, ask_llm
import pandas as pd
from eda import run_eda
from router import route_query
from forecast import run_forecast, ForecastError
from analyst import handle_top_k_query



st.set_page_config(page_title="Personal AI Data Analyst", layout="wide")
st.title("Personal AI Data Analyst — Interactive Dashboard")



uploaded = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv","xls","xlsx","json"])
if uploaded is None:
    st.info("Upload a CSV / XLSX / JSON to get started. Suggestions will appear automatically.")
    st.stop()

# Load data
try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.success("File loaded.")
with st.expander("Preview data (first 100 rows)"):
    st.dataframe(df.head(100))
# Run automatic EDA once
try:
    eda_summary = run_eda(df)
except Exception as e:
    st.error(f"EDA failed: {e}")
    st.stop()

from eda_view import render_eda

with st.expander("Automatic EDA summary"):
    render_eda(eda_summary)


# Generate suggestions
suggestions = suggest_prompts(df)
st.markdown("## Suggested analyses (pick one or write your own)")
col1, col2 = st.columns([3,1])
with col1:
    selected = st.selectbox("Choose a suggested prompt", options=suggestions)
    custom = st.text_area("Or write a custom prompt (leave blank to use the selected suggestion)", height=80)
with col2:
    st.markdown("**Quick actions**")
    if st.button("Show suggestions again"):
        st.write(suggestions)

# Determine final prompt
final_prompt = custom.strip() if custom and custom.strip() else selected

st.markdown("### Final prompt")
st.write(final_prompt)

# Run button
# Run button
if st.button("Run analysis"):
    with st.spinner("Running..."):

        # ---------- ROUTE QUERY ----------
        task = route_query(final_prompt)

        # ---------- FAST DETERMINISTIC HANDLERS ----------
        res = handle_top_k_query(final_prompt, df)
        if res:
            if res["type"] == "dataframe":
                st.markdown("#### Output (table)")
                st.dataframe(res["df"], use_container_width=True)

                csv = res["df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download result as CSV",
                    data=csv,
                    file_name="result.csv",
                    mime="text/csv",
                )
                st.stop()

        # ---------- STATS / EDA ----------
        if task == "stats":
            st.markdown("#### Deterministic EDA result")
            st.json(eda_summary)
            st.stop()

        # ---------- TIME-SERIES FORECAST ----------
        if task == "forecast":
            date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

            if not date_cols:
                for col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_cols.append(col)
                        break
                    except Exception:
                        pass

            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            if not date_cols or not numeric_cols:
                st.error("Forecasting requires one datetime column and one numeric column.")
                st.stop()

            try:
                forecast_res = run_forecast(
                    df,
                    date_col=date_cols[0],
                    target_col=numeric_cols[0],
                    steps=12,
                )
            except ForecastError as e:
                st.error(f"Forecast failed: {e}")
                st.stop()

            st.markdown("#### Time-Series Forecast")
            st.write(f"**Model:** {forecast_res['model']} {forecast_res['order']}")
            st.json(forecast_res)
            st.stop()

        # ---------- FALLBACK ----------
        st.error(
            "This query is not supported yet.\n\n"
            "Try a simpler analytical request such as:\n"
            "- top / bottom values\n"
            "- sorting\n"
            "- filtering\n"
            "- basic statistics"
        )
        st.stop()


    # Display result
    if res["type"] == "text":
        st.markdown("#### Output (text)")
        st.text(res["output"])
    elif res["type"] == "dataframe":
        st.markdown("#### Output (table)")
        st.dataframe(res["df"])
        # Provide CSV download
        csv = res["df"].to_csv(index=False).encode("utf-8")
        st.download_button("Download result as CSV", data=csv, file_name="result.csv", mime="text/csv")
    elif res["type"] == "image":
        st.markdown("#### Output (chart)")
        st.image(res["path"], use_column_width=True)
    else:
        st.write("Unknown result type", res)