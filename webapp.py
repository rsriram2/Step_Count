import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

st.set_page_config(
    page_title="Step Count Distribution",
    layout="wide"
)

csvfile = 'steps_test.csv.gz'
data = pd.read_csv(csvfile, compression='gzip')

data['age_cat_display'] = data['age_cat'].str.extract(r'\[(\d+),(\d+)\)').apply(lambda x: f"{x[0]}-{int(x[1]) - 1}", axis=1)

def plot_kde_plotly(data, gender, age_range, user_step_count=None):
    original_age_range = data[data['age_cat_display'] == age_range]['age_cat'].iloc[0]
    subset = data if gender == "Overall" else data[data['gender'] == gender]
    subset = subset[subset['age_cat'] == original_age_range]
    
    # Get the KDE
    values = subset['value'].dropna()
    kde = gaussian_kde(values)
    x_range = np.linspace(0, max(values.max(), 35000), 500)
    y_range = kde(x_range)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_range, y=y_range,
        mode='lines',
        fill='tozeroy',
        line=dict(color='steelblue'),
        name=f"{gender} KDE",
        hovertemplate="Step Count: %{x}<br>Density: %{y:.6f}<extra></extra>" 
    ))

    if user_step_count is not None:
        quantile = subset[subset['value'] <= user_step_count]['q'].max()
        fig.add_trace(go.Scatter(
            x=[user_step_count, user_step_count],
            y=[0, max(y_range)],
            mode='lines',
            line=dict(color='crimson', dash='dash'),
            name=f"Your Step Count ({quantile * 100:.2f}%)",
            hovertemplate="Your Step Count: %{x}<extra></extra>"

        ))

    # Layout
    fig.update_layout(
        title_text="",
        xaxis_title="Step Count",
        yaxis_title="Density",
        xaxis_title_font=dict(size=14, color='white', family='Arial Black'),
        yaxis_title_font=dict(size=14, color='white', family='Arial Black'),
        title_x=0.5,
        template="plotly_dark",
        margin=dict(l=30, r=30, t=10, b=40),
        legend=dict(
            x=0.75,
            y=0.95,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=13, family="Arial", color="white")            
        )
    )
    
    fig.update_yaxes(
    tickformat=".6f",  
    exponentformat="none"
    )
    fig.update_xaxes(
    showgrid=True,          
    gridwidth=1,
    gridcolor="rgba(255,255,255,0.1)",  
    tickformat=".0f",
    exponentformat="none"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
    <h1 style='text-align: center; padding: 0.5rem; color: white; background-color: rgba(70, 130, 180, 0.3); border-radius: 10px;'>
        Step Count Distribution
    </h1>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

age_ranges_display = sorted(data['age_cat_display'].unique())
default_age_group = "20-29" if "20-29" in age_ranges_display else age_ranges_display[0]

left_col, spacer, right_col = st.columns([2, 0.3, 3])
    
with left_col:
    st.subheader("Your Information")
    gender = st.selectbox("Gender", ["Overall", "Male", "Female"])
    age_range = st.selectbox("Age Range", options=age_ranges_display, index=age_ranges_display.index(default_age_group))
    
    original_age_range = data[data['age_cat_display'] == age_range]['age_cat'].iloc[0]
    subset = data[data['age_cat'] == original_age_range] if gender == "Overall" else data[(data['gender'] == gender) & (data['age_cat'] == original_age_range)]
    median_step_count = subset['value'].median()
    user_step_count = st.number_input("Step Count", min_value=0, value=int(median_step_count), step=500)

    if user_step_count:
        quantile = subset[subset['value'] <= user_step_count]['q'].max()
        if pd.isna(quantile):
            st.write("### Your step count is below the threshold for this demographic.")
        else:
            percentile = round(quantile * 100, 2)
            steps_to_90th = round(max(0, subset[subset['q'] >= 0.9]['value'].min() - user_step_count))
            steps_to_95th = round(max(0, subset[subset['q'] >= 0.95]['value'].min() - user_step_count))
            steps_to_99th = round(max(0, subset[subset['q'] >= 0.99]['value'].min() - user_step_count))

            st.markdown(f"""
            <div style="background-color: rgba(70, 130, 180, 0.3); padding:20px; border-radius:10px; margin-top:20px;">
                <h4 style="color:#ffffff;"> Based on your input:</h4>
                <p style="color:#d1d5db; font-size:16px; margin-bottom: 10px;">
                    You are in the <strong style="color:#60a5fa;">{percentile:.2f}th percentile</strong>.
                </p>
                <ul style="color:#d1d5db; font-size:15px; line-height:1.6;">
                    <li> You need <strong><u>{steps_to_90th}</u></strong> more steps to reach the <strong><u>90th</u></strong> percentile.</li>
                    <li> You need <strong><u>{steps_to_95th}</u></strong> more steps to reach the <strong><u>95th</u></strong> percentile.</li>
                    <li> You need <strong><u>{steps_to_99th}</u></strong> more steps to reach the <strong><u>99th</u></strong> percentile.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with right_col:
    st.markdown(
        f"<div style='text-align: center; font-size: 24px; font-weight: bold; margin-top: 4px; white-space: nowrap;'>"
        f"Step Count Distribution: {gender} (Age {age_range})"
        f"</div>",
        unsafe_allow_html=True
    )
    plot_kde_plotly(data, gender, age_range, user_step_count)