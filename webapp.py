import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from PIL import Image, ImageOps
import base64

# Config page
st.set_page_config(page_title="Step Count Distribution", layout="wide")

# Custom dark mode CSS
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #d1d5db;
    }
    .reportview-container, .main {
        background-color: #0e1117;
        color: #d1d5db;
    }
    h1, h2, h3, h4 {
        color: white;
    }
    </style>
    <h1 style='text-align: center; padding: 0.5rem; color: white; background-color: #1f2937; border-radius: 10px;'>
        Step Count Distribution
    </h1>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Step Count Distribution", "Algorithm Comparison", "Methods", "About Us"])

with tab1:
    # Load data
    csvfile = 'utils/steps_all_algorithms.csv.gz'
    data = pd.read_csv(csvfile, compression='gzip')

    # Reformat age category
    data['age_cat_display'] = data['age_cat'].str.extract(r'\[(\d+),(\d+)\)').apply(lambda x: f"{x[0]}-{int(x[1]) - 1}", axis=1)

    # Function: KDE plot for a single algorithm (with percentiles, medians, red dotted line)
    def plot_single_algo_kde(data, gender, age_range, user_step_count=None, percentile_value=None):
        algo = "Stepcount SSL"
        subset = data[data['name'] == algo]
        subset_age = subset[subset['age_cat_display'] == age_range]
        if subset_age.empty:
            st.warning("No data for selected demographic.")
            return
        original_age_range = subset_age['age_cat'].iloc[0]
        sub = subset if gender == "Overall" else subset[subset['gender'] == gender]
        sub = sub[sub['age_cat'] == original_age_range]

        if sub.empty:
            st.warning("No data for selected demographic.")
            return

        values = sub['value'].dropna()
        kde = gaussian_kde(values)
        x_range = np.linspace(0, max(values.max(), 35000), 500)
        y_range = kde(x_range)
        area = np.trapz(y_range, x_range)
        y_range = y_range * (100 / area)

        # Percentiles for hover
        raw_pct = np.array([
            sub[sub['value'] <= xi]['q'].max() * 100 for xi in x_range
        ])
        capped = np.minimum(raw_pct, 99)
        hover_labels = [
            f"{p:.2f}%" if raw_pct[i] < 100 else "99+%"
            for i, p in enumerate(capped)
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#60a5fa'),
            name=f"Step Count Distribution",
            text=hover_labels,
            hovertemplate=(
                "Step Count: %{x}<br>"
                "Percentile: %{text}<extra></extra>"
            )
        ))

        # User step count marker
        if user_step_count is not None:
            qt = sub[sub['value'] <= user_step_count]['q'].max() * 100
            label = f"{qt:.2f}%" if qt < 100 else "99+%"
            fig.add_trace(go.Scatter(
                x=[user_step_count, user_step_count],
                y=[0, max(y_range)],
                mode='lines',
                line=dict(color='crimson', dash='dash'),
                name=f"Your Step Count ({label})",
                hovertemplate="Step Count: %{x}<br>Percentile: "+label+"<extra></extra>"
            ))

        fig.update_layout(
            title_text="",
            template="plotly_dark",
            margin=dict(l=30, r=30, t=10, b=40),
            xaxis_title="Step Count",
            yaxis_title="Percent per step count", 
            xaxis_title_font=dict(size=14, color='white', family='Arial Black'),
            yaxis_title_font=dict(size=14, color='white', family='Arial Black'),
            title_x=0.5,
            legend=dict(
                x=0.75, y=0.95,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                font=dict(size=13, color="white")
            )
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", tickformat=".0f")
        fig.update_yaxes(tickformat=".3f")
        st.plotly_chart(fig, use_container_width=True)

    # UI layout
    age_ranges_display = sorted(data['age_cat_display'].unique())
    default_age_group = "20-29" if "20-29" in age_ranges_display else age_ranges_display[0]

    left_col, spacer, right_col = st.columns([2, 0.3, 3])

    with left_col:
        st.subheader("Your Information")
        gender = st.selectbox("Gender", ["Overall", "Male", "Female"])
        age_range = st.selectbox("Age Range", options=age_ranges_display, index=age_ranges_display.index(default_age_group))

        rf_subset = data[(data['name'] == "Stepcount SSL") & (data['age_cat_display'] == age_range)]
        if gender != "Overall":
            rf_subset = rf_subset[rf_subset['gender'] == gender]
        rf_subset = rf_subset.copy()

        median_step_count = rf_subset['value'].median()
        median_val = int(round(median_step_count, -3)) if not np.isnan(median_step_count) else 0

        # Session state initialization
        if "last_median" not in st.session_state:
            st.session_state["last_median"] = median_val
        if "user_step_count" not in st.session_state:
            st.session_state["user_step_count"] = median_val
        if "step_changed" not in st.session_state:
            st.session_state["step_changed"] = False

        if not st.session_state.step_changed:
            st.session_state.user_step_count = median_val
        st.session_state.last_median = median_val

        def mark_changed():
            st.session_state.step_changed = True

        user_step_count = st.number_input(
            "Step Count",
            min_value=0,
            step=500,
            key="user_step_count",
            on_change=mark_changed
        )

        quantile = rf_subset[rf_subset['value'] <= user_step_count]['q'].max()
        if pd.isna(quantile):
            percentile = None
            st.write("### Your step count is below the threshold for this demographic (based on Stepcount SSL).")
        else:
            percentile = round(quantile * 100, 2)
            steps_to_90th = round(max(0, rf_subset[rf_subset['q'] >= 0.9]['value'].min() - user_step_count))
            steps_to_95th = round(max(0, rf_subset[rf_subset['q'] >= 0.95]['value'].min() - user_step_count))
            steps_to_99th = round(max(0, rf_subset[rf_subset['q'] >= 0.99]['value'].min() - user_step_count))

            insight_lines = []
            if percentile < 90:
                insight_lines.append(f"<li> You need <strong><u>{steps_to_90th}</u></strong> more steps to reach the <strong><u>90th</u></strong> percentile.</li>")
            if percentile < 95:
                insight_lines.append(f"<li> You need <strong><u>{steps_to_95th}</u></strong> more steps to reach the <strong><u>95th</u></strong> percentile.</li>")
            if percentile < 99:
                insight_lines.append(f"<li> You need <strong><u>{steps_to_99th}</u></strong> more steps to reach the <strong><u>99th</u></strong> percentile.</li>")

            ul_block = f"""
                <ul style="color:#e5e7eb; font-size:15px; line-height:1.6;">
                    {''.join(insight_lines)}
                </ul>
            """ if insight_lines else ""

            display_pct = "99+" if percentile >= 100 else f"{percentile:.2f}th"
            st.markdown(f"""
                <div style="background-color: #1f2937; padding:20px; border-radius:10px; margin-top:20px;">
                    <h4 style="color:#ffffff;"> Based on your information:</h4>
                    <p style="color:#93c5fd; font-size:16px; margin-bottom: 10px;">
                        You are in the <strong style="color:#60a5fa;">{display_pct} percentile</strong> (based on Stepcount SSL).
                    </p>
                    {ul_block}
                </div>
            """, unsafe_allow_html=True)

    with right_col:
        st.markdown(
            f"<div style='text-align: center; font-size: 24px; font-weight: bold; margin-top: 4px; color: white;'>"
            f"Step Count Distribution: {gender} (Age {age_range})"
            f"</div>",
            unsafe_allow_html=True
        )
        plot_single_algo_kde(
            data,
            gender,
            age_range,
            user_step_count=user_step_count,
            percentile_value=percentile
        )
    st.divider()

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def profile_card(img_b64: str, caption_html: str, obj_pos: str="50% 20%") -> str:
    return f"""
    <div style="text-align:center; margin-bottom:2rem;">
      <div style="display:inline-block; width:200px; text-align:center;">
        <div style="width:200px; height:200px; overflow:hidden; border-radius:8px;">
          <img
            src="data:image/jpeg;base64,{img_b64}"
            style="
              width:100%;
              height:100%;
              object-fit:cover;
              object-position: {obj_pos};
              display:block;
            "
          />
        </div>
        <p style="color:#e5e7eb; margin:0.5rem 0 0 0; line-height:1.3; font-size:14px;">
          {caption_html}
        </p>
      </div>
    </div>
    """

with tab2:
    algo_options = sorted([n for n in data['name'].unique() if n != "Best guess"])
    default_algos = ["Stepcount RF", "Stepcount SSL"] if all(a in algo_options for a in ["Stepcount RF", "Stepcount SSL"]) else algo_options[:2]

    st.subheader("Algorithm Comparison")
    left_col, right_col = st.columns([2, 2.8], gap="large")

    with left_col:
        selected_algos = st.multiselect(
            "Algorithms to Compare",
            options=algo_options,
            default=default_algos,
            help="Overlay step count distributions for multiple algorithms."
        )
        gender = st.selectbox("Gender", ["Overall", "Male", "Female"], key="comp_gender")
        age_range = st.selectbox("Age Range", options=age_ranges_display, index=age_ranges_display.index(default_age_group), key="comp_age")
        user_step_count = st.number_input("Step Count", value=9000, min_value=0, step=500, key="comp_user_step_count")

    with right_col:
        st.markdown("#### Compare with Step-Detection Algorithms")
        def plot_multi_algo_kde(data, algorithms, gender, age_range, user_step_count=None):
            color_cycle = [
                "#60a5fa", "#f59e42", "#10b981", "#e11d48", "#a855f7", "#facc15", "#64748b", "#f472b6"
            ]
            color_map = {algo: color_cycle[i % len(color_cycle)] for i, algo in enumerate(algorithms)}
            fig = go.Figure()
            max_peak = 0
            for algo in algorithms:
                subset = data[data['name'] == algo]
                subset_age = subset[subset['age_cat_display'] == age_range]
                if subset_age.empty:
                    continue
                original_age_range = subset_age['age_cat'].iloc[0]
                sub = subset if gender == "Overall" else subset[subset['gender'] == gender]
                sub = sub[sub['age_cat'] == original_age_range]
                if sub.empty:
                    continue
                values = sub['value'].dropna()
                kde = gaussian_kde(values)
                x_range = np.linspace(0, max(values.max(), 35000), 500)
                y_range = kde(x_range)
                area = np.trapz(y_range, x_range)
                y_range = y_range * (100 / area)
                max_peak = max(max_peak, y_range.max())
                raw_pct = np.array([
                    sub[sub['value'] <= xi]['q'].max() * 100 for xi in x_range
                ])
                capped = np.minimum(raw_pct, 99)
                hover_labels = [
                    f"{p:.2f}%" if raw_pct[i] < 100 else "99+%"
                    for i, p in enumerate(capped)
                ]
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color=color_map[algo]),
                    name=f"{algo} Distribution",
                    text=hover_labels,
                    hovertemplate=(
                        "Step Count: %{x}<br>"
                        "Percentile: %{text}<extra></extra>"
                    )
                ))
            if user_step_count is not None and max_peak > 0:
                fig.add_trace(go.Scatter(
                    x=[user_step_count, user_step_count],
                    y=[0, max_peak],
                    mode='lines',
                    line=dict(color='crimson', dash='dash'),
                    name=f"Your Step Count",
                    hovertemplate="Step Count: %{x}<extra></extra>"
                ))
            fig.update_layout(
                title_text="",
                template="plotly_dark",
                margin=dict(l=30, r=30, t=10, b=40),
                xaxis_title="Step Count",
                yaxis_title="Percent per step count",
                xaxis_title_font=dict(size=14, color='white', family='Arial Black'),
                yaxis_title_font=dict(size=14, color='white', family='Arial Black'),
                title_x=0.5,
                legend=dict(
                    x=0.75, y=0.95, bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(255,255,255,0.2)", borderwidth=1,
                    font=dict(size=13, color="white")
                )
            )
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", tickformat=".0f")
            fig.update_yaxes(tickformat=".3f")
            st.plotly_chart(fig, use_container_width=True)
        plot_multi_algo_kde(data, selected_algos, gender, age_range, user_step_count=user_step_count)

with tab3:
    st.markdown(
        """
        <style>
          .stAlert p {
            font-size: 18px !important;
            font-weight: 600 !important;
            line-height: 1.5 !important;
            color: white !important;
          }
          .stAlert a {
            color: #3ea6ff !important;      
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Methodology")
    st.info(
        """
        Step count estimates were obtained by applying a machine-learning based 
        [step-count algorithm](https://journals.lww.com/acsm-msse/fulltext/2024/10000/self_supervised_machine_learning_to_characterize.9.aspx) to raw accelerometry data from the National Health and Nutrition Examination Survey (NHANES). Survey weights provided by NHANES were used to adjust estimates to be nationally representative. See more details on the methodology below. """
    )

    st.divider()
    pdf_path = "https://drive.google.com/file/d/1V-rkGjpVFvfDMoVWhiwkAj_RywtHh_28/preview"
    pdf_display = f"""
        <div style="display: flex; justify-content: center;">
            <iframe
                src="{pdf_path}"
                width="50%"
                height="900px"
                style="border: none; border-radius: 12px; background: #222;"
            ></iframe>
        </div>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

with tab4:
    st.markdown(
        """
        <style>
          .stAlert p {
            font-size: 18px !important;
            font-weight: 600 !important;
            line-height: 1.5 !important;
            color: white !important;
          }
          .stAlert a {
            color: #3ea6ff !important;      
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("About Us")
    st.info(
        "Welcome to the Step Count Distribution App! This app helps you visualize "
        "step count distributions based on demographic data. Use the controls on the "
        "main page to explore the data and gain insights."
    )
    st.divider()

    # Encode each headshot
    b64_1 = img_to_b64("utils/rushil_headshot.png")
    b64_2 = img_to_b64("utils/john_headshot.jpeg")
    b64_3 = img_to_b64("utils/LK_Headshot.JPG")
    b64_4 = img_to_b64("utils/ciprian_headshot.jpg")

    st.markdown(
        '<h2 style="text-align:center; color:white; margin-bottom:1rem; padding-left:27px;">Meet the Team</h2>',
        unsafe_allow_html=True
    )

    team_cards = f"""
    <div style="display: flex; justify-content: space-between; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
        {profile_card(b64_1, '<strong>Rushil Srirambhatla</strong>: B.S. Applied Mathematics & Statistics at Johns Hopkins University | '
            '<a href="https://github.com/rsriram2" target="_blank" style="color:#3ea6ff;">GitHub</a>', obj_pos="50% 20%")}
        {profile_card(b64_2, '<strong>John Muschelli</strong>: Associate Research Professor at Johns Hopkins Bloomberg School of Public Health | ' 
            '<a href="https://johnmuschelli.com/" target="_blank" style="color:#3ea6ff;">Website</a>', obj_pos="50% 80%")}
        {profile_card(b64_3, '<strong>Lily Koffman</strong>: Biostatistics PhD Candidate at Johns Hopkins Bloomberg School of Public Health | '
            '<a href="https://lilykoff.com" target="_blank" style="color:#3ea6ff;">Website</a>', obj_pos="70% 2%")}
        {profile_card(b64_4, '<strong>Ciprian Crainiceanu</strong>: Professor of Biostatistics at Johns Hopkins Bloomberg School of Public Health | '
            '<a href="http://www.ciprianstats.org/home" target="_blank" style="color:#3ea6ff;">Website</a>')}
    </div>
    """

    st.markdown(team_cards, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<p style="text-align:center; color:#e5e7eb; margin-top:2rem; font-weight:bold; font-size:25px">'
        'We hope you find this tool useful for exploring step count data!'
        '</p>',
        unsafe_allow_html=True
    )