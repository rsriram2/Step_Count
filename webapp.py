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

tab1, tab2 = st.tabs(["Step Count Distribution", "About Us"])

with tab1:
    # Load data
    csvfile = 'utils/steps_test.csv.gz'
    data = pd.read_csv(csvfile, compression='gzip')

    # Reformat age category
    data['age_cat_display'] = data['age_cat'].str.extract(r'\[(\d+),(\d+)\)').apply(lambda x: f"{x[0]}-{int(x[1]) - 1}", axis=1)

    # KDE plot
    def plot_kde_plotly(data, gender, age_range, user_step_count=None):
        original_age_range = data[data['age_cat_display'] == age_range]['age_cat'].iloc[0]
        subset = data if gender == "Overall" else data[data['gender'] == gender]
        subset = subset[subset['age_cat'] == original_age_range]

        values = subset['value'].dropna()
        kde = gaussian_kde(values)
        x_range = np.linspace(0, max(values.max(), 35000), 500)
        y_range = kde(x_range)
        
        area = np.trapz(y_range, x_range)
        y_range = y_range * (100 / area)


        fig = go.Figure()
        
        raw_pct = np.array([
            subset[subset['value'] <= xi]['q'].max() * 100
            for xi in x_range
        ])

        # 2) cap *and* build hover-labels
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
            line=dict(color='steelblue'),
            name=f"{gender} Distribution",
            text=hover_labels,   # <-- use text instead of customdata
            hovertemplate=(
                "Step Count: %{x}<br>"
                "Percentile: %{text}<extra></extra>"
            )
        ))
        
        if user_step_count is not None:
            qt = subset[subset['value'] <= user_step_count]['q'].max() * 100
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
        
        original_age_range = data[data['age_cat_display'] == age_range]['age_cat'].iloc[0]
        subset = data[data['age_cat'] == original_age_range] if gender == "Overall" else data[(data['gender'] == gender) & (data['age_cat'] == original_age_range)]
        median_step_count = subset['value'].median()
        # 1) Compute the rounded median for the current demo
        median_val = int(round(median_step_count, -3))

        # 2) Initialize session‐state on first run
        if "last_median" not in st.session_state:
            st.session_state["last_median"] = median_val
        if "user_step_count" not in st.session_state:
            st.session_state["user_step_count"] = median_val
        if "step_changed" not in st.session_state:
            st.session_state["step_changed"] = False

        # 3) Only update user_step_count to the new median if they never changed it
        if not st.session_state.step_changed:
            st.session_state.user_step_count = median_val

        # 4) Record this median for the next rerun
        st.session_state.last_median = median_val

        # 5) Define a callback that marks “they’ve changed the step count”
        def mark_changed():
            st.session_state.step_changed = True

        # 6) Render the number_input with on_change
        user_step_count = st.number_input(
            "Step Count",
            min_value=0,
            step=500,
            key="user_step_count",
            on_change=mark_changed
        )

        if user_step_count:
            quantile = subset[subset['value'] <= user_step_count]['q'].max()
            if pd.isna(quantile):
                st.write("### Your step count is below the threshold for this demographic.")
            else:
                percentile = round(quantile * 100, 2)
                if percentile >= 100:
                    display_pct = "99+"
                else:
                    display_pct = f"{percentile:.2f}th"
                steps_to_90th = round(max(0, subset[subset['q'] >= 0.9]['value'].min() - user_step_count))
                steps_to_95th = round(max(0, subset[subset['q'] >= 0.95]['value'].min() - user_step_count))
                steps_to_99th = round(max(0, subset[subset['q'] >= 0.99]['value'].min() - user_step_count))

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
                
                st.markdown(f"""
                    <div style="background-color: #1f2937; padding:20px; border-radius:10px; margin-top:20px;">
                        <h4 style="color:#ffffff;"> Based on your information:</h4>
                        <p style="color:#93c5fd; font-size:16px; margin-bottom: 10px;">
                            You are in the <strong style="color:#60a5fa;">{display_pct} percentile</strong>.
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
        plot_kde_plotly(data, gender, age_range, user_step_count)

    st.divider()

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# helper that returns an HTML “card” with a square-cropped <img> + caption
def profile_card(img_b64: str, caption_html: str, obj_pos: str="50% 20%") -> str:
    return f"""
    <div style="text-align:center; margin-bottom:2rem;">
      <div style="display:inline-block; width:200px; text-align:center;">
        <!-- square container crops via overflow:hidden -->
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
    
    st.header("Methodology")
    st.info(
        """
        Step count estimates were obtained by applying a machine-learning based 
        [step-count algorithm](https://journals.lww.com/acsm-msse/fulltext/2024/10000/self_supervised_machine_learning_to_characterize.9.aspx) to raw accelerometry data from the National Health and Nutrition Examination Survey (NHANES). Survey weights provided by NHANES were used to adjust estimates to be nationally representative. See more details on the methodology [here](https://pubmed.ncbi.nlm.nih.gov/39589008/). """
    )
    
    st.divider()

    # Encode each headshot
    b64_1 = img_to_b64("utils/rushil_headshot.png")
    b64_2 = img_to_b64("utils/john_headshot.jpeg")
    b64_3 = img_to_b64("utils/LK_Headshot.JPG")
    b64_4 = img_to_b64("utils/ciprian_headshot.jpg")


    # Create a centered layout for the team cards
    st.markdown(
        '<h2 style="text-align:center; color:white; margin-bottom:1rem; padding-left:27px;">Meet the Team</h2>',
        unsafe_allow_html=True
    )

    # Create a flexbox layout for centering
    team_cards = f"""
    <div style="display: flex; justify-content: space-between; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
        {profile_card(b64_1, '<strong>Rushil Srirambhatla</strong>: B.S. in Applied Mathematics & Statistics at Johns Hopkins University | '
            '<a href="https://github.com/rsriram2" target="_blank" style="color:#3ea6ff;">GitHub</a>', obj_pos="50% 20%")}
        {profile_card(b64_2, '<strong>John Muschelli</strong>: Associate Research Professor at Johns Hopkins Bloomberg School of Public Health' , obj_pos="50% 80%")}
        {profile_card(b64_3, '<strong>Lily Koffman</strong>: Biostatistics PhD Candidate at Johns Hopkins Bloomberg School of Public Health | '
            '<a href="https://lilykoff.com" target="_blank" style="color:#3ea6ff;">Website</a>', obj_pos="70% 2%")}
        {profile_card(b64_4, '<strong>Ciprian Crainiceanu</strong>: Professor of Biostatistics at Johns Hopkins Bloomberg School of Public Health')}
    
    </div>
    """

    st.markdown(team_cards, unsafe_allow_html=True)

    st.divider()

    # Footer full-width below
    st.markdown(
        '<p style="text-align:center; color:#e5e7eb; margin-top:2rem; font-weight:bold; font-size:25px">'
        'We hope you find this tool useful for exploring step count data!'
        '</p>',
        unsafe_allow_html=True
    )