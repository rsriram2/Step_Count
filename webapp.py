import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re

csvfile = 'steps_test.csv.gz'
data = pd.read_csv(csvfile, compression='gzip')

data['age_cat_display'] = data['age_cat'].str.extract(r'\[(\d+),(\d+)\)').apply(lambda x: f"{x[0]}-{int(x[1]) - 1}", axis=1)

def plot_histogram(data, gender, age_range, user_step_count=None):
    plt.figure(figsize=(8, 6))
    
    # Find the original age_cat corresponding to the display value
    original_age_range = data[data['age_cat_display'] == age_range]['age_cat'].iloc[0]
    
    if gender == "Overall":
        subset = data[data['age_cat'] == original_age_range]
    else:
        subset = data[(data['gender'] == gender) & (data['age_cat'] == original_age_range)]
    
    sns.kdeplot(subset['value'], fill=True, alpha=0.5, label=f'{gender} Density')
    
    if user_step_count is not None:
        quantile = subset[subset['value'] <= user_step_count]['q'].max()
        plt.axvline(user_step_count, color='r', linestyle='dashed', linewidth=2, label=f'Your Step Count (Percentile: {quantile * 100:.2f}%)')
    
    plt.xlim(0, max(data['value'].max(), 10000))
    plt.ylim(0, 0.0002)
    
    plt.xlabel("Step Count")
    plt.ylabel("Density")
    plt.title(f"Step Count Distribution for {gender} (Age Range {age_range})")
    plt.legend()
    st.pyplot(plt)

st.title("Step Count Distribution")

gender = st.selectbox(
    "Select Gender",
    options=["Overall", "Male", "Female"]
)

age_ranges_display = sorted(data['age_cat_display'].unique())

default_age_group = "20-30" if "20-30" in age_ranges_display else age_ranges_display[0]
age_range = st.selectbox("Select your age range", options=age_ranges_display, index=age_ranges_display.index(default_age_group))

original_age_range = data[data['age_cat_display'] == age_range]['age_cat'].iloc[0]
if gender == "Overall":
    subset = data[data['age_cat'] == original_age_range]
else:
    subset = data[(data['gender'] == gender) & (data['age_cat'] == original_age_range)]

median_step_count = subset['value'].median()
user_step_count = st.number_input("Enter your step count", min_value=0, value=int(median_step_count), step=500)

st.markdown(
    f"<h3 style='text-align: center; white-space: nowrap;'>Step Count Distribution for {gender} (Age Range {age_range})</h3>",
    unsafe_allow_html=True)
plot_histogram(data, gender, age_range, user_step_count)

if user_step_count:
    quantile = subset[subset['value'] <= user_step_count]['q'].max()
    steps_to_90th = round(max(0, subset[subset['q'] >= 0.9]['value'].min() - user_step_count))
    steps_to_95th = round(max(0, subset[subset['q'] >= 0.95]['value'].min() - user_step_count))
    steps_to_99th = round(max(0, subset[subset['q'] >= 0.99]['value'].min() - user_step_count))
    
    if pd.isna(quantile):
        st.write("### Based on the information you entered, your step count is below the minimum threshold for the selected demographic.")
    else:
        percentile = round(quantile * 100, 2)
        st.write(f"### Based on the information you entered, you are in the {percentile:.2f}th percentile.")
    
    st.write(f"### You need {steps_to_90th} more steps to reach the 90th percentile.")
    st.write(f"### You need {steps_to_95th} more steps to reach the 95th percentile.")
    st.write(f"### You need {steps_to_99th} more steps to reach the 99th percentile.")