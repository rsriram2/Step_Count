import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re

csvfile = '/Users/rushil/PH_Website/steps_test.csv.gz'
data = pd.read_csv(csvfile, compression='gzip')

def is_age_in_range(age, age_range):
    match = re.match(r'\[(\d+),(\d+)\)', age_range)
    if match:
        start_age, end_age = int(match.group(1)), int(match.group(2))
        return start_age <= age < end_age
    return False

def plot_histogram(data, gender, age, user_step_count=None):
    plt.figure(figsize=(8, 6))
    
    subset = data[(data['gender'] == gender) & (data['age_cat'].apply(lambda x: is_age_in_range(age, x)))]
    
    plt.hist(subset['value'], bins=30, alpha=0.5, density=True)
    
    if user_step_count is not None:
        quantile = subset[subset['value'] <= user_step_count]['q'].max()
        plt.axvline(user_step_count, color='r', linestyle='dashed', linewidth=2, label=f'Your Step Count (Quantile: {quantile:.2f})')
    
    plt.xlabel("Step Count")
    plt.ylabel("Density")
    plt.title(f"Step Count Distribution for {gender} (Age {age})")
    plt.legend()
    st.pyplot(plt)

st.title("Step Count Distribution")

gender = st.selectbox(
    "Select Gender",
    options=["Male", "Female"]
)

age = st.number_input("Enter your age", min_value=1, max_value=100)

user_step_count = st.number_input("Enter your step count", min_value=0)

st.write(f"## Step Count Distribution for {gender} (Age {age})")
plot_histogram(data, gender, age, user_step_count)