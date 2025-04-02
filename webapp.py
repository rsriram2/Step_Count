import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

csvfile = '/Users/rushil/PH_Website/steps_test.csv.gz'
data = pd.read_csv(csvfile, compression='gzip')
subset_data = data.iloc[::100]
st.write(subset_data)