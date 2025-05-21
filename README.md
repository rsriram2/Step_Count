# Step_Count

An interactive web application for visualizing and analyzing step-count data across demographic groups.

---

## Table of Contents

1. [About the Project](#about-the-project)
2. [Features](#features)
3. [Demo](#demo)
4. [Tech Stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Running the App](#running-the-app)
8. [Project Structure](#project-structure)
9. [Configuration](#configuration)
10. [Usage](#usage)
11. [Methods](#methods)
12. [About Us](#about-us)
13. [Contact](#contact)

---

## About the Project

`Step_Count` is a Streamlit-based application designed to:

- Present **interactive visualizations** of daily step-count distributions by demographic factors.
- Compare groups using **normalized KDE curves** (percent of total steps on the y-axis).
- **Estimate survival benefit** for different step counts and demographics, based on published survival models.
- Provide a comprehensive **Methods** section embedding the full research paper detailing data sources, preprocessing, and analysis.
- Showcase the development team via an **About Us** section with member bios and links.

This tool helps researchers and public health practitioners explore activity patterns and demographic disparities in physical activity.

---

## Features

- **Interactive Filters:** Dynamically filter data by age, gender, and other demographic variables.
- **Distribution Plots:** Histograms and kernel density estimates (KDE) for step counts.
- **Normalized Curves:** KDE curves normalized to show percent of total steps, facilitating cross-group comparison.
- **Survival Probability Estimates:** Quantify estimated improvements in all-cause survival probability for a 500-step increase, stratified by age, gender, and algorithm. (Available for users aged 50+.)
- **Integrated Documentation:** A dedicated Methods tab with embedded research paper content.
- **Team Profiles:** About Us page with team roles, photos, and external links.

---

## Demo

*View the live app:* https://stepcount-7kh4v28xvprobjpp9kxyzz.streamlit.app/?embed_options=dark_theme

---

## Tech Stack

- **Language:** Python 3.8+
- **Framework:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly

---

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

---

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/rsriram2/Step_Count.git
    cd Step_Count
    ```

2. **Create a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## Running the App

Launch the Streamlit application:

```bash
streamlit run webapp.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to explore the dashboard.

---

## Project Structure

```
Step_Count/
├── .devcontainer/         # Development container settings for VS Code
├── .streamlit/            # Streamlit configuration (theme, page title)
├── misc/                  # Static assets (screenshots, logos, sample data)
├── utils/                 # Helper modules and utility functions, data files (including survival improvement data)
├── webapp.py              # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Project overview and instructions
```

---

## Configuration

- `.streamlit/`: Customize page title, icon, and theme settings.
- `.devcontainer/`: Defines the development container for consistent environment setup.

---

## Usage

After launching the app:

1. Use the **Filters** sidebar to select demographic subsets.
2. View **Distribution** and **KDE** plots updating in real time.
3. **Find your percentile and survival probability improvement:**  
   - Enter your step count and demographics.
   - If eligible (age ≥ 50, gender specified), view the estimated improvement in all-cause survival probability for a 500-step increase, based on published models.
4. Switch to the **Methods** tab for detailed documentation and embedded research paper.
5. Visit the **About Us** tab for team member profiles and contact links.

---

## Survival Estimate Implementation

The app provides estimated improvements in all-cause survival probability for a 500-step increase, stratified by age group, gender, and step count algorithm. These survival estimates are shown for users aged 50 and above with specified gender, based on models from [peer-reviewed studies](https://journals.lww.com/acsm-msse/fulltext/2024/10000/self_supervised_machine_learning_to_characterize.9.aspx).

**Implementation details:**
- Survival improvement data are precomputed and stored in `utils/steps_survival_all_algorithms_inc10.csv.gz`.
- In the app, after the user enters their demographics and step count, the relevant survival probability change is displayed (if available) in the right panel.
- The survival panel is shown only for eligible age/gender combinations; otherwise, a notice is shown.
- Survival estimates are stratified by age brackets (e.g., 50–59, 60–69), gender, and algorithm.

---

## Methods

All methodological details—data description, preprocessing steps, statistical analyses, and validation—are accessible under the **Methods** tab within the app. The embedded paper includes figures and full details.

---

## About Us

- Rushil Srirambhatla: B.S. Applied Mathematics & Statistics at Johns Hopkins University
- John Muschelli: Associate Research Professor at Johns Hopkins Bloomberg School of Public Health
- Lily Koffman: Biostatistics PhD Candidate at Johns Hopkins Bloomberg School of Public Health
- Ciprian Crainiceanu: Professor of Biostatistics at Johns Hopkins Bloomberg School of Public Health

---

## Contact

- **Project Lead:** Rushil Srirambhatla ([GitHub](https://github.com/rsriram2))
- **Email:** [rushil@example.com](mailto:rsriram2@jh.edu)

Feel free to open issues or reach out with questions or suggestions!
