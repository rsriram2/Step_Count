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
- Provide a comprehensive **Methods** section embedding the full research paper detailing data sources, preprocessing, and analysis.
- Showcase the development team via an **About Us** section with member bios and links.

This tool helps researchers and public health practitioners explore activity patterns and demographic disparities in physical activity.

---

## Features

- **Interactive Filters:** Dynamically filter data by age, gender, and other demographic variables.
- **Distribution Plots:** Histograms and kernel density estimates (KDE) for step counts.
- **Normalized Curves:** KDE curves normalized to show percent of total steps, facilitating cross-group comparison.
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
├── utils/                 # Helper modules and utility functions
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
3. Switch to the **Methods** tab for detailed documentation and embedded research paper.
4. Visit the **About Us** tab for team member profiles and contact links.

---

## Methods

All methodological details—data description, preprocessing steps, statistical analyses, and validation—are accessible under the **Methods** tab within the app. The embedded paper includes figures and tables for deeper insight.

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
