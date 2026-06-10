# Predictive Analytics for Incident Resolution Time

## Project Overview

This project analyzes IT service management incident records and builds predictive models to estimate incident resolution time. The goal is to understand which operational factors influence how long an incident takes to resolve and to support faster, more data-driven incident management decisions.

The project uses R for data cleaning, feature engineering, exploratory data analysis, regression modeling, classification modeling, and model evaluation.

## Objective

The main objective is to predict incident resolution time using historical incident event log data.

Key questions explored in this project include:

- How long do incidents typically take to resolve?
- Which categories, priorities, impacts, and assignment groups are associated with longer resolution times?
- Can machine learning models predict resolution time accurately?
- Can incidents be classified as normal-resolution or long-resolution cases?
- Which features are most important in predicting incident resolution time?

## Modular R Code Structure

The project is organized into reusable R modules for readability and maintainability.

```text
Predictive-Analytics-for-Incident-Resolution-Time/
│
├── data/
│   └── incident_event_log.csv
│
├── R/
│   ├── 00_libraries.R
│   ├── 01_data_import_cleaning.R
│   ├── 02_eda_functions.R
│   ├── 03_feature_engineering.R
│   ├── 04_model_preparation.R
│   ├── 05_classification_models.R
│   └── 06_model_evaluation.R
├── main.R
├── requirements.txt
├── README.md
├── ABOUT_GITHUB.txt
└── .gitignore
```

## Author

**Girish S Chandrappa**
