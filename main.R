# main.R
# Main workflow for Predictive Analytics for Incident Resolution Time.

source("R/00_libraries.R")
source("R/01_data_import_cleaning.R")
source("R/02_eda_functions.R")
source("R/03_feature_engineering.R")
source("R/04_model_preparation.R")
source("R/05_regression_models.R")
source("R/06_classification_models.R")
source("R/07_model_evaluation.R")

# 1. Load and clean data
incident_raw <- load_incident_data("data/incident_event_log.csv")
incident_clean <- clean_incident_data(incident_raw)

# 2. Feature engineering
incident_features <- engineer_incident_features(incident_clean)

# 3. Exploratory analysis examples
print(summary(incident_features$resolution_time_hours))
print(plot_resolution_distribution(incident_features))
print(plot_log_resolution_distribution(incident_features))

if ("priority" %in% names(incident_features)) {
  print(summarize_resolution_by_group(incident_features, "priority"))
  print(plot_resolution_by_group(incident_features, "priority"))
}

# 4. Regression modeling: predict log resolution time
regression_data <- prepare_regression_data(incident_features)
regression_models <- train_regression_models(regression_data$train)
regression_results <- evaluate_regression_models(regression_models, regression_data$test)
print(regression_results)
print(plot_model_comparison(regression_results, "RMSE"))

# 5. Classification modeling: predict long vs normal resolution time
classification_data <- prepare_classification_data(incident_features)
classification_models <- train_classification_models(classification_data$train)
classification_results <- evaluate_classification_models(classification_models, classification_data$test)
print(classification_results)
print(plot_model_comparison(classification_results, "Accuracy"))

# 6. Feature importance example
print(plot_random_forest_importance(regression_models$random_forest))
