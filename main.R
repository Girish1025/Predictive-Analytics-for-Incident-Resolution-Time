# main.R
# Main workflow for Predictive Analytics for Incident Resolution Time.

source("R/00_libraries.R")
source("R/01_data_import_cleaning.R")
source("R/02_eda_functions.R")
source("R/03_feature_engineering.R")
source("R/04_model_preparation.R")
source("R/06_classification_models.R")
source("R/07_model_evaluation.R")

dir.create("outputs/plots", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/results", recursive = TRUE, showWarnings = FALSE)

# 1. Load and clean data
incident_raw <- load_incident_data("data/incident_event_log.csv")
incident_clean <- clean_incident_data(incident_raw)

# 2. Feature engineering
incident_features <- engineer_incident_features(incident_clean)

# 3. Exploratory analysis examples
print(summary(incident_features$resolution_time_hours))
print(table(incident_features$time_group))

resolution_distribution_plot <- plot_resolution_distribution(incident_features)
log_resolution_distribution_plot <- plot_log_resolution_distribution(incident_features)
ggsave("outputs/plots/resolution_distribution.png", resolution_distribution_plot, width = 9, height = 6)
ggsave("outputs/plots/log_resolution_distribution.png", log_resolution_distribution_plot, width = 9, height = 6)

if ("priority" %in% names(incident_features)) {
  priority_summary <- summarize_resolution_by_group(incident_features, "priority")
  print(priority_summary)
  write.csv(priority_summary, "outputs/results/priority_resolution_summary.csv", row.names = FALSE)

  priority_resolution_plot <- plot_resolution_by_group(incident_features, "priority")
  ggsave("outputs/plots/priority_resolution.png", priority_resolution_plot, width = 9, height = 6)
}

# Correlation and categorical-association analysis 
eda_numeric_data <- incident_features %>%
  select(any_of(c("reassignment_count", "reopen_count", "sys_mod_count", "time_taken")))
correlation_matrix <- build_numeric_correlation(eda_numeric_data)
write.csv(correlation_matrix, "outputs/results/numeric_correlation_matrix.csv")

correlation_plot <- plot_correlation_heatmap(correlation_matrix)
ggsave("outputs/plots/numeric_correlation_heatmap.png", correlation_plot, width = 9, height = 7)

eda_categorical_cols <- intersect(
  c(
    "location", "category", "subcategory", "u_symptom", "impact", "urgency",
    "priority", "assignment_group", "knowledge", "u_priority_confirmation"
  ),
  names(incident_features)
)
cramers_v_results <- calculate_cramers_v(incident_features, eda_categorical_cols)
write.csv(cramers_v_results, "outputs/results/cramers_v_associations.csv", row.names = FALSE)

cramers_v_plot <- plot_cramers_v(cramers_v_results)
ggsave("outputs/plots/cramers_v_associations.png", cramers_v_plot, width = 10, height = 8)

# 4. Classification modeling: predict incident resolution-time group
classification_data <- prepare_classification_data(incident_features)

target_before_smote_plot <- plot_target_distribution(
  classification_data$train,
  "target",
  "Target Distribution Before SMOTE"
)
ggsave("outputs/plots/target_distribution_before_smote.png", target_before_smote_plot, width = 8, height = 6)

classification_models <- train_classification_models(classification_data$train)

target_after_smote_plot <- plot_target_distribution(
  classification_models$smoted_training_data,
  "target",
  "Target Distribution After SMOTE"
)
ggsave("outputs/plots/target_distribution_after_smote.png", target_after_smote_plot, width = 8, height = 6)

classification_results <- evaluate_classification_models(classification_models, classification_data$test)
print(classification_results)
write.csv(classification_results, "outputs/results/classification_results.csv", row.names = FALSE)

classification_comparison_plot <- plot_model_comparison(classification_results, "Accuracy")
ggsave("outputs/plots/classification_model_comparison.png", classification_comparison_plot, width = 9, height = 6)

# 5. Model diagnostics 
all_metrics_plot <- plot_all_classification_metrics(classification_results)
ggsave("outputs/plots/classification_all_metrics_comparison.png", all_metrics_plot, width = 10, height = 7)

diagnostics <- build_classification_diagnostics(classification_models, classification_data$test)
for (model_name in names(diagnostics)) {
  confusion_plot <- plot_confusion_matrix(diagnostics[[model_name]], model_name)
  ggsave(
    paste0("outputs/plots/confusion_matrix_", model_name, ".png"),
    confusion_plot,
    width = 8,
    height = 6
  )

  roc_plot <- plot_multiclass_roc(diagnostics[[model_name]], model_name)
  ggsave(
    paste0("outputs/plots/roc_curves_", model_name, ".png"),
    roc_plot,
    width = 9,
    height = 7
  )
}

c50_importance_plot <- plot_c50_importance(classification_models$c50)
ggsave("outputs/plots/c50_importance.png", c50_importance_plot, width = 9, height = 7)

rf_importance_plot <- plot_random_forest_importance(classification_models$random_forest)
ggsave("outputs/plots/random_forest_importance.png", rf_importance_plot, width = 9, height = 6)
