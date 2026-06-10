# R/07_model_evaluation.R
# Model evaluation and comparison functions.

evaluate_regression_models <- function(models, test_data) {
  actual <- test_data$target

  results <- lapply(names(models), function(model_name) {
    preds <- predict_regression_model(models[[model_name]], model_name, test_data)

    rmse_value <- sqrt(mean((actual - preds)^2, na.rm = TRUE))
    mae_value <- mean(abs(actual - preds), na.rm = TRUE)
    r2_value <- cor(actual, preds, use = "complete.obs")^2

    data.frame(
      Model = model_name,
      RMSE = rmse_value,
      MAE = mae_value,
      R_Squared = r2_value
    )
  })

  dplyr::bind_rows(results) %>% arrange(RMSE)
}

evaluate_classification_models <- function(models, test_data) {
  test_data$target <- as.factor(test_data$target)
  class_levels <- models$class_levels

  model_names <- setdiff(names(models), c("class_levels", "smoted_training_data"))

  results <- lapply(model_names, function(model_name) {
    prediction_result <- predict_classification_model(
      models[[model_name]],
      model_name,
      test_data,
      class_levels
    )

    eval_data <- tibble::tibble(
      truth = factor(test_data$target, levels = class_levels),
      prediction = factor(prediction_result$predictions, levels = class_levels)
    )

    data.frame(
      Model = model_name,
      Accuracy = yardstick::accuracy(eval_data, truth, prediction)$.estimate,
      Precision = yardstick::precision(eval_data, truth, prediction, estimator = "macro")$.estimate,
      Recall = yardstick::recall(eval_data, truth, prediction, estimator = "macro")$.estimate,
      F1 = yardstick::f_meas(eval_data, truth, prediction, estimator = "macro")$.estimate
    )
  })

  dplyr::bind_rows(results) %>% arrange(desc(Accuracy))
}

plot_model_comparison <- function(results_df, metric = "RMSE") {
  ggplot(results_df, aes(x = reorder(Model, .data[[metric]]), y = .data[[metric]])) +
    geom_col(fill = "steelblue", alpha = 0.75) +
    coord_flip() +
    labs(
      title = paste("Model Comparison by", metric),
      x = "Model",
      y = metric
    ) +
    theme_minimal()
}

plot_all_classification_metrics <- function(results_df) {
  results_df %>%
    tidyr::pivot_longer(
      cols = c(Accuracy, Precision, Recall, F1),
      names_to = "Metric",
      values_to = "Score"
    ) %>%
    ggplot(aes(x = Model, y = Score, fill = Metric)) +
    geom_col(position = "dodge") +
    labs(title = "Classification Model Performance", x = NULL, y = "Score") +
    theme_minimal()
}

build_classification_diagnostics <- function(models, test_data) {
  model_names <- setdiff(names(models), c("class_levels", "smoted_training_data"))

  lapply(stats::setNames(model_names, model_names), function(model_name) {
    prediction_result <- predict_classification_model(
      models[[model_name]],
      model_name,
      test_data,
      models$class_levels
    )

    list(
      truth = factor(test_data$target, levels = models$class_levels),
      predictions = prediction_result$predictions,
      probabilities = prediction_result$probabilities
    )
  })
}

plot_confusion_matrix <- function(diagnostic, model_name) {
  confusion_df <- as.data.frame(table(
    Actual = diagnostic$truth,
    Predicted = diagnostic$predictions
  ))

  ggplot(confusion_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 4) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = paste("Confusion Matrix:", model_name), x = "Actual", y = "Predicted") +
    theme_minimal()
}

build_multiclass_roc_data <- function(diagnostic) {
  class_levels <- levels(diagnostic$truth)

  dplyr::bind_rows(lapply(class_levels, function(class_name) {
    roc_input <- data.frame(
      truth = factor(
        ifelse(diagnostic$truth == class_name, "event", "other"),
        levels = c("event", "other")
      ),
      probability = diagnostic$probabilities[, class_name]
    )

    yardstick::roc_curve(roc_input, truth, probability) %>%
      mutate(Class = class_name)
  }))
}

plot_multiclass_roc <- function(diagnostic, model_name) {
  roc_data <- build_multiclass_roc_data(diagnostic)

  ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = Class)) +
    geom_path(linewidth = 1) +
    geom_abline(linetype = "dashed", color = "gray50") +
    coord_equal() +
    labs(title = paste("One-vs-Rest ROC Curves:", model_name), x = "1 - Specificity", y = "Sensitivity") +
    theme_minimal()
}

plot_c50_importance <- function(model, top_n = 20) {
  importance_df <- as.data.frame(C50::C5imp(model)) %>%
    tibble::rownames_to_column("Feature") %>%
    arrange(desc(Overall)) %>%
    head(top_n)

  ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
    geom_col(fill = "steelblue", alpha = 0.8) +
    coord_flip() +
    labs(title = "C5.0 Feature Importance", x = "Feature", y = "Overall Importance") +
    theme_minimal()
}

plot_random_forest_importance <- function(model, top_n = 20) {
  importance_df <- as.data.frame(randomForest::importance(model))
  importance_df$Feature <- rownames(importance_df)

  metric_col <- if ("IncNodePurity" %in% names(importance_df)) "IncNodePurity" else names(importance_df)[1]

  importance_df %>%
    arrange(desc(.data[[metric_col]])) %>%
    head(top_n) %>%
    ggplot(aes(x = reorder(Feature, .data[[metric_col]]), y = .data[[metric_col]])) +
    geom_col(fill = "steelblue", alpha = 0.75) +
    coord_flip() +
    labs(
      title = "Top Random Forest Feature Importance",
      x = "Feature",
      y = metric_col
    ) +
    theme_minimal()
}
