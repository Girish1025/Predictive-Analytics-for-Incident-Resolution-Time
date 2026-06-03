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

  results <- lapply(names(models), function(model_name) {
    preds <- predict_classification_model(models[[model_name]], model_name, test_data)
    cm <- caret::confusionMatrix(preds, test_data$target)

    data.frame(
      Model = model_name,
      Accuracy = as.numeric(cm$overall["Accuracy"]),
      Kappa = as.numeric(cm$overall["Kappa"]),
      Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
      Specificity = as.numeric(cm$byClass["Specificity"])
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
