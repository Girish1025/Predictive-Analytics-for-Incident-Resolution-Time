# R/06_classification_models.R
# Multi-class classification models following the original code.R workflow.

apply_smote_to_training_data <- function(train_data) {
  train_data$target <- as.factor(train_data$target)

  if (length(levels(train_data$target)) < 2) {
    return(train_data)
  }

  smote_recipe <- recipes::recipe(target ~ ., data = train_data) %>%
    themis::step_smote(target, over_ratio = 1)

  prep_smote <- recipes::prep(smote_recipe)
  recipes::juice(prep_smote)
}

train_classification_models <- function(train_data) {
  smoted_data <- apply_smote_to_training_data(train_data)
  smoted_data$target <- as.factor(smoted_data$target)

  class_levels <- levels(smoted_data$target)

  c50_model <- C50::C5.0(target ~ ., data = smoted_data)

  rf_model <- randomForest::randomForest(
    target ~ .,
    data = smoted_data,
    ntree = 500,
    mtry = floor(sqrt(ncol(smoted_data) - 1)),
    importance = TRUE
  )

  list(
    c50 = c50_model,
    random_forest = rf_model,
    class_levels = class_levels,
    smoted_training_data = smoted_data
  )
}

predict_classification_model <- function(model, model_name, test_data, class_levels) {
  x_test <- test_data %>% select(-target)

  predictions <- predict(model, newdata = x_test, type = "class")
  predictions <- factor(predictions, levels = class_levels)
  probabilities <- predict(model, newdata = x_test, type = "prob")
  probabilities <- as.matrix(probabilities[, class_levels, drop = FALSE])

  list(
    predictions = predictions,
    probabilities = probabilities
  )
}
