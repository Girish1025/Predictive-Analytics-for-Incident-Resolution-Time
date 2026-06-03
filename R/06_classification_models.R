# R/06_classification_models.R
# Classification models for predicting long vs normal resolution time.

train_classification_models <- function(train_data) {
  train_data$target <- as.factor(train_data$target)
  x_train <- train_data %>% select(-target)
  y_train <- train_data$target

  tree_model <- rpart::rpart(target ~ ., data = train_data, method = "class")

  rf_model <- randomForest::randomForest(
    x = x_train,
    y = y_train,
    ntree = 300,
    importance = TRUE
  )

  xgb_y <- ifelse(y_train == levels(y_train)[2], 1, 0)
  xgb_train <- xgboost::xgb.DMatrix(data = as.matrix(x_train), label = xgb_y)
  xgb_model <- xgboost::xgboost(
    data = xgb_train,
    objective = "binary:logistic",
    eval_metric = "auc",
    nrounds = 150,
    max_depth = 5,
    eta = 0.08,
    verbose = 0
  )

  list(
    decision_tree = tree_model,
    random_forest = rf_model,
    xgboost = xgb_model
  )
}

predict_classification_model <- function(model, model_name, test_data) {
  x_test <- test_data %>% select(-target)

  if (model_name == "xgboost") {
    probabilities <- predict(model, as.matrix(x_test))
    predictions <- ifelse(probabilities >= 0.5, levels(test_data$target)[2], levels(test_data$target)[1])
    predictions <- factor(predictions, levels = levels(test_data$target))
  } else {
    predictions <- predict(model, newdata = test_data, type = "class")
  }

  predictions
}
