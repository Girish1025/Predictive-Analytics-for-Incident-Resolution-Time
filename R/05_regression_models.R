# R/05_regression_models.R
# Regression models for predicting incident resolution time.

train_regression_models <- function(train_data) {
  x_train <- train_data %>% select(-target)
  y_train <- train_data$target

  lm_model <- lm(target ~ ., data = train_data)

  tree_model <- rpart::rpart(target ~ ., data = train_data, method = "anova")

  rf_model <- randomForest::randomForest(
    x = x_train,
    y = y_train,
    ntree = 300,
    importance = TRUE
  )

  xgb_train <- xgboost::xgb.DMatrix(data = as.matrix(x_train), label = y_train)
  xgb_model <- xgboost::xgboost(
    data = xgb_train,
    objective = "reg:squarederror",
    nrounds = 150,
    max_depth = 5,
    eta = 0.08,
    verbose = 0
  )

  list(
    linear_regression = lm_model,
    decision_tree = tree_model,
    random_forest = rf_model,
    xgboost = xgb_model
  )
}

predict_regression_model <- function(model, model_name, test_data) {
  x_test <- test_data %>% select(-target)

  if (model_name == "xgboost") {
    preds <- predict(model, as.matrix(x_test))
  } else {
    preds <- predict(model, newdata = test_data)
  }

  preds
}
