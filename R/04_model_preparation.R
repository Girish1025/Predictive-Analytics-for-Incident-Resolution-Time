# R/04_model_preparation.R
# Prepare regression and classification datasets.

remove_unusable_columns <- function(df) {
  columns_to_drop <- intersect(
    c(
      "number", "sys_id", "incident_state", "active", "closed_at", "resolved_at",
      "opened_at", "sys_created_at", "sys_updated_at", "resolution_time_hours",
      "resolution_time_hours_capped", "log_resolution_time", "long_resolution",
      "time_taken", "time_group"
    ),
    names(df)
  )
  df %>% select(-all_of(columns_to_drop))
}

code_r_model_columns <- c(
  "reassignment_count", "reopen_count", "sys_mod_count", "location",
  "category", "subcategory", "u_symptom", "impact", "urgency",
  "assignment_group", "knowledge", "u_priority_confirmation",
  "opened_hour", "opened_day", "opened_month", "opened_weekend",
  "time_group"
)

code_r_categorical_columns <- c(
  "location", "category", "subcategory", "u_symptom", "impact", "urgency",
  "assignment_group", "knowledge", "u_priority_confirmation",
  "opened_day", "opened_month", "opened_weekend"
)

drop_constant_predictors <- function(df, target_col = "target") {
  predictor_cols <- setdiff(names(df), target_col)
  keep_predictor <- vapply(
    df[predictor_cols],
    function(x) dplyr::n_distinct(x, na.rm = TRUE) > 1,
    logical(1)
  )

  df %>% select(all_of(names(keep_predictor)[keep_predictor]), all_of(target_col))
}

encode_categorical_predictors <- function(train_data, test_data, categorical_cols) {
  encoders <- list()
  available_cols <- intersect(categorical_cols, names(train_data))

  for (col in available_cols) {
    levels_train <- sort(unique(as.character(train_data[[col]])))
    levels_train <- levels_train[!is.na(levels_train)]
    encoders[[col]] <- levels_train

    train_data[[col]] <- as.numeric(factor(as.character(train_data[[col]]), levels = levels_train))
    test_data[[col]] <- as.numeric(factor(as.character(test_data[[col]]), levels = levels_train))

    train_data[[col]][is.na(train_data[[col]])] <- 0
    test_data[[col]][is.na(test_data[[col]])] <- 0
  }

  list(train = train_data, test = test_data, encoders = encoders)
}

prepare_regression_data <- function(df, train_ratio = 0.70) {
  modeling_df <- remove_unusable_columns(df)
  modeling_df$target <- df$log_resolution_time
  modeling_df <- drop_constant_predictors(modeling_df)

  encoded_x <- model.matrix(target ~ ., data = modeling_df)[, -1]
  encoded_df <- as.data.frame(encoded_x)
  encoded_df$target <- modeling_df$target

  set.seed(123)
  train_index <- createDataPartition(encoded_df$target, p = train_ratio, list = FALSE)

  list(
    train = encoded_df[train_index, ],
    test = encoded_df[-train_index, ]
  )
}

prepare_classification_data <- function(df, train_ratio = 0.70) {
  modeling_df <- df %>%
    select(any_of(code_r_model_columns)) %>%
    filter(!is.na(time_group))

  if (dplyr::n_distinct(modeling_df$time_group) < 2) {
    stop(
      "The code.R time_group bins produced fewer than two classes. ",
      "Check the time_taken distribution or revise the Immediate/Short/Long/Very Long breaks."
    )
  }

  set.seed(123)
  train_index <- createDataPartition(modeling_df$time_group, p = train_ratio, list = FALSE)

  train_data <- modeling_df[train_index, ]
  test_data <- modeling_df[-train_index, ]

  y_train <- as.factor(train_data$time_group)
  y_test <- factor(test_data$time_group, levels = levels(y_train))

  train_data$time_group <- NULL
  test_data$time_group <- NULL

  encoded <- encode_categorical_predictors(train_data, test_data, code_r_categorical_columns)
  train_data <- encoded$train
  test_data <- encoded$test

  train_data$target <- y_train
  test_data$target <- y_test

  train_data <- drop_constant_predictors(train_data)
  keep_cols <- names(train_data)
  test_data <- test_data %>% select(all_of(keep_cols))

  list(
    train = train_data,
    test = test_data,
    label_encoders = encoded$encoders
  )
}
