# R/04_model_preparation.R
# Prepare regression and classification datasets.

remove_unusable_columns <- function(df) {
  columns_to_drop <- intersect(
    c(
      "number", "sys_id", "incident_state", "active", "closed_at", "resolved_at",
      "opened_at", "sys_created_at", "sys_updated_at", "resolution_time_hours",
      "resolution_time_hours_capped", "log_resolution_time", "long_resolution"
    ),
    names(df)
  )
  df %>% select(-all_of(columns_to_drop))
}

prepare_regression_data <- function(df, train_ratio = 0.70) {
  modeling_df <- df %>%
    select(-long_resolution, -resolution_time_hours) %>%
    mutate(target = log_resolution_time)

  modeling_df <- modeling_df %>% select(-any_of(c("resolved_at", "opened_at", "closed_at")))

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
    select(-resolution_time_hours, -resolution_time_hours_capped, -log_resolution_time) %>%
    mutate(target = long_resolution)

  modeling_df <- modeling_df %>% select(-any_of(c("resolved_at", "opened_at", "closed_at")))

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
