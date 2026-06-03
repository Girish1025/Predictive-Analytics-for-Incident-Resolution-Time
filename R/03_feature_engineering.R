# R/03_feature_engineering.R
# Feature engineering for incident resolution prediction.

cap_outliers <- function(x, lower_q = 0.01, upper_q = 0.99) {
  lower <- quantile(x, lower_q, na.rm = TRUE)
  upper <- quantile(x, upper_q, na.rm = TRUE)
  pmin(pmax(x, lower), upper)
}

engineer_incident_features <- function(df) {
  # Cap extreme resolution-time values for more stable modeling
  df$resolution_time_hours_capped <- cap_outliers(df$resolution_time_hours)
  df$log_resolution_time <- log1p(df$resolution_time_hours_capped)

  # Binary classification target: long vs normal resolution time
  threshold <- median(df$resolution_time_hours_capped, na.rm = TRUE)
  df$long_resolution <- ifelse(df$resolution_time_hours_capped > threshold, "Long", "Normal")
  df$long_resolution <- as.factor(df$long_resolution)

  # Time-based features from opened_at
  if ("opened_at" %in% names(df)) {
    df$opened_hour <- lubridate::hour(df$opened_at)
    df$opened_day <- lubridate::wday(df$opened_at, label = TRUE)
    df$opened_month <- lubridate::month(df$opened_at, label = TRUE)
    df$opened_weekend <- ifelse(lubridate::wday(df$opened_at) %in% c(1, 7), "Weekend", "Weekday")
  }

  # Convert common ordinal columns to factors if present
  factor_cols <- intersect(
    c("priority", "impact", "urgency", "category", "subcategory", "assignment_group", "contact_type", "location"),
    names(df)
  )
  for (col in factor_cols) {
    df[[col]] <- as.factor(df[[col]])
  }

  return(df)
}
