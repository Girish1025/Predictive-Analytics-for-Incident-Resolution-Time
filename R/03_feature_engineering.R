# R/03_feature_engineering.R
# Feature engineering for incident resolution prediction.

cap_outliers <- function(x, lower_q = 0.01, upper_q = 0.99) {
  lower <- quantile(x, lower_q, na.rm = TRUE)
  upper <- quantile(x, upper_q, na.rm = TRUE)
  pmin(pmax(x, lower), upper)
}

create_time_group <- function(time_taken) {
  labels <- c("Immediate", "Short", "Long", "Very Long")
  code_r_breaks <- c(-Inf, 1, 24, 72, Inf)

  time_group <- cut(
    time_taken,
    breaks = code_r_breaks,
    labels = labels,
    right = FALSE
  )

  if (dplyr::n_distinct(time_group[!is.na(time_group)]) >= 2) {
    attr(time_group, "breaks") <- code_r_breaks
    attr(time_group, "break_source") <- "code.R fixed breaks"
    return(time_group)
  }

  quartiles <- as.numeric(stats::quantile(time_taken, probs = c(0.25, 0.50, 0.75), na.rm = TRUE))
  fallback_breaks <- c(-Inf, quartiles, Inf)

  if (length(unique(fallback_breaks)) != length(fallback_breaks)) {
    stop("Unable to create four time_group classes because time_taken does not have enough distinct values.")
  }

  time_group <- cut(
    time_taken,
    breaks = fallback_breaks,
    labels = labels,
    right = FALSE
  )
  attr(time_group, "breaks") <- fallback_breaks
  attr(time_group, "break_source") <- "quartile fallback breaks"
  time_group
}

engineer_incident_features <- function(df) {
  # code.R uses time_taken as the modeled resolution-time feature.
  df$time_taken <- df$resolution_time_hours

  df$time_group <- create_time_group(df$time_taken)
  df$time_group <- as.factor(df$time_group)

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

  # Convert common ordinal columns to factors
  factor_cols <- intersect(
    c(
      "location", "category", "subcategory", "u_symptom", "impact", "urgency",
      "assignment_group", "knowledge", "u_priority_confirmation"
    ),
    names(df)
  )
  for (col in factor_cols) {
    df[[col]] <- as.factor(df[[col]])
  }

  return(df)
}
