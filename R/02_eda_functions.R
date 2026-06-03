# R/02_eda_functions.R
# Exploratory data analysis helper functions.

plot_resolution_distribution <- function(df, target = "resolution_time_hours") {
  ggplot(df, aes_string(x = target)) +
    geom_histogram(bins = 40, fill = "steelblue", alpha = 0.75) +
    scale_x_continuous(labels = scales::comma) +
    labs(
      title = "Distribution of Incident Resolution Time",
      x = "Resolution Time in Hours",
      y = "Incident Count"
    ) +
    theme_minimal()
}

plot_log_resolution_distribution <- function(df, target = "resolution_time_hours") {
  ggplot(df, aes_string(x = paste0("log1p(", target, ")"))) +
    geom_histogram(bins = 40, fill = "steelblue", alpha = 0.75) +
    labs(
      title = "Log-Transformed Distribution of Incident Resolution Time",
      x = "log1p(Resolution Time Hours)",
      y = "Incident Count"
    ) +
    theme_minimal()
}

plot_categorical_distribution <- function(df, feature, top_n = 15) {
  df %>%
    count(.data[[feature]], sort = TRUE) %>%
    head(top_n) %>%
    ggplot(aes(x = reorder(.data[[feature]], n), y = n)) +
    geom_col(fill = "steelblue", alpha = 0.75) +
    coord_flip() +
    labs(
      title = paste("Top", top_n, "Categories for", feature),
      x = feature,
      y = "Count"
    ) +
    theme_minimal()
}

summarize_resolution_by_group <- function(df, feature, target = "resolution_time_hours") {
  df %>%
    group_by(.data[[feature]]) %>%
    summarise(
      incident_count = n(),
      avg_resolution_hours = mean(.data[[target]], na.rm = TRUE),
      median_resolution_hours = median(.data[[target]], na.rm = TRUE),
      p90_resolution_hours = quantile(.data[[target]], 0.90, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(avg_resolution_hours))
}

plot_resolution_by_group <- function(df, feature, target = "resolution_time_hours", top_n = 10) {
  summary_df <- summarize_resolution_by_group(df, feature, target) %>%
    head(top_n)

  ggplot(summary_df, aes(x = reorder(.data[[feature]], avg_resolution_hours), y = avg_resolution_hours)) +
    geom_col(fill = "steelblue", alpha = 0.75) +
    coord_flip() +
    labs(
      title = paste("Average Resolution Time by", feature),
      x = feature,
      y = "Average Resolution Time in Hours"
    ) +
    theme_minimal()
}
