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

plot_target_distribution <- function(df, target = "target", title = "Target Class Distribution") {
  ggplot(df, aes(x = .data[[target]])) +
    geom_bar(fill = "steelblue", alpha = 0.8) +
    labs(title = title, x = "Resolution Time Group", y = "Incident Count") +
    theme_minimal()
}

build_numeric_correlation <- function(df) {
  numeric_data <- df %>% select(where(is.numeric))
  cor(numeric_data, use = "complete.obs")
}

plot_correlation_heatmap <- function(correlation_matrix) {
  correlation_df <- as.data.frame(as.table(correlation_matrix))
  names(correlation_df) <- c("Feature_X", "Feature_Y", "Correlation")

  ggplot(correlation_df, aes(x = Feature_X, y = Feature_Y, fill = Correlation)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", Correlation)), size = 3) +
    scale_fill_gradient2(low = "firebrick", mid = "white", high = "steelblue", limits = c(-1, 1)) +
    coord_equal() +
    labs(title = "Numeric Feature Correlation Matrix", x = NULL, y = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

calculate_cramers_v <- function(df, categorical_cols) {
  available_cols <- intersect(categorical_cols, names(df))
  pairs <- combn(available_cols, 2, simplify = FALSE)

  dplyr::bind_rows(lapply(pairs, function(pair) {
    contingency <- table(df[[pair[1]]], df[[pair[2]]])
    chi_result <- suppressWarnings(chisq.test(contingency))
    sample_size <- sum(contingency)
    min_dimension <- min(nrow(contingency) - 1, ncol(contingency) - 1)
    cramers_v <- if (min_dimension > 0) {
      sqrt(as.numeric(chi_result$statistic) / (sample_size * min_dimension))
    } else {
      NA_real_
    }

    data.frame(
      Feature_1 = pair[1],
      Feature_2 = pair[2],
      Chi_Squared = as.numeric(chi_result$statistic),
      P_Value = chi_result$p.value,
      Cramers_V = cramers_v
    )
  })) %>%
    arrange(desc(Cramers_V))
}

plot_cramers_v <- function(cramers_v_df, top_n = 20) {
  plot_df <- cramers_v_df %>%
    filter(!is.na(Cramers_V)) %>%
    head(top_n) %>%
    mutate(Feature_Pair = paste(Feature_1, Feature_2, sep = " / "))

  ggplot(plot_df, aes(x = reorder(Feature_Pair, Cramers_V), y = Cramers_V)) +
    geom_col(fill = "steelblue", alpha = 0.8) +
    coord_flip() +
    labs(title = "Top Categorical Feature Associations", x = NULL, y = "Cramér's V") +
    theme_minimal()
}
