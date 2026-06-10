# R/01_data_import_cleaning.R
# Functions for loading and cleaning the incident event log dataset.

clean_column_names <- function(df) {
  names(df) <- tolower(names(df))
  names(df) <- gsub("[^a-z0-9]+", "_", names(df))
  names(df) <- gsub("(^_|_$)", "", names(df))
  df
}

parse_datetime_safe <- function(x) {
  lubridate::parse_date_time(
    x,
    orders = c("dmy HM", "dmy HMS", "ymd HM", "ymd HMS", "mdy HM", "mdy HMS"),
    tz = "UTC"
  )
}

get_mode <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

impute_from_group_mode <- function(df, target_col, group_col) {
  if (!all(c(target_col, group_col) %in% names(df))) return(df)

  impute_map <- df %>%
    dplyr::filter(!is.na(.data[[target_col]]), !is.na(.data[[group_col]])) %>%
    dplyr::count(.data[[group_col]], .data[[target_col]], sort = TRUE) %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_col))) %>%
    dplyr::slice_head(n = 1) %>%
    dplyr::ungroup() %>%
    dplyr::select(dplyr::all_of(group_col), mapped_value = dplyr::all_of(target_col))

  df %>%
    dplyr::left_join(impute_map, by = group_col) %>%
    dplyr::mutate(
      !!target_col := ifelse(is.na(.data[[target_col]]), mapped_value, .data[[target_col]])
    ) %>%
    dplyr::select(-mapped_value)
}

apply_code_r_imputation <- function(df) {
  df <- impute_from_group_mode(df, "location", "caller_id")

  if ("category" %in% names(df)) {
    df <- df[!is.na(df$category), ]
  }

  df <- impute_from_group_mode(df, "subcategory", "category")

  if ("u_symptom" %in% names(df)) {
    df$u_symptom[is.na(df$u_symptom)] <- "Details unavailable"
  }

  df <- impute_from_group_mode(df, "assignment_group", "subcategory")

  if ("assignment_group" %in% names(df)) {
    df <- df[!is.na(df$assignment_group), ]
  }

  df
}

load_incident_data <- function(file_path = "data/incident_event_log.csv") {
  df <- read.csv(file_path, stringsAsFactors = FALSE)
  df <- clean_column_names(df)
  return(df)
}

clean_incident_data <- function(df) {
  # Replace common missing-value markers with NA
  df[df == "?"] <- NA
  df[df == ""] <- NA
  df[df == "NULL"] <- NA
  df[df == "null"] <- NA

  # Remove duplicate records
  df <- dplyr::distinct(df)

  # Parse timestamp columns if available
  datetime_cols <- intersect(c("opened_at", "resolved_at", "closed_at", "sys_created_at", "sys_updated_at"), names(df))
  for (col in datetime_cols) {
    df[[col]] <- parse_datetime_safe(df[[col]])
  }

  # Keep only records with opened_at and resolved_at for resolution-time modeling
  if (all(c("opened_at", "resolved_at") %in% names(df))) {
    df <- df[!is.na(df$opened_at) & !is.na(df$resolved_at), ]
    df$resolution_time_hours <- as.numeric(difftime(df$resolved_at, df$opened_at, units = "hours"))
    df <- df[!is.na(df$resolution_time_hours) & df$resolution_time_hours >= 0, ]
  } else {
    stop("The dataset must contain opened_at and resolved_at columns to calculate resolution time.")
  }

  df <- apply_code_r_imputation(df)

  # Impute categorical missing values using mode
  categorical_cols <- names(df)[sapply(df, is.character)]
  for (col in categorical_cols) {
    mode_value <- get_mode(df[[col]])
    df[[col]][is.na(df[[col]])] <- mode_value
  }

  # Impute numeric missing values using median
  numeric_cols <- names(df)[sapply(df, is.numeric)]
  for (col in numeric_cols) {
    med_value <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med_value
  }

  return(df)
}
