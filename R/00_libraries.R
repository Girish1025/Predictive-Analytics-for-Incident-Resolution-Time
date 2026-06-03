# R/00_libraries.R
# Load required libraries for the incident resolution time prediction workflow.

required_packages <- c(
  "lubridate",
  "dplyr",
  "ggplot2",
  "caret",
  "rpart",
  "rpart.plot",
  "randomForest",
  "xgboost",
  "Matrix",
  "yardstick",
  "recipes",
  "themis",
  "tibble",
  "tidyr",
  "scales"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

invisible(lapply(required_packages, install_if_missing))
invisible(lapply(required_packages, library, character.only = TRUE))

options(max.print = 100000)
set.seed(123)
