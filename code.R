
#Loading required Libraries
library(lubridate)
library(dplyr)
library(corrplot)
library(vcd)
library(caret)
library(UBL)
library(rpart)
library(rpart.plot)
library(rattle)
library(themis)
library(recipes)
library(tibble)
library(yardstick)
library(ggplot2)
library(tidyr)
library(scales)
library(C50)
library(randomForest)
library(xgboost)
library(Matrix)


#Importing Dataset

df_data_main <- read.csv('E:/UTA materials/Sem 2/6303/Final Project/incident_event_log.csv')

#Taking copy of the dataset
df_data = df_data_main
str(df_data)
#summary of the dataset
summary(df_data)
head(df_data)

#Listing the data types of the features
str(df_data)

#summary of categorical and numerical columns
categorical_col <- sapply(df_data, is.factor) | sapply(df_data, is.character)

categorical_col <- names(df_data)[categorical_col]

numerical_col <- sapply(df_data, is.numeric)

numerical_col <- names(df_data)[numerical_col]

print(categorical_col)

print(numerical_col)

#Summary of the categorical columns
summary(df_data[,categorical_col])

#Summary of the numerical columns
summary(df_data[,numerical_col])

#Evaluating the rows with string "?"
sapply(df_data, function(col) sum(as.character(col) == "?", na.rm = TRUE))

#Dropping the rows with "?" for the feature resolved_at
df_data <- df_data[df_data$resolved_at != "?", ]


########################################################################

#Feature engineering

# Using lubridate function to parse date & times
df_data$opened_at <- dmy_hm(df_data$opened_at, tz = "UTC")
df_data$resolved_at <- dmy_hm(df_data$resolved_at, tz = "UTC")

# Initialize the time_taken and calculate it
df_data$time_taken <- NA
df_data$time_taken <- as.numeric(
  difftime(df_data$resolved_at, df_data$opened_at, units = "hours")
)

#Summary of the feature time_taken
summary(df_data$time_taken)

#Checking number of rows with string "?" in the data set
sapply(df_data, function(col) sum(as.character(col) == "?", na.rm = TRUE))

#########################Data Prep rocessing############################################
#########################Handling Missing values #######################################

##Taking copy of the data set
data_model = df_data

#Imputing missing values on feature location based on caller_id

location_map <- data_model %>%
  filter(!is.na(caller_id) & !is.na(location) & location != "?") %>%
  group_by(caller_id, location) %>%
  tally() %>%
  arrange(caller_id, desc(n)) %>%
  slice_head(n = 1) %>%
  select(caller_id, location) %>%
  rename(mapped_location = location)

data_model <- data_model %>%
  left_join(location_map, by = "caller_id")

# Update the location feature only if it's missing or "?" AND caller_id is not NA
data_model <- data_model %>%
  mutate(location = ifelse(
    !is.na(caller_id) & (is.na(location) | location == "?"),
    mapped_location,
    location
  )) %>%
  select(-mapped_location)


#Imputing missing values on feature subcategory based on category

#Dropping rows where category is ?

data_model <- data_model[data_model$category != "?", ]

# Replacing "?" with NA for cleaner processing on feature "Subcategory"
data_model$subcategory[data_model$subcategory == "?"] <- NA

# Creating mapping for the most frequent subcategory for each category
subcategory_map <- data_model %>%
  filter(!is.na(subcategory)) %>%
  group_by(category, subcategory) %>%
  tally() %>%
  arrange(category, desc(n)) %>%
  slice_head(n = 1) %>%
  select(category, subcategory) %>%
  rename(imputed_subcategory = subcategory)


data_model <- data_model %>%
  left_join(subcategory_map, by = "category") %>%
  mutate(subcategory = ifelse(is.na(subcategory), imputed_subcategory, subcategory)) %>%
  select(-imputed_subcategory)

#Imputing the missing values (rows with ?) for feature u_symptoms with "Details unavailable"
data_model$u_symptom[data_model$u_symptom == "?"] <- "Details unavailable"

#Imputing the feature "assingment_group" based on the most frequent value for each sub category

# Replacing "?" with NA for cleaner processing in feature "Assingment_group"
data_model$assignment_group[data_model$assignment_group == "?"] <- NA

# Creating mapping of most frequent assignment_group per subcategory
group_map <- data_model %>%
  filter(!is.na(assignment_group)) %>%
  group_by(subcategory, assignment_group) %>%
  tally() %>%
  arrange(subcategory, desc(n)) %>%
  slice_head(n = 1) %>%
  select(subcategory, assignment_group) %>%
  rename(imputed_group = assignment_group)

data_model <- data_model %>%
  left_join(group_map, by = "subcategory") %>%
  mutate(assignment_group = ifelse(is.na(assignment_group), imputed_group, assignment_group)) %>%
  select(-imputed_group)


#Selecting features required for  model

data_model <- data_model[, !(names(data_model) %in% c(
  "number", "incident_state", "active", "made_sla", "caller_id",
  "opened_by", "sys_created_by", "sys_created_at", "sys_updated_by",
  "sys_updated_at", "contact_type", "cmdb_ci", "assigned_to", "notify","opened_at",
  "closed_code", "resolved_by", "resolved_at", "closed_at","problem_id","rfc","vendor","caused_by"
))]

#Describe of the model features
str(data_model)

#Displaying the feature names used for the model
colnames(data_model)

#Performing Correlation analysis
#Numeric
# Selecting numeric columns only
numeric_data <- data_model %>% select(where(is.numeric))

#Computing correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")  # handles NAs

print(cor_matrix)

# Plotting heatmap with correlation values
corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",   
  tl.col = "black",        
  tl.srt = 45,             
  number.cex = 0.7,
  col = colorRampPalette(c("red", "white", "blue"))(200)
)

#Chi square test to check association between categorical features
# Selecting categorical columns
cat_data <- data_model %>% select(where(~ is.factor(.) || is.character(.)))
cat_data <- cat_data %>% mutate(across(everything(), as.factor))

cat_pairs <- combn(names(cat_data), 2, simplify = FALSE)

# Calculate Chi-square and Cramérs V
results <- lapply(cat_pairs, function(pair) {
  tbl <- table(cat_data[[pair[1]]], cat_data[[pair[2]]])
  chi <- suppressWarnings(chisq.test(tbl))
  v <- assocstats(tbl)$cramer
  data.frame(var1 = pair[1], var2 = pair[2], chi_sq = chi$statistic, p_value = chi$p.value, cramers_v = v)
}) %>% bind_rows()

# Fetching the top associations
results %>%
  arrange(desc(cramers_v)) %>%
  filter(cramers_v > 0.3)  # show only moderate+ associations

#From cramers_v association we found that "priority" is highly associated with "impact" and "urgency" hence we are dropping this feature
data_model <- data_model[, !(names(data_model) %in% "priority")]


#Checking for any rows with values "?" for the features selected for the model building
sapply(data_model, function(col) sum(as.character(col) == "?", na.rm = TRUE))


############################################Model Building###############################
#lising out the categorical columns present in the data set
categorical_cols <- c("location", "category", "subcategory", "u_symptom", 
                      "impact", "urgency", "assignment_group", 
                      "knowledge", "u_priority_confirmation")

#Taking copy of the data set
data_model_class = data_model

#Creating bins for the feature time_taken and update it into new column time_group
data_model_class$time_group <- cut(
  data_model_class$time_taken,
  breaks = c(-Inf, 1, 24, 72, Inf),  # combine 1–4 and 4–24 → now 1–24
  labels = c("Immediate", "Short", "Long", "Very Long"),
  right = FALSE
)

#Analyzing class distribution in featuer time_group
table(data_model_class$time_group)

#Dropping feature "time_taken" as it is not useful for our model because of the feature presence "time_group"
data_model_class <- data_model_class[, names(data_model_class) != "time_taken"]

#Checking for empty values 
colSums(is.na(data_model_class))

#We observed two rows which are empty for the feature "assignment_group" and we are dropping them as they will not affect much
data_model_class <- data_model_class[!is.na(data_model_class$assignment_group), ]

#Checking for empty values 
colSums(is.na(data_model_class))

##############################Decision tree model####################################

#Splitting the data set into training and testing set with 70:30 ratio
set.seed(1025)
train_index <- createDataPartition(data_model_class$time_group, p = 0.7, list = FALSE)
X_train <- data_model_class[train_index, ]
X_test  <- data_model_class[-train_index, ]

#converting target feature in training data set into factors
y_train <- as.factor(X_train$time_group)

#Converting the target feature in testing data set into factors based on the levels in training target feature
y_test  <- factor(X_test$time_group, levels = levels(y_train))

#Dropping target column from the predictos data set X_train and X_test
X_train$time_group <- NULL
X_test$time_group  <- NULL

#Label encoding the predictors

label_encoders <- list()

# Applying label encoding to categorical columns
for (col in categorical_cols) {
  
  # Fitting encoder on training data
  levels_train <- unique(X_train[[col]])
  label_encoders[[col]] <- levels_train
  
  # Encoding training data set
  X_train[[col]] <- as.numeric(factor(X_train[[col]], levels = levels_train))
  
  # Encoding testing data set using same levels as in training set
  X_test[[col]] <- as.numeric(factor(X_test[[col]], levels = levels_train))
}



# Taking a copy for the encoded data set
X_train_enc <- X_train
X_test_enc <- X_test

#Analysing the target feature before performing SMOTE operation to handle data imbalance issue

y_train <- as.factor(y_train)

ggplot(data = as.data.frame(y_train), aes(x = y_train)) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Distribution of Target",
    x = "Time Group",
    y = "Count"
  ) +
  theme_minimal()


#Handling data imbalance in target feature 

# Combining training data set features and target
X_train_enc$time_group <- y_train 

# Creating a recipe with SMOTE function by balancing all classes to the target feature
smote_recipe <- recipe(time_group ~ ., data = X_train_enc) %>%
  step_smote(time_group, over_ratio = 1) 

prep_smote <- prep(smote_recipe)
smoted_data <- juice(prep_smote)

# Evaluating the target feature class distribution after SMOTE 
table(smoted_data$time_group)


# Training the  Decision Tree model

model_dt <- rpart(
  time_group ~ ., 
  data = smoted_data, 
  method = "class",
  control = rpart.control(cp = 0.01)
)

tree_dt <- summary(model_dt)
head(tree_dt$split)


#Displaying the tree
rpart.plot(model_dt, type = 2, extra = 104, cex = 0.6, main = "Decision Tree After SMOTE")



# Predicting the target feature for testing data set
pred_test <- predict(model_dt, newdata = X_test, type = "class")

#Confusion matrix
confusionMatrix(pred_test, y_test)
conf_matrix$overall      
conf_matrix$byClass      


# Predicting probabilities
pred_probs <- predict(model_dt, newdata = X_test, type = "prob")


# Ensure factor levels match original labels
label_levels <- levels(y_train)

y_test <- factor(y_test, levels = label_levels)
pred_test <- factor(pred_test, levels = label_levels)

#Getting performance metric values
accuracy(eval_data, truth, prediction)
precision(eval_data, truth, prediction, estimator = "macro")
recall(eval_data, truth, prediction, estimator = "macro")
f_meas(eval_data, truth, prediction, estimator = "macro")

#ROC-AUC value calculation
roc_auc(eval_data, truth, `Immediate`, `Short`, `Long`, `Very Long`, estimator = "macro_weighted")

#F!-SCORE
f_meas(eval_data, truth, prediction, estimator = "micro")


#Displaying the obtained metrics in table
metrics <- tibble(
  Accuracy = accuracy(eval_data, truth, prediction)$.estimate,
  Precision = precision(eval_data, truth, prediction, estimator = "macro")$.estimate,
  Recall = recall(eval_data, truth, prediction, estimator = "macro")$.estimate,
  F1 = f_meas(eval_data, truth, prediction, estimator = "macro")$.estimate
)

metrics

#Plotting ROC curve

# Ensure class probability columns matches factor levels
colnames(eval_data)[3:ncol(eval_data)] <- levels(eval_data$truth)

# Computing ROC curve data
roc_data <- roc_curve(eval_data, truth, Immediate, Short, Long, `Very Long`) %>%
  rename(class = .level)

# Calculating the AUC value per class using one-vs-all logic
class_levels <- levels(eval_data$truth)

auc_data <- lapply(class_levels, function(cls) {
  eval_data %>%
    mutate(truth_bin = ifelse(truth == cls, cls, paste0("not_", cls))) %>%
    mutate(truth_bin = factor(truth_bin, levels = c(cls, paste0("not_", cls)))) %>%
    roc_auc(truth = truth_bin, !!sym(cls)) %>%
    mutate(class = cls)
}) %>%
  bind_rows() %>%
  mutate(
    auc_label = paste0("AUC: ", percent(.estimate, accuracy = 0.1))
  )

# combing AUC to ROC data
roc_data <- roc_data %>%
  left_join(auc_data, by = "class")

# Plotting ROC cuvre for each classes with AUC label
ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "steelblue", size = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  facet_wrap(~ class) +
  geom_text(
    data = auc_data,
    aes(x = 0.6, y = 0.1, label = auc_label),
    inherit.aes = FALSE,
    color = "black"
  ) +
  labs(
    title = "ROC Curve per Class with AUC",
    x = "1 - Specificity",
    y = "Sensitivity"
  ) +
  theme_minimal()

# AUC for each class with one versus rest logic
roc_auc(
  eval_data,
  truth = truth,
  Immediate, Short, Long, `Very Long`,
  estimator = "macro"
)

roc_auc(
  eval_data,
  truth = truth,
  Immediate, Short, Long, `Very Long`,
  estimator = "macro_weighted"
)

#Feature importance for the model Decision tree 

importance_df <- data.frame(
  Feature = names(model_dt$variable.importance),
  Importance = model_dt$variable.importance
) %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 10)  # Top 10

#Plotting the obtained feature importance in graph for better visualization
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "darkorange") +
  coord_flip() +
  labs(
    title = "Top 10 Feature Importances for decision Decision Tree",
    x = "Feature",
    y = "Importance"
  ) +
  theme_minimal()

######################
model_dt1 <- rpart(
  time_group ~ .,
  data = smoted_data,
  method = "class",
  parms = list(split = "information"),  # ← entropy/information gain
  control = rpart.control(cp = 0.01)
)

printcp(model_dt1)     # Shows CP table
summary(model_dt1)     # Shows splits and improvements

tree_summary <- summary(model_dt1)
head(tree_summary$splits)

top_info_gain <- as.data.frame(tree_summary$splits) %>%
  tibble::rownames_to_column("Feature") %>%
  arrange(desc(improve)) %>%
  slice_head(n = 10)

print(top_info_gain)

# Predicting the target feature for testing data set
pred_test_dt1 <- predict(model_dt1, newdata = X_test, type = "class")

#Confusion matrix
confusionMatrix(pred_test_dt1, y_test)

################################Model C50######################################

#Training the model C5.0
model_c50 <- C5.0(time_group ~ ., data = smoted_data)
summary(model_c50)
C5imp(model_c50)

# Predicting the values on the testing data set target feature 
pred_test_c50 <- predict(model_c50, newdata = X_test)

# Evaluating the class probabilities
pred_probs_c50 <- predict(model_c50, newdata = X_test, type = "prob")

#Confusion matrix 
confusionMatrix(pred_test_c50, y_test)

pred_test_c50 <- predict(model_c50, newdata = X_test)
pred_probs_c50 <- predict(model_c50, newdata = X_test, type = "prob")

# Combine truth, prediction, and probabilities into one data frame
eval_data_c50 <- data.frame(
  truth = y_test,
  prediction = pred_test_c50,
  pred_probs_c50
)

# Ensure column names matches the levels in the target
colnames(eval_data_c50)[3:ncol(eval_data_c50)] <- levels(eval_data_c50$truth)

# Macro-weighted AUC score 
roc_auc(eval_data_c50, truth, Immediate, Short, Long, `Very Long`, estimator = "macro_weighted")

# AUC Score for each class
class_levels <- levels(eval_data_c50$truth)

per_class_auc <- lapply(class_levels, function(cls) {
  eval_data_c50 %>%
    mutate(truth_bin = ifelse(truth == cls, cls, paste0("not_", cls))) %>%
    mutate(truth_bin = factor(truth_bin, levels = c(cls, paste0("not_", cls)))) %>%
    roc_auc(truth = truth_bin, !!sym(cls)) %>%
    mutate(class = cls)
}) %>% bind_rows()

print(per_class_auc)

roc_data <- roc_curve(eval_data_c50, truth, Immediate, Short, Long, `Very Long`) %>%
  rename(class = .level)

# Adding AUC labels for the calculated roc_data
auc_labels <- per_class_auc %>%
  mutate(label = paste0(class, " (AUC: ", percent(.estimate, 0.1), ")")) %>%
  select(class, label)

roc_data <- left_join(roc_data, auc_labels, by = "class")

# Plotting the ROC curve
ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = label)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = "Multiclass ROC Curve (C5.0)",
    x = "1 - Specificity",
    y = "Sensitivity",
    color = "Class (AUC)"
  ) +
  theme_minimal()


# Displaying the metrics in table
metrics_c50 <- tibble(
  Accuracy = accuracy(eval_data_c50, truth, prediction)$.estimate,
  Precision = precision(eval_data_c50, truth, prediction, estimator = "macro")$.estimate,
  Recall = recall(eval_data_c50, truth, prediction, estimator = "macro")$.estimate,
  F1 = f_meas(eval_data_c50, truth, prediction, estimator = "macro")$.estimate
)

metrics_c50


# one versus rest ROC Curve
roc_data_c50 <- roc_curve(eval_data_c50, truth, Immediate, Short, Long, `Very Long`) %>%
  rename(class = .level)

# AUC per class
auc_data_c50 <- lapply(levels(eval_data_c50$truth), function(cls) {
  eval_data_c50 %>%
    mutate(truth_bin = ifelse(truth == cls, cls, paste0("not_", cls))) %>%
    mutate(truth_bin = factor(truth_bin, levels = c(cls, paste0("not_", cls)))) %>%
    roc_auc(truth = truth_bin, !!sym(cls)) %>%
    mutate(class = cls)
}) %>%
  bind_rows() %>%
  mutate(
    auc_label = paste0("AUC: ", percent(.estimate, accuracy = 0.1))
  )

# Joining AUC to ROC data
roc_data_c50 <- roc_data_c50 %>%
  left_join(auc_data_c50, by = "class")

# Plotting ROC curve
ggplot(roc_data_c50, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "purple", size = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  facet_wrap(~ class) +
  geom_text(
    data = auc_data_c50,
    aes(x = 0.6, y = 0.1, label = auc_label),
    inherit.aes = FALSE
  ) +
  labs(title = "C50 ROC per Class", x = "1 - Specificity", y = "Sensitivity") +
  theme_minimal()

smoted_data$time_group <- as.factor(smoted_data$time_group)


#Feature importance for the model C5.0
importance_c50 <- C5imp(model_c50)

importance_df_c50 <- as.data.frame(importance_c50) %>%
  rownames_to_column("Feature") %>%
  arrange(desc(Overall)) %>%
  slice_head(n = 10)

#Plotting the feature importance
ggplot(importance_df_c50, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 Important Features (C5.0)",
    x = "Feature",
    y = "Importance (Gain Ratio)"
  ) +
  theme_minimal()


###################################Random forest###############################################

#Training the model
set.seed(1025)
model_rf <- randomForest(
  time_group ~ ., 
  data = smoted_data,
  ntree = 500,
  mtry = floor(sqrt(ncol(smoted_data) - 1)),
  importance = TRUE
)

print(model_rf)
varImpPlot(model_rf)


#Displaying the tree
tree_df <- getTree(model_rf, k = 1, labelVar = TRUE)
tree_df

varImpPlot(model_rf, main = "Variable Importance (Mean Decrease in Gini)")


# View split statistics
head(explain_forest)

#Predicting the target feature in the testing data set
pred_rf <- predict(model_rf, newdata = X_test)
pred_rf_train <- predict(model_rf, newdata = X_train)

# Ensure levels match for evaluation
pred_rf <- factor(pred_rf, levels = levels(y_test))
pred_rf_train <- factor(pred_rf_train, levels = levels(y_train))

#Confusion matrix for the model Random Forest

conf_matrix_rf <- confusionMatrix(pred_rf, y_test)
print(conf_matrix_rf)
conf_matrix_rf$overall  
conf_matrix_rf$byClass   

confusion_matrix_rf_train <- confusionMatrix(pred_rf_train, y_train)
print(confusion_matrix_rf_train)
table(y_train)
length(pred_rf)
length(y_train)


cm_df <- as.data.frame(conf_matrix_rf$table)
colnames(cm_df) <- c("Prediction", "Reference", "Freq")

# Plot using ggplot2
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1, size = 5) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted")

# Predicting the probabilities for the ROC-AUC
pred_probs_rf <- predict(model_rf, newdata = X_test, type = "prob")


eval_data_rf <- tibble(
  truth = y_test,
  prediction = pred_rf
) %>%
  bind_cols(as_tibble(pred_probs_rf))  # bind probability columns

# displayingthe obtained metrics in table
metrics_rf <- tibble(
  Accuracy = accuracy(eval_data_rf, truth, prediction)$.estimate,
  Precision = precision(eval_data_rf, truth, prediction, estimator = "macro")$.estimate,
  Recall = recall(eval_data_rf, truth, prediction, estimator = "macro")$.estimate,
  F1 = f_meas(eval_data_rf, truth, prediction, estimator = "macro")$.estimate
)

metrics_rf

# ROC - AUC value calculation and plotting 

# One versus rest ROC curves
roc_data_rf <- roc_curve(eval_data_rf, truth, Immediate, Short, Long, `Very Long`) %>%
  rename(class = .level)

# AUC per class
auc_data_rf <- lapply(levels(eval_data_rf$truth), function(cls) {
  eval_data_rf %>%
    mutate(truth_bin = ifelse(truth == cls, cls, paste0("not_", cls))) %>%
    mutate(truth_bin = factor(truth_bin, levels = c(cls, paste0("not_", cls)))) %>%
    roc_auc(truth = truth_bin, !!sym(cls)) %>%
    mutate(class = cls)
}) %>%
  bind_rows() %>%
  mutate(
    auc_label = paste0("AUC: ", percent(.estimate, accuracy = 0.1))
  )

roc_data_rf <- roc_data_rf %>%
  left_join(auc_data_rf, by = "class")

# Plotting the curve
ggplot(roc_data_rf, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "darkgreen", size = 1) +
  geom_abline(linetype = "dashed") +
  facet_wrap(~ class) +
  geom_text(
    data = auc_data_rf,
    aes(x = 0.6, y = 0.1, label = auc_label),
    inherit.aes = FALSE
  ) +
  labs(title = "Random Forest ROC per Class", x = "1 - Specificity", y = "Sensitivity") +
  theme_minimal()

# Plotting the ROC curve

importance_rf <- randomForest::importance(model_rf)

importance_df_rf <- data.frame(
  Feature = rownames(importance_rf),
  Importance = importance_rf[, "MeanDecreaseGini"]
) %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 10)

ggplot(importance_df_rf, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 Important Features (Random Forest)",
    x = "Feature",
    y = "Mean Decrease in Gini"
  ) +
  theme_minimal()

###################################XGBoost model#####################################

# Converting target feature to numeric labels
label_levels <- levels(smoted_data$time_group)
smoted_data$time_group <- as.numeric(factor(smoted_data$time_group, levels = label_levels)) - 1
y_test_num <- as.numeric(factor(y_test, levels = label_levels)) - 1

# Converting the data into matrix format
dtrain <- xgb.DMatrix(data = as.matrix(smoted_data %>% select(-time_group)), label = smoted_data$time_group)
dtest  <- xgb.DMatrix(data = as.matrix(X_test), label = y_test_num)

# Training the model
set.seed(1025)
model_xgb <- xgboost(
  data = dtrain,
  objective = "multi:softprob",         
  num_class = length(label_levels),
  nrounds = 100,
  max_depth = 5,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  eval_metric = "mlogloss",
  verbose = 0
)

# Predict target variables in the testing data set
pred_probs_xgb <- predict(model_xgb, dtest)
pred_matrix <- matrix(pred_probs_xgb, ncol = length(label_levels), byrow = TRUE)

pred_xgb <- max.col(pred_matrix) - 1
pred_xgb_factor <- factor(pred_xgb, levels = 0:(length(label_levels) - 1), labels = label_levels)

#Confusion metrics

conf_matrix_xgb <- confusionMatrix(pred_xgb_factor, y_test)
print(conf_matrix_xgb)
conf_matrix_xgb$overall
conf_matrix_xgb$byClass

cm_xgb <- as.data.frame(conf_matrix_xgb$table)
colnames(cm_xgb) <- c("Prediction", "Reference", "Freq")

# Plot using ggplot2
ggplot(cm_xgb, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1, size = 5) +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted")



eval_data_xgb <- tibble(
  truth = y_test,
  prediction = pred_xgb_factor
) %>%
  bind_cols(as_tibble(pred_matrix))

# Assigning column names to match class labels
colnames(eval_data_xgb)[3:ncol(eval_data_xgb)] <- label_levels

# Displaying the metrics in data
metrics_xgb <- tibble(
  Accuracy = accuracy(eval_data_xgb, truth, prediction)$.estimate,
  Precision = precision(eval_data_xgb, truth, prediction, estimator = "macro")$.estimate,
  Recall = recall(eval_data_xgb, truth, prediction, estimator = "macro")$.estimate,
  F1 = f_meas(eval_data_xgb, truth, prediction, estimator = "macro")$.estimate
)

metrics_xgb

# ROC AUC curve

roc_data_xgb <- roc_curve(eval_data_xgb, truth, Immediate, Short, Long, `Very Long`) %>%
  rename(class = .level)

auc_data_xgb <- lapply(label_levels, function(cls) {
  eval_data_xgb %>%
    mutate(truth_bin = ifelse(truth == cls, cls, paste0("not_", cls))) %>%
    mutate(truth_bin = factor(truth_bin, levels = c(cls, paste0("not_", cls)))) %>%
    roc_auc(truth = truth_bin, !!sym(cls)) %>%
    mutate(class = cls)
}) %>%
  bind_rows() %>%
  mutate(
    auc_label = paste0("AUC: ", percent(.estimate, accuracy = 0.1))
  )

roc_data_xgb <- roc_data_xgb %>%
  left_join(auc_data_xgb, by = "class")

#Plotting the curve 
ggplot(roc_data_xgb, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "firebrick", size = 1) +
  geom_abline(linetype = "dashed") +
  facet_wrap(~ class) +
  geom_text(data = auc_data_xgb, aes(x = 0.6, y = 0.1, label = auc_label), inherit.aes = FALSE) +
  labs(title = "XGBoost ROC per Class", x = "1 - Specificity", y = "Sensitivity") +
  theme_minimal()

# Feature importance graphs
importance_matrix <- xgb.importance(model = model_xgb)
importance_top10 <- importance_matrix %>%
  arrange(desc(Gain)) %>%
  slice_head(n = 10)

xgb.plot.importance(importance_top10, rel_to_first = TRUE, top_n = 10)

#Models performance metrics comparison

metrics_all <- bind_rows(
  metrics  %>% mutate(Model = "Decision Tree"),
  metrics_c50 %>% mutate(Model = "C5.0"),
  metrics_rf  %>% mutate(Model = "Random Forest"),
  metrics_xgb %>% mutate(Model = "XGBoost")
) %>%
  relocate(Model)


print(metrics_all)

# Plotting metrics comparison graph
metrics_long <- metrics_all %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Score")

ggplot(metrics_long, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Comparison on Evaluation Metrics", y = "Score", x = "") +
  theme_minimal()
