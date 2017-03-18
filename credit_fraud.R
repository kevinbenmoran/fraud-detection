#set seed to get replicable results
set.seed(321)

#install appropriate libraries
library(dplyr)
library(readr)
library(ggplot2)
library(caret)
library(reshape2)
library(ROSE)
library(pROC)
library(DMwR)
library(purrr)
library(randomForest)


##function to use pretty scientific notation in graphs using ggplot
scNotation <- function(l) {
  # turn in to character string in scientific notation
  l <- format(l, scientific = TRUE)
  # quote the part before the exponent to keep all the digits
  l <- gsub("^(.*)e", "'\\1'e", l)
  # turn the 'e+' into plotmath format
  l <- gsub("e", "%*%10^", l)
  # return this as an expression
  parse(text=l)
}


setwd("~/Desktop/GitHub2/Fraud/R")
data <- read_csv("creditcard.csv", col_types = cols(Class = col_factor(c(0,1))))

glimpse(data)


#Check ratio of fraudulent to genuine transactions

##Summarize data

Check_Proportion <- function(x) {
  x %>%
    group_by(Class) %>%
    summarise(no_rows = length(Class), prop = no_rows / nrow(x))
}


Check_Proportion(data)

# Plot classes to examine imbalance
ggplot(data, aes(x = Class, fill = Class)) +
  xlab("Type of transaction") +
  ylab("Number") +
  ggtitle("Real vs Fraudulent Transactions") +
  geom_bar()

# Basically non existent on ggplot

#Normalize amount data and check, and remove amount
data$ScaledAmount <- scale(data$Amount)[1:nrow(data)]
mean(data$ScaledAmount)
sd(data$ScaledAmount)

data <- select(data, -Time, -Amount)
data <- select(data, ScaledAmount, 1:28, Class)

#Split Data into Training and Test Set
trainIndex <- createDataPartition(data$Class, p = 0.7, list = FALSE)
data_train <- data[trainIndex,]
data_test <- data[-trainIndex,]
rm(trainIndex)

# Check proportions of newly created datasets
Check_Proportion(data_train)
Check_Proportion(data_test)


# Undersampling
# Use ROSE function to undersample majority class
data_under <- ovun.sample(Class ~., data = data_train, method = "under",
                          N = 2*nrow(data_train[data_train$Class == 1,]))$data
Check_Proportion(data_under)
ggplot(data_under, aes(x = Class, fill = Class)) +
  geom_bar() +
  xlab("Type of transaction") +
  ylab("Transactions") +
  ggtitle('Real vs. Fraudulent Transactions') +
  scale_fill_discrete(name = "Type", labels = c("Real", "Fraudulent"))


# Oversampling
# Fully oversmapling with a large dataset such as this is inefficient
data_over <- ovun.sample(Class ~., data = data_train, method = "over", 
                         N = 2*nrow(data_train[data_train$Class == 0,]))$data
Check_Proportion(data_over)

################## Fill in plot here

#Both
data_both <- ovun.sample(Class ~., data = data_train, method = "both", p=0.5,
                         N = nrow(data_train))$data
Check_Proportion(data_both)

################## Fill in plot here

#ROSE (Random Over Sampling Examples)
data_rose <- ROSE(Class ~., data_train)$data
Check_Proportion(data_rose)

#SMOTE
data_train_for_smote <- as.data.frame(data_train)
data_smote <- SMOTE(Class ~., data_train_for_smote, perc.over = 100, perc.under = 200)
Check_Proportion(data_smote)

# |-------------------Models-------------------------|



########
run_logistic <- function(d, cut_off = 0.5) {
  
  
  logitModel <- glm(Class ~., 
                    family = binomial(link = 'logit'),
                    data   = d)
  
  p_train <- predict(logitModel, type = 'response')
  p_train <- ifelse(p_train > cut_off, 1, 0)
  print("Training Set Confusion Matrix:")
  conf_train <- confusionMatrix(p_train, d$Class)
  print(conf_train)
  
  
  p_test <- predict(logitModel, newdata = data_test, type = 'response')
  p_test <- ifelse(p_test > cut_off, 1, 0)
  conf_test <- confusionMatrix(p_test,data_test$Class)
  print("Test Set Confusion Matrix:")
  print(conf_test)
  
  
  comparison_plot <-cbind(d, p_train)
  glimpse
  print(qplot(factor(Class), data=comparison_plot, geom="bar", fill=factor(p_train)))
  return (conf_test)
}

#################
# Function to return given parameters from a confusion matrix
# Can be expanded

access_conf <- function(x, measure) {
  if (measure == "Accuracy") {
    val <- x[[3]][[1]]
  } else if (measure == "Specificity") {
    val <- x[[4]][[2]]
  } else if (measure == "Sensitivity") {
    val <- x[[4]][[1]]
  } else {
    stop("No valid measure selected")
  }
  
  return (round(val, 4))
}






# -------------------Accessing Confusion matrices--------------
conf_data <- run_logistic(data_train)
conf_under <- run_logistic(data_under)
conf_over <- run_logistic(data_over)
conf_both <- run_logistic(data_both)
conf_rose <- run_logistic(data_rose)
conf_smote <- run_logistic(data_smote)

df <- data_frame(sampling = c('data', 'under', 'over', 'both', 'rose', 'smote'))
datasets <- list(conf_data, conf_under, conf_over, conf_both, conf_rose, conf_smote)

df$accuracy <- unlist(map(datasets, access_conf, measure ="Accuracy"))
df$sensitivity <- unlist(map(datasets, access_conf, measure ="Sensitivity"))
df$specificity <- unlist(map(datasets, access_conf, measure ="Specificity"))
df



# --------------------Random Forest-----------------------------

p_train <- predict(logitModel, type = 'response')


rf_smote <- train(Class ~ ., data = data_smote, method = "rf")


predictors <- names(data_smote)[names(data_smote) != 'Class']
pred <- predict(rf_smote, data_test[,predictors], type = "prob")
# pred_adj <- ifelse(pred[1] > 0.55, 0, 1)
conf_rf_smote <- confusionMatrix(pred, data_test$Class)
# conf_rf_smote_adj <- confusionMatrix(pred_adj, data_test$Class)


# -----------------Random Forest Function-----------------------

run_ml <- function(x, model, threshold = 0.5) {
  
  ml_model <- train(Class ~ ., data = x, method = model)


  predictors <- names(x)[names(x) != 'Class']
  pred <- predict(ml_model, data_test[,predictors], type = "prob")
  pred <- ifelse(pred[1] > threshold, 0, 1)
  conf <- confusionMatrix(pred, data_test$Class)
  print(roc(response = as.numeric(data_test$Class), predictor = as.numeric(pred)))
  conf

}

conf_rf_under <- run_ml(data_under, model = "rf")
conf_rf_smote <- run_ml(data_smote, model = "rf")

conf_rf_smote
conf_rf_under

