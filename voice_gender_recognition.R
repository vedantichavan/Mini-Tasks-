# Install required packages if not already installed
install.packages("randomForest")

# Load the necessary libraries
library(randomForest)
library(ggplot2)


#------------------*Preprocess Data*---------------------#

# Load the voice dataset
voice_data <- read.csv("D:/voice.csv")
print(voice_data)
# Check the structure of the data
str(voice_data)

# Convert the label column to a factor 
voice_data$label <- as.factor(voice_data$label)

# Feature scaling
voice_data[,-ncol(voice_data)] <- scale(voice_data[,-ncol(voice_data)])


#----------------*Training Random Forest Classifier*----------#

# Split data into training and test sets
set.seed(123)
index <- sample(1:nrow(voice_data), 0.8 * nrow(voice_data))
train_data <- voice_data[index, ]
test_data <- voice_data[-index, ]

# Train Random Forest classifier
rf_model <- randomForest(label ~ ., data=train_data, ntree=100)

# Evaluate the model on the test set
pred <- predict(rf_model, test_data)
confusion_matrix <- table(pred, test_data$label)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 2)))
# Compute confusion matrix
confusion_matrix <- table(pred, test_data$label)

# Extract confusion matrix components
TP <- confusion_matrix[2, 2]  # True Positives
TN <- confusion_matrix[1, 1]  # True Negatives
FP <- confusion_matrix[1, 2]  # False Positives
FN <- confusion_matrix[2, 1]  # False Negatives

# Calculate metrics
specificity <- TN / (TN + FP)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)  # Same as Sensitivity
f1_score <- 2 * ((precision * recall) / (precision + recall))

# Print metrics
cat("Specificity:", round(specificity, 2), "\n")
cat("Precision:", round(precision, 2), "\n")
cat("Recall (Sensitivity):", round(recall, 2), "\n")
cat("F1-Score:", round(f1_score, 2), "\n")



#----------------*Plot the importance of features in the model*-----------------

feature_importance <- importance(rf_model)
feature_importance_df <- data.frame(Feature = rownames(feature_importance), Importance = feature_importance[, 1])

# Plot the feature importance
ggplot(feature_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Feature Importance for Voice Gender Prediction", x = "Feature", y = "Importance") +
  theme_minimal()

