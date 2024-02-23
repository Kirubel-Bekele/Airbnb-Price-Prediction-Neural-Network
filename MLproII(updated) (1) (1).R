# Load necessary libraries
library(caret) # For data splitting, pre-processing, and feature engineering
library(neuralnet) # For neural network modeling
library(dplyr) # For data manipulation
library(pROC) # For ROC curve and AUC analysis
library(fastDummies) # For converting categorical variables into dummy/indicator variables
library(sigmoid) # For the sigmoid function, though not directly used in the provided code snippet

# Data loading and initial preprocessing
# Read the AirbnbListings.csv file into a dataframe. Treat empty strings and "NA" as NA (missing values). Convert strings to factors for categorical analysis.

df = read.csv("AirbnbListings.csv", sep=",", stringsAsFactors = T, na.strings=c("","NA"))

#.1 Preprocess the data and prepare it for the running neural network. (30 points)
# Data cleaning and transformation
df$bathrooms <- as.numeric(df$bathrooms) # Convert 'bathrooms' column to numeric type in case it was imported as a different type.
summary(df) # Provides a statistical summary of the dataframe, useful for initial data analysis.

# Feature engineering: Creating dummy variables
# Convert categorical variables ('neighborhood', 'superhost', 'room_type') into dummy variables. Remove the original columns from the dataframe and also remove the first dummy column of each categorical variable to avoid dummy variable trap.
df_dummies = dummy_cols(df, select_columns = c('neighborhood', 'superhost','room_type'), 
                        remove_selected_columns = T,
                        remove_first_dummy = T)
# Convert 'host_since' to Date format
df$host_since <- as.Date(df$host_since, format="%Y-%m-%d")
# Calculate 'age' in days
df$age <- as.numeric(Sys.Date() - df$host_since)
# Create a final dataframe excluding 'listing_id' and columns, possibly because they are identifiers or not relevant for modeling.
final_df = df_dummies %>% select(-c('listing_id','host_since'))

# Data partitioning into training and testing sets
set.seed(123) # Set a seed for reproducibility of results.
index = sample(nrow(final_df), 0.7*nrow(final_df)) # Randomly sample 70% of the rows to create a training set index.

train_data = final_df[index,] # Create the training dataset based on the index.
test_data = final_df[-index,] # Create the testing dataset with the remaining data.

# Missing values analysis
sapply(train_data, function(x){sum(is.na(x))}) # Count NAs in the training data.
sapply(test_data, function(x){sum(is.na(x))}) # Count NAs in the testing data.

sum(is.na(train_data$bedrooms))/nrow(train_data) # Calculate the proportion of missing values in the 'bedrooms' column of the training data.
sum(is.na(train_data$`room_type_Private room`))/nrow(train_data)
sum(is.na(train_data$`room_type_Shared room`))/nrow(train_data)
sum(is.na(train_data$host_acceptance_rate))/nrow(train_data) # Calculate the proportion of missing values in the 'host_acceptance_rate' column of the training data.
sum(is.na(test_data$bedrooms))/nrow(test_data) # Calculate the proportion of missing values in the 'bedrooms' column of the testing data.
sum(is.na(test_data$host_acceptance_rate))/nrow(test_data) # Calculate the proportion of missing values in the 'host_acceptance_rate' column of the testing data.
sum(is.na(test_data$`room_type_Private room`))/nrow(train_data)
sum(is.na(test_data$`room_type_Shared room`))/nrow(train_data)

# Define a function to compute the mode for categorical variables
mode_imputation <- function(x) {
  ux <- unique(x[!is.na(x)])
  most_frequent <- ux[which.max(tabulate(match(x, ux)))]
  x[is.na(x)] <- most_frequent
  return(x)
}

# Impute missing values for numerical variables with median, one by one
train_data$bedrooms[is.na(train_data$bedrooms)] <- median(train_data$bedrooms, na.rm = TRUE)
train_data$host_acceptance_rate[is.na(train_data$host_acceptance_rate)] <- median(train_data$host_acceptance_rate, na.rm = TRUE)
test_data$bedrooms[is.na(test_data$bedrooms)] <- median(test_data$bedrooms, na.rm = TRUE)
test_data$host_acceptance_rate[is.na(test_data$host_acceptance_rate)] <- median(test_data$host_acceptance_rate, na.rm = TRUE)

# Impute missing values for categorical variables with mode, one by one
# For the training data
train_data$`room_type_Private room` <- mode_imputation(train_data$`room_type_Private room`)
train_data$`room_type_Shared room` <- mode_imputation(train_data$`room_type_Shared room`)

# For the testing data
test_data$`room_type_Private room` <- mode_imputation(test_data$`room_type_Private room`)
test_data$`room_type_Shared room` <- mode_imputation(test_data$`room_type_Shared room`)

# Verify the imputation
sum(is.na(train_data$bedrooms)) / nrow(train_data) # Verify no missing values remain in the 'bedrooms' column of the training data.
sum(is.na(train_data$host_acceptance_rate)) / nrow(train_data) # Verify no missing values remain in the 'host_acceptance_rate' column of the training data.
sum(is.na(test_data$bedrooms)) / nrow(test_data) # Verify no missing values remain in the 'bedrooms' column of the testing data.
sum(is.na(test_data$host_acceptance_rate)) / nrow(test_data) # Verify no missing values remain in the 'host_acceptance_rate' column of the testing data.

# Data normalization
scale_vals = preProcess(train_data, method = 'range') # Use caret's preProcess function to scale the training data to the range [0, 1].
train_data_s = predict(scale_vals, train_data) # Apply the scaling to the training data.
test_data_s = predict(scale_vals, test_data) # Apply the scaling to the test data based on training data parameters.
train_data_s <- as.data.frame(train_data_s) # Convert the scaled training data back to a dataframe format.

# Rename columns to remove spaces, facilitating easier reference in modeling and analysis
colnames(train_data_s)[colnames(train_data_s) == "neighborhood_Dupont Circle"] <- "neighborhood_Dupont_Circle"
colnames(train_data_s)[colnames(train_data_s) == "neighborhood_Foggy Bottom"] <- "neighborhood_Foggy_Bottom"
colnames(train_data_s)[colnames(train_data_s) == "neighborhood_Georgetown"] <- "neighborhood_Georgetown"
colnames(train_data_s)[colnames(train_data_s) == "neighborhood_Shaw"] <- "neighborhood_Shaw"
colnames(train_data_s)[colnames(train_data_s) == "neighborhood_Union Station"] <- "neighborhood_Union_Station"
colnames(train_data_s)[colnames(train_data_s) == "superhost_TRUE"] <- "superhost_TRUE"
colnames(train_data_s)[colnames(train_data_s) == "room_type_Entire home/apt"] <- "room_type_Entire_home_apt"
colnames(train_data_s)[colnames(train_data_s) == "room_type_Private room"] <- "room_type_Private_room"
colnames(train_data_s)[colnames(train_data_s) == "room_type_Shared room"] <- "room_type_Shared_room"

# Repeat the process for test_data_s for consistency between training and test datasets
colnames(test_data_s)[colnames(test_data_s) == "neighborhood_Dupont Circle"] <- "neighborhood_Dupont_Circle"
colnames(test_data_s)[colnames(test_data_s) == "neighborhood_Foggy Bottom"] <- "neighborhood_Foggy_Bottom"
colnames(test_data_s)[colnames(test_data_s) == "neighborhood_Georgetown"] <- "neighborhood_Georgetown"
colnames(test_data_s)[colnames(test_data_s) == "neighborhood_Shaw"] <- "neighborhood_Shaw"
colnames(test_data_s)[colnames(test_data_s) == "neighborhood_Union Station"] <- "neighborhood_Union_Station"
colnames(test_data_s)[colnames(test_data_s) == "superhost_TRUE"] <- "superhost_TRUE"
colnames(test_data_s)[colnames(test_data_s) == "room_type_Entire home/apt"] <- "room_type_Entire_home_apt"
colnames(test_data_s)[colnames(test_data_s) == "room_type_Private room"] <- "room_type_Private_room"
colnames(test_data_s)[colnames(test_data_s) == "room_type_Shared room"] <- "room_type_Shared_room"


#2 Train two different neural network models using the 'neuralnet' package and the 'caret' package (based on the 'nnet' method or another neural network package that caret supports). (30 points)  
# Train a neural network model (NN1) with one hidden layer of 1 neurons, using ReLU activation function and a maximum step limit
NN1 = neuralnet(price~.,
                data=train_data_s,
                linear.output = TRUE,
                stepmax = 1e+06,
                act.fct = tanh,
                hidden = 1)

# Display NN1 structure, plot it, show net results, activation function, and result matrix for analysis
NN1
plot(NN1)
NN1$net.result[[1]]
NN1$act.fct 
NN1$result.matrix



# Make predictions with NN1 on test data, rescale predictions to original price scale, and compute performance metrics
pred1 = predict(NN1, test_data_s)
pred1_acts = pred1*(max(train_data$price)-min(train_data$price))+min(train_data$price)
metrics_NN1 <- postResample(pred1_acts, test_data$price)
print(metrics_NN1)



# Plot actual vs. predicted prices for NN1
plot(test_data$price, pred1_acts, 
     main = "Actual vs Predicted Prices for NN1",
     xlab = "Actual Prices",
     ylab = "Predicted Prices",
     pch = 19, col = "blue")
abline(a = 0, b = 1, lty = 2, col = "red")
legend("topright", legend = c("Predictions", "Identity Line"), 
       col = c("blue", "red"), pch = c(19, NA), lty = c(NA, 2))


# Train another neural network model (NN2) with two hidden layers of 5 and 3 neurons, respectively, using ReLU and a lower maximum step limit
NN2 = neuralnet(price~.,
                data=train_data_s,
                linear.output = TRUE,
                stepmax = 1e+06,
                act.fct = relu,
                hidden=c(5,2))

# Display NN2 structure, plot it, show net results, activation function, and result matrix for analysis
NN2
plot(NN2)
NN2$net.result[[1]]
NN2$act.fct 
NN2$result.matrix

# Make predictions with NN2 on test data, rescale predictions to original price scale, and compute performance metrics
pred2 = predict(NN2, test_data_s)
pred2_acts = pred2*(max(train_data$price)-min(train_data$price))+min(train_data$price)
metrics_NN2 <- postResample(pred2_acts, test_data$price)
print(metrics_NN2)

# Calculate rescaled predictions for NN2
pred2_rescaled <- pred2 * (max(df$price) - min(df$price)) + min(df$price)
# Plot actual vs. predicted prices for NN2
plot(test_data$price, pred2_rescaled, 
     main = "Actual vs Predicted Prices for NN2",
     xlab = "Actual Prices",
     ylab = "Predicted Prices",
     pch = 19, col = "green")
abline(a = 0, b = 1, lty = 2, col = "red")
legend("topright", legend = c("Predictions", "Identity Line"), 
       col = c("green", "red"), pch = c(19, NA), lty = c(NA, 2))





# Set up 10-fold cross-validation for model training
ctrl = trainControl(method="cv", number = 10)

# Set a random seed for reproducible results
set.seed(123)

# Train a neural network model (NN_caret1) using the 'nnet' method from the 'caret' package with default tuning
NN_caret1 = train(price ~., 
                  data = train_data_s,
                  method = "nnet",
                  trControl = ctrl, #'trControl' not 'trcontrol'
                  tuneLength = 5, #'tuneLength' not 'tunelength'
                  trace = FALSE,
                  linout = TRUE) # Ensures linear output for regression
# Display the trained model summary
NN_caret1

# Make predictions on the test set with the trained model NN_caret1
predictions_NN_caret1 <- predict(NN_caret1, test_data_s)
# Rescale predictions back to the original price range
pred_NN_caret1_acts <- predictions_NN_caret1 * (max(train_data$price) - min(train_data$price)) + min(train_data$price)

# Compute performance metrics for NN_caret1 predictions
metrics_NN_caret1 <- postResample(pred = pred_NN_caret1_acts, obs = test_data$price) 

# Print predictions and metrics for model evaluation
print(predictions_NN_caret1)
print(metrics_NN_caret1)

# Plot actual vs. predicted prices for NN_caret1
plot(test_data$price, pred_NN_caret1_acts, 
     xlab = "Actual Price", ylab = "Predicted Price", 
     main = "Model 3 (NN_caret1) Predictions vs Actual",
     col = "darkorange", pch = 16)
abline(a = 0, b = 1, col = "red")  # Adds a reference line for perfect prediction



# Define a tuning grid for the next neural network model NN_caret2
myGrid = expand.grid(size = seq(1, 10, 1), 
                     decay = seq(0.01, 0.2, 0.04))

# Train another neural network model (NN_caret2) with custom tuning grid
NN_caret2 = train(
  price ~., 
  data = train_data_s,
  linout = TRUE,
  method = "nnet",
  tuneGrid = myGrid,
  trControl = ctrl, 
  trace = FALSE
)
# Display the trained model summary
NN_caret2

# Make predictions on the test set with NN_caret2
predictions_NN_caret2 = predict(NN_caret2, test_data_s)
# Rescale predictions to the original price range
pred_NN_caret2_acts = predictions_NN_caret2 * (max(train_data$price) - min(train_data$price)) + min(train_data$price)

# Compute performance metrics for NN_caret2 predictions
metrics_NN_caret2 <- postResample(pred = pred_NN_caret2_acts, obs = test_data$price)
print(metrics_NN_caret2)

# Plot actual vs. predicted prices for NN_caret2
plot(test_data$price, pred_NN_caret2_acts, 
     xlab = "Actual Price", ylab = "Predicted Price", 
     main = "Model 4 (NN_caret2) Predictions vs Actual",
     col = "purple", pch = 16)
abline(a = 0, b = 1, col = "red") 

#3. Compare the results of these models by their model evaluation metrics (RMSE, R-squared, and MAE). Which one is a better model, and why? (Hint: caret package has a function that calculates these three regression measures.
# Print performance metrics for all trained models for comparison
cat("Metrics for NN1:\n")
print(metrics_NN1)
cat("\nMetrics for NN2:\n")
print(metrics_NN2)
cat("\nMetrics for NN_caret1:\n")
print(metrics_NN_caret1)
cat("\nMetrics for NN_caret2:\n")
print(metrics_NN_caret2)

#4. 
# Fit linear regression model using the training data
linear_model <- lm(price ~ ., data = train_data_s)
# Display summary of the linear model
summary(linear_model)
# Predict prices using the linear model on test data
predicted_prices_linear <- predict(linear_model, newdata = test_data_s)
# Calculate RMSE for linear regression predictions
rmse_linear <- sqrt(mean((predicted_prices_linear - test_data$price)^2))
print(paste("RMSE for Linear Regression:", rmse_linear))
# Plot actual vs. predicted prices for linear regression
plot(test_data$price, predicted_prices_linear, 
     main = "Actual vs Predicted Prices - Linear Regression",
     xlab = "Actual Prices",
     ylab = "Predicted Prices",
     pch = 19, col = "blue")
abline(a = 0, b = 1, lty = 2, col = "red") # Add identity line for reference
legend("topright", legend = c("Predictions", "Identity Line"), 
       col = c("blue", "red"), pch = c(19, NA), lty = c(NA, 2))
