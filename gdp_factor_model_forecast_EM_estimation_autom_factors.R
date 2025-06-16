# Load required libraries
library(readr)
library(dplyr)
library(KFAS)
library(ggplot2)
library(tidyr)

# Load and preprocess GDP data
data <- read_csv("current_FRED_QD_Quarterly.csv", show_col_types = FALSE)
data$sasdate <- as.Date(data$sasdate, format = "%m/%d/%Y")

# Compute log-differenced GDP
gdp_diff <- diff(log(data$GDPC1))
dates <- data$sasdate[-1]

# Prepare macroeconomic panel for PCA
macro_raw <- data %>% select(-sasdate, -GDPC1)

# Keep only variables that are strictly positive
macro_clean <- macro_raw %>%
  select(where(~ all(. > 0, na.rm = TRUE)))

# Log-difference and standardize
macro_diff <- macro_clean %>%
  mutate(across(everything(), ~ log(.) - lag(log(.)))) %>%
  drop_na()

macro_scaled <- macro_diff %>%
  select(where(~ sd(., na.rm = TRUE) > 1e-6)) %>%
  scale()

# Perform PCA and choose k based on 85% variance explained
pca_result <- prcomp(macro_scaled, center = FALSE, scale. = FALSE)
explained_var <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cum_var <- cumsum(explained_var)
k <- which(cum_var >= 0.85)[1]  # minimum k to explain 85% variance

# Extract factors
factors <- pca_result$x[, 1:k]
colnames(factors) <- paste0("F", 1:k)

# Align GDP and factors
gdp_aligned <- tail(gdp_diff, n = nrow(factors))
dates_aligned <- tail(dates, n = nrow(factors))

# Create data frame for KFAS model
df_model <- data.frame(Date = dates_aligned, GDP = gdp_aligned, factors)
df_model_ts <- ts(df_model[, -1], start = c(1959, 2), frequency = 4)

# Create regression formula dynamically
f_names <- paste0("F", 1:k)
fmla <- as.formula(paste("GDP ~", paste(f_names, collapse = " + ")))

# Define state space model with estimated H
SSM <- SSModel(fmla, data = df_model_ts,
               H = NA, distribution = "gaussian")

# Fit using EM
fit <- fitSSM(SSM, inits = rep(0.1, k + 1), method = "BFGS")

# Kalman smoothing
kfs_result <- KFS(fit$model)
fitted_vals <- as.numeric(kfs_result$muhat)
actual_vals <- as.numeric(df_model_ts[, "GDP"])

# Create data frame for plotting
df_plot <- data.frame(
  Date = as.Date(df_model$Date),
  Actual = actual_vals,
  Fitted = fitted_vals
)

# Plot actual vs fitted
ggplot(df_plot, aes(x = Date)) +
  geom_line(aes(y = Actual), color = "black", linewidth = 1) +
  geom_line(aes(y = Fitted), color = "blue", linetype = "dotted", linewidth = 1) +
  labs(
    title = paste("EM-based Kalman Filter Forecast (", k, " PCA Factors)", sep = ""),
    y = "Log-Diff GDP",
    x = "Date"
  ) +
  theme_minimal()

# Accuracy metrics
rmse <- sqrt(mean((fitted_vals - actual_vals)^2))
mae <- mean(abs(fitted_vals - actual_vals))

cat("EM-Kalman Forecast Accuracy (In-Sample):\n")
cat(sprintf("RMSE: %.4f\n", rmse))
cat(sprintf("MAE : %.4f\n", mae))

# Compute and plot residuals
residuals <- actual_vals - fitted_vals
df_resid <- data.frame(Date = df_model$Date, Residuals = residuals)

ggplot(df_resid, aes(x = Date, y = Residuals)) +
  geom_line(color = "darkred") +
  labs(title = "In-Sample Residuals (EM-Kalman PCA Forecast)",
       x = "Date", y = "Residual") +
  theme_minimal()

acf(residuals, main = "ACF of In-Sample Residuals")

