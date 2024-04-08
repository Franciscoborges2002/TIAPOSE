# File: forecasting_methods.R

# Function to determine the best forecasting method for a department
determine_best_method <- function(results, department) {
  median_nmae <- c(
    median(results[[department]]$ev_holtwinters),
    median(results[[department]]$ev_arima),
    median(results[[department]]$ev_mlpe),
    median(results[[department]]$ev_nnetar),
    median(results[[department]]$ev_ets),
    median(results[[department]]$ev_lm),
    median(results[[department]]$ev_rf),
    median(results[[department]]$ev_naive)
  )
  methods <- c("Holt-Winters", "ARIMA", "mlpe", "nnetar", "ETS", "LM", "Random Forest", "Naive")
  best_method <- methods[which.min(median_nmae)]
  return(best_method)
}

# Function to determine the best forecasting method globally
determine_global_best_method <- function(results, departments) {
  all_median_nmae <- numeric(length(departments))
  for (i in 1:length(departments)) {
    all_median_nmae[i] <- median(results[[departments[i]]]$ev_holtwinters)
  }
  best_department <- departments[which.min(all_median_nmae)]
  return(determine_best_method(results, best_department))
}

