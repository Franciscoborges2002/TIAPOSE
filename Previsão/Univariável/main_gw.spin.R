# Main file

# Load necessary libraries
library(forecast)
library(rminer)

# Source other files
source("holtwinters_gw.R")
source("arima_gw.R")
source("mlpe_gw.R")
source("nnetar_gw.R")
source("ets_gw.R")
source("lm_gw.R")

# Read data
cat("read walmart time series:")
dados <- read.csv("walmart.csv")

# Extract department time series data
departments <- c("WSdep1", "WSdep2")#, "WSdep3", "WSdep4") # Update with actual department names
results <- list()

for (dep in departments) {
  cat("\nProcessing department:", dep, "\n")
  
  d <- dados[[dep]]
  L <- length(d) 
  K <- 4 
  
  # Set parameters
  Test <- K 
  S <- round(K/1) 
  Runs <- 8 
  
  W <- (L-Test)-(Runs-1)*S 
  W_arima <- (L-Test)-(Runs-1)*S
  W_nnetar <- (L-Test)-(Runs-1)*S
  W_ets <- (L-Test)-(Runs-1)*S
  
  timelags <- c(1,4,8) 
  D <- CasesSeries(d,timelags) 
  W_mlpe <- W-max(timelags) 
  W_lm <- W-max(timelags)
  
  YR <- diff(range(d)) 
  
  ev_holtwinters <- vector(length=Runs) 
  ev_mlpe <- vector(length=Runs) 
  ev_arima <- vector(length=Runs) 
  ev_nnetar <- vector(length=Runs) 
  ev_ets <- vector(length=Runs) 
  ev_lm <- vector(length=Runs) 
  
  # Growing window loop
  for(b in 1:Runs)  {
    H <- holdout(d, ratio=Test, mode="incremental", iter=b, window=W, increment=S) 
    
    # HoltWinters
    hw_results <- holtwinters_gw(d, Test, W, S, b, H)
    Pred_holtwinters <- hw_results[[1]]
    ev_holtwinters[b] <- hw_results[[2]]
    mae_hw <- hw_results[[3]]
    rmse_hw <- hw_results[[4]]
    r2_hw <- hw_results[[5]]
    
    # ARIMA
    arima_results <- arima_gw(d, Test, W_arima, S, b, H)
    Pred_arima <- arima_results[[1]]
    ev_arima[b] <- arima_results[[2]]
    mae_arima <- arima_results[[3]]
    rmse_arima <- arima_results[[4]]
    r2_arima <- arima_results[[5]]
    
    # mlpe
    mlpe_results <- mlpe_gw(D, Test, W_mlpe, S, b)
    Pred_mlpe <- mlpe_results[[1]]
    ev_mlpe[b] <- mlpe_results[[2]]
    mae_mlpe <- mlpe_results[[3]]
    rmse_mlpe <- mlpe_results[[4]]
    r2_mlpe <- mlpe_results[[5]]
    
    # nnetar
    nnetar_results <- nnetar_gw(d, Test, W_nnetar, S, b)
    Pred_nnetar <- nnetar_results[[1]]
    ev_nnetar[b] <- nnetar_results[[2]]
    mae_nnetar <- nnetar_results[[3]]
    rmse_nnetar <- nnetar_results[[4]]
    r2_nnetar <- nnetar_results[[5]]
    
    # ETS
    ets_results <- ets_gw(d, Test, W_ets, S, b)
    Pred_ets <- ets_results[[1]]
    ev_ets[b] <- ets_results[[2]]
    mae_ets <- ets_results[[3]]
    rmse_ets <- ets_results[[4]]
    r2_ets <- ets_results[[5]]
    
    # LM
    lm_results <- lm_gw(D, Test, W_lm, S, b)
    Pred_lm <- lm_results[[1]]
    ev_lm[b] <- lm_results[[2]]
    mae_lm <- lm_results[[3]]
    rmse_lm <- lm_results[[4]]
    r2_lm <- lm_results[[5]]
    
    # Gráfico para o XGBoost
    plot(d[H$ts], type="l", col="black", xlab="Time", ylab="Value", main="XGBoost Forecast vs Actuals")
    lines(Pred_holtwinters, type="l", col="blue")
    legend("topleft", legend=c("Actual", "XGBoost Forecast"), col=c("black", "blue"), lty=1:2, cex=0.8)
    
    # Aguardar um pouco para permitir a observação do gráfico (opcional)
    Sys.sleep(2)
    
    # Gráfico para o LM
    plot(d[H$ts], type="l", col="black", xlab="Time", ylab="Value", main="LM Forecast vs Actuals")
    lines(Pred_lm, type="l", col="red")
    legend("topleft", legend=c("Actual", "LM Forecast"), col=c("black", "red"), lty=1:2, cex=0.8)
    
    # Aguardar um pouco para permitir a observação do gráfico (opcional)
    Sys.sleep(2)
    
  }
    

  # Store results  armazenar os resultados das análises para cada departamento. Sem essa parte do código, os resultados não seriam armazenados e não haveria uma maneira sistemática de aceder e analisar as métricas de desempenho para cada modelo e departamento.
  results[[dep]] <- list(
    ev_holtwinters = ev_holtwinters,
    ev_arima = ev_arima,
    ev_mlpe = ev_mlpe,
    ev_nnetar = ev_nnetar,
    ev_ets = ev_ets,
    ev_lm = ev_lm,
    mae_hw = mae_hw,
    mae_arima = mae_arima,
    mae_mlpe = mae_mlpe,
    mae_nnetar = mae_nnetar,
    mae_ets = mae_ets,
    mae_lm = mae_lm,
    rmse_hw = rmse_hw,
    rmse_arima = rmse_arima,
    rmse_mlpe = rmse_mlpe,
    rmse_nnetar = rmse_nnetar,
    rmse_ets = rmse_ets,
    rmse_lm = rmse_lm,
    r2_hw = r2_hw,
    r2_arima = r2_arima,
    r2_mlpe = r2_mlpe,
    r2_nnetar = r2_nnetar,
    r2_ets = r2_ets,
    r2_lm = r2_lm
  )
}

# Show median of each model
for (dep in departments) {
  cat("Median NMAE values for", dep, "department:\n")
  cat("Holt-Winters median NMAE:", median(results[[dep]]$ev_holtwinters), "\n")
  cat("ARIMA median NMAE:", median(results[[dep]]$ev_arima), "\n")
  cat("mlpe median NMAE:", median(results[[dep]]$ev_mlpe), "\n")
  cat("nnetar median NMAE:", median(results[[dep]]$ev_nnetar), "\n")
  cat("ETS median NMAE:", median(results[[dep]]$ev_ets), "\n")
  cat("LM median NMAE:", median(results[[dep]]$ev_lm), "\n\n")
}

# Show median of each model for MAE
for (dep in departments) {
  cat("Median MAE values for", dep, "department:\n")
  cat("Holt-Winters median MAE:", median(results[[dep]]$mae_hw), "\n")
  cat("ARIMA median MAE:", median(results[[dep]]$mae_arima), "\n")
  cat("mlpe median MAE:", median(results[[dep]]$mae_mlpe), "\n")
  cat("nnetar median MAE:", median(results[[dep]]$mae_nnetar), "\n")
  cat("ETS median MAE:", median(results[[dep]]$mae_ets), "\n")
  cat("LM median MAE:", median(results[[dep]]$mae_lm), "\n\n")
}

# Show median of each model for RMSE
for (dep in departments) {
  cat("Median RMSE values for", dep, "department:\n")
  cat("Holt-Winters median RMSE:", median(results[[dep]]$rmse_hw), "\n")
  cat("ARIMA median RMSE:", median(results[[dep]]$rmse_arima), "\n")
  cat("mlpe median RMSE:", median(results[[dep]]$rmse_mlpe), "\n")
  cat("nnetar median RMSE:", median(results[[dep]]$rmse_nnetar), "\n")
  cat("ETS median RMSE:", median(results[[dep]]$rmse_ets), "\n")
  cat("LM median RMSE:", median(results[[dep]]$rmse_lm), "\n\n")
}

# Show median of each model for R^2
for (dep in departments) {
  cat("Median R^2 values for", dep, "department:\n")
  cat("Holt-Winters median R^2:", median(results[[dep]]$r2_hw), "\n")
  cat("ARIMA median R^2:", median(results[[dep]]$r2_arima), "\n")
  cat("mlpe median R^2:", median(results[[dep]]$r2_mlpe), "\n")
  cat("nnetar median R^2:", median(results[[dep]]$r2_nnetar), "\n")
  cat("ETS median R^2:", median(results[[dep]]$r2_ets), "\n")
  cat("LM median R^2:", median(results[[dep]]$r2_lm), "\n\n")
}

