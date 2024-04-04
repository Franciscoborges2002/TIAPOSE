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
d1 <- dados$WSdep1  
L <- length(d1) 
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
D <- CasesSeries(d1,timelags) 
W_mlpe <- W-max(timelags) 
W_lm <- W-max(timelags)

YR <- diff(range(d1)) 


ev_holtwinters <- vector(length=Runs) 
ev_mlpe <- vector(length=Runs) 
ev_arima <- vector(length=Runs) 
ev_nnetar <- vector(length=Runs) 
ev_ets <- vector(length=Runs) 
ev_lm <- vector(length=Runs) 

# Growing window loop
for(b in 1:Runs)  {
  H <- holdout(d1, ratio=Test, mode="incremental", iter=b, window=W, increment=S) 
  
  # HoltWinters
  hw_results <- holtwinters_gw(d1, Test, W, S, b, H)
  Pred_holtwinters <- hw_results[[1]]
  ev_holtwinters[b] <- hw_results[[2]]
  mae_hw <- hw_results[[3]]
  rmse_hw <- hw_results[[4]]
  r2_hw <- hw_results[[5]]
  
  # ARIMA
  arima_results <- arima_gw(d1, Test, W_arima, S, b, H)
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
  nnetar_results <- nnetar_gw(d1, Test, W_nnetar, S, b)
  Pred_nnetar <- nnetar_results[[1]]
  ev_nnetar[b] <- nnetar_results[[2]]
  mae_nnetar <- nnetar_results[[3]]
  rmse_nnetar <- nnetar_results[[4]]
  r2_nnetar <- nnetar_results[[5]]
  
  # ETS
  ets_results <- ets_gw(d1, Test, W_ets, S, b)
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
  
  # Plot graphs
  mgraph(d1[H$ts], Pred_holtwinters, graph="REG", Grid=10,
         col=c("black","blue","green","purple","orange","red","pink"),
         leg=list(pos="topleft",leg=c("target","HW pred.","arima","nn","ets","mlpe","lm")))
  lines(Pred_arima, pch=19, cex=0.5, type="b", col="green")
  lines(Pred_mlpe, pch=19, cex=0.5, type="b", col="red")
  lines(Pred_nnetar, pch=19, cex=0.5, type="b", col="blue")
  lines(Pred_ets, pch=19, cex=0.5, type="b", col="orange")
  lines(Pred_lm, pch=19, cex=0.5, type="b", col="pink")
  mpause() # wait for enter
  
 
}

# Show median of each model
cat("Median NMAE values for Holt-Winters, mlpe, ARIMA, nnetar, ETS, and LM:\n")
cat("Holt-Winters median NMAE:", median(ev_holtwinters), "\n")
cat("ARIMA median NMAE:", median(ev_arima), "\n")
cat("mlpe median NMAE:", median(ev_mlpe), "\n")
cat("nnetar median NMAE:", median(ev_nnetar), "\n")
cat("ETS median NMAE:", median(ev_ets), "\n")
cat("LM median NMAE:", median(ev_lm), "\n\n")

# Show median of each model for MAE
cat("Median MAE values for Holt-Winters, mlpe, ARIMA, nnetar, ETS, and LM:\n")
cat("Holt-Winters median MAE:", median(mae_hw), "\n")
cat("ARIMA median MAE:", median(mae_arima), "\n")
cat("mlpe median MAE:", median(mae_mlpe), "\n")
cat("nnetar median MAE:", median(mae_nnetar), "\n")
cat("ETS median MAE:", median(mae_ets), "\n")
cat("LM median MAE:", median(mae_lm), "\n\n")

# Show median of each model for RMSE
cat("Median RMSE values for Holt-Winters, mlpe, ARIMA, nnetar, ETS, and LM:\n")
cat("Holt-Winters median RMSE:", median(rmse_hw), "\n")
cat("ARIMA median RMSE:", median(rmse_arima), "\n")
cat("mlpe median RMSE:", median(rmse_mlpe), "\n")
cat("nnetar median RMSE:", median(rmse_nnetar), "\n")
cat("ETS median RMSE:", median(rmse_ets), "\n")
cat("LM median RMSE:", median(rmse_lm), "\n\n")

# Show median of each model for R^2
cat("Median R^2 values for Holt-Winters, mlpe, ARIMA, nnetar, ETS, and LM:\n")
cat("Holt-Winters median R^2:", median(r2_hw), "\n")
cat("ARIMA median R^2:", median(r2_arima), "\n")
cat("mlpe median R^2:", median(r2_mlpe), "\n")
cat("nnetar median R^2:", median(r2_nnetar), "\n")
cat("ETS median R^2:", median(r2_ets), "\n")
cat("LM median R^2:", median(r2_lm), "\n\n")

