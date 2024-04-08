library(forecast)

arima_gw <- function(d1, Test, W_arima, S, b, H) {
  H_arima <- holdout(d1, ratio=Test, mode="incremental", iter=b, window=W_arima, increment=S) 
  dtr <- ts(d1[H_arima$tr], frequency=K)
  M_arima <- auto.arima(dtr) 
  Pred_arima <- forecast(M_arima, h=length(H_arima$ts))$mean[1:Test] 
  ev_arima <- mmetric(y=d1[H_arima$ts], x=Pred_arima, metric="NMAE", val=YR)
  mae_arima <- mmetric(y=d1[H_arima$ts], x=Pred_arima, metric="MAE")
  rmse_arima <- mmetric(y=d1[H_arima$ts], x=Pred_arima, metric="RMSE")
  # Calculating R2
  r2_arima <- cor(d1[H_arima$ts], Pred_arima)^2
  
  cat("arima iter:", b, "TR from:", H_arima$tr[1], "to:", (H_arima$tr[1] + length(H_arima$tr) - 1), "size:", length(H_arima$tr),
      "TS from:", H_arima$ts[1], "to:", H_arima$ts[length(H_arima$ts)], "size:", length(H_arima$ts),
      "nmae:", ev_arima, "mae:", mae_arima, "rmse:", rmse_arima, "r2:", r2_arima, "\n")
  
  return(list(Pred_arima, ev_arima, mae_arima, rmse_arima, r2_arima))
}
