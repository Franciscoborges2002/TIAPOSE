library(forecast)

nnetar_gw <- function(d1, Test, W_nnetar, S, b) {
  H_nnetar <- holdout(d1, ratio=Test, mode="incremental", iter=b, window=W_nnetar, increment=S) 
  M_nnetar <- nnetar(ts(d1[H_nnetar$tr], frequency=K)) 
  Pred_nnetar <- forecast(M_nnetar, h=length(H_nnetar$ts))$mean[1:Test] 
  ev_nnetar <- mmetric(y=d1[H_nnetar$ts], x=Pred_nnetar, metric="NMAE", val=YR)
  mae_nnetar <- mmetric(y=d1[H_nnetar$ts], x=Pred_nnetar, metric="MAE")
  rmse_nnetar <- mmetric(y=d1[H_nnetar$ts], x=Pred_nnetar, metric="RMSE")
  # Calculating R2
  r2_nnetar <- cor(d1[H_nnetar$ts], Pred_nnetar)^2
  
  cat("nnetar iter:", b, "TR from:", H_nnetar$tr[1], "to:", (H_nnetar$tr[1] + length(H_nnetar$tr) - 1), "size:", length(H_nnetar$tr),
      "TS from:", H_nnetar$ts[1], "to:", H_nnetar$ts[length(H_nnetar$ts)], "size:", length(H_nnetar$ts),
      "nmae:", ev_nnetar, "mae:", mae_nnetar, "rmse:", rmse_nnetar, "r2:", r2_nnetar, "\n")
  
  return(list(Pred_nnetar, ev_nnetar, mae_nnetar, rmse_nnetar, r2_nnetar))
}
