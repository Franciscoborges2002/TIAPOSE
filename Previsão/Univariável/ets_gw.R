library(forecast)

ets_gw <- function(d1, Test, W_ets, S, b) {
  H_ets <- holdout(d1, ratio=Test, mode="incremental", iter=b, window=W_ets, increment=S) 
  M_ets <- ets(ts(d1[H_ets$tr], frequency=K)) 
  Pred_ets <- forecast(M_ets, h=length(H_ets$ts))$mean[1:Test] 
  ev_ets <- mmetric(y=d1[H_ets$ts], x=Pred_ets, metric="NMAE", val=YR)
  mae_ets <- mmetric(y=d1[H_ets$ts], x=Pred_ets, metric="MAE")
  rmse_ets <- mmetric(y=d1[H_ets$ts], x=Pred_ets, metric="RMSE")
  # Calculating R2
  r2_ets <- cor(d1[H_ets$ts], Pred_ets)^2
  
  cat("ets iter:", b, "TR from:", H_ets$tr[1], "to:", (H_ets$tr[1] + length(H_ets$tr) - 1), "size:", length(H_ets$tr),
      "TS from:", H_ets$ts[1], "to:", H_ets$ts[length(H_ets$ts)], "size:", length(H_ets$ts),
      "nmae:", ev_ets, "mae:", mae_ets, "rmse:", rmse_ets, "r2:", r2_ets, "\n")
  
  return(list(Pred_ets, ev_ets, mae_ets, rmse_ets, r2_ets))
}
