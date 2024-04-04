library(forecast)

holtwinters_gw <- function(d1, Test, W, S, b, H) {
  H_hw <- holdout(d1, ratio=Test, mode="incremental", iter=b, window=W, increment=S)   
  M_hw <- suppressWarnings(HoltWinters(ts(d1[H_hw$tr], frequency=K))) 
  Pred_hw <- forecast(M_hw, h=length(H_hw$ts))$mean[1:Test] 
  ev_hw <- mmetric(y=d1[H_hw$ts], x=Pred_hw, metric="NMAE", val=YR)
  mae_hw <- mmetric(y=d1[H_hw$ts], x=Pred_hw, metric="MAE")
  rmse_hw <- mmetric(y=d1[H_hw$ts], x=Pred_hw, metric="RMSE")
  r2_hw <- mmetric(y=d1[H_hw$ts], x=Pred_hw, metric="R2")
  
  cat("Holtwinters iter:", b, "TR from:", H_hw$tr[1], "to:", (H_hw$tr[1] + length(H_hw$tr) - 1), "size:", length(H_hw$tr),
      "TS from:", H_hw$ts[1], "to:", H_hw$ts[length(H_hw$ts)], "size:", length(H_hw$ts),
      "nmae:", ev_hw, "mae:", mae_hw, "rmse:", rmse_hw, "r2:", r2_hw, "\n")
  
  return(list(Pred_hw, ev_hw, mae_hw, rmse_hw, r2_hw))
}
