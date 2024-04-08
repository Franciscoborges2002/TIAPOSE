library(rminer)
library(randomForest)

rf_gw <- function(D, Test, W_rf, S, b) {
  H_rf <- holdout(D$y, ratio=Test, mode="incremental", iter=b, window=W_rf, increment=S)   
  M_rf <- randomForest(y ~ ., data = D[H_rf$tr,]) 
  Pred_rf <- predict(M_rf, newdata = D[H_rf$ts,]) 
  ev_rf <- mmetric(y=D$y[H_rf$ts], x=Pred_rf, metric="NMAE", val=YR)
  mae_rf <- mmetric(y=D$y[H_rf$ts], x=Pred_rf, metric="MAE")
  rmse_rf <- mmetric(y=D$y[H_rf$ts], x=Pred_rf, metric="RMSE")
  # Calculating R2
  r2_rf <- cor(D$y[H_rf$ts], Pred_rf)^2
  
  cat("rf iter:", b, "TR from:", H_rf$tr[1], "to:", (H_rf$tr[1] + length(H_rf$tr) - 1), "size:", length(H_rf$tr),
      "TS from:", H_rf$ts[1], "to:", H_rf$ts[length(H_rf$ts)], "size:", length(H_rf$ts),
      "nmae:", ev_rf, "mae:", mae_rf, "rmse:", rmse_rf, "r2:", r2_rf, "\n")
  
  return(list(Pred_rf, ev_rf, mae_rf, rmse_rf, r2_rf))
}
