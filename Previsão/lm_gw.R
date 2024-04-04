library(rminer)

lm_gw <- function(D, Test, W_lm, S, b) {
  H_lm <- holdout(D$y, ratio=Test, mode="incremental", iter=b, window=W_lm, increment=S)   
  M_lm <- fit(y~., D[H_lm$tr,], model="lm") 
  Pred_lm <- lforecast(M_lm, D, start=(length(H_lm$tr)+1), Test) 
  ev_lm <- mmetric(y=D$y[H_lm$ts], x=Pred_lm, metric="NMAE", val=YR)
  mae_lm <- mmetric(y=D$y[H_lm$ts], x=Pred_lm, metric="MAE")
  rmse_lm <- mmetric(y=D$y[H_lm$ts], x=Pred_lm, metric="RMSE")
  # Calculating R2
  r2_lm <- cor(D$y[H_lm$ts], Pred_lm)^2
  
  cat("lm iter:", b, "TR from:", H_lm$tr[1], "to:", (H_lm$tr[1] + length(H_lm$tr) - 1), "size:", length(H_lm$tr),
      "TS from:", H_lm$ts[1], "to:", H_lm$ts[length(H_lm$ts)], "size:", length(H_lm$ts),
      "nmae:", ev_lm, "mae:", mae_lm, "rmse:", rmse_lm, "r2:", r2_lm, "\n")
  
  return(list(Pred_lm, ev_lm, mae_lm, rmse_lm, r2_lm))
}
