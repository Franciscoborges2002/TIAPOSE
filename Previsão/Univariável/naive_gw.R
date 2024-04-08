library(rminer)

naive_gw <- function(D, Test, W_naive, S, b) {
  H_naive <- holdout(D$y, ratio=Test, mode="incremental", iter=b, window=W_naive, increment=S)   
  Pred_naive <- rep(mean(D[H_naive$tr, "y"]), length(H_naive$ts))
  ev_naive <- mmetric(y=D$y[H_naive$ts], x=Pred_naive, metric="NMAE", val=YR)
  mae_naive <- mmetric(y=D$y[H_naive$ts], x=Pred_naive, metric="MAE")
  rmse_naive <- mmetric(y=D$y[H_naive$ts], x=Pred_naive, metric="RMSE")
  # Calculating R2
  r2_naive <- cor(D$y[H_naive$ts], Pred_naive)^2
  
  cat("naive iter:", b, "TR from:", H_naive$tr[1], "to:", (H_naive$tr[1] + length(H_naive$tr) - 1), "size:", length(H_naive$tr),
      "TS from:", H_naive$ts[1], "to:", H_naive$ts[length(H_naive$ts)], "size:", length(H_naive$ts),
      "nmae:", ev_naive, "mae:", mae_naive, "rmse:", rmse_naive, "r2:", r2_naive, "\n")
  
  return(list(Pred_naive, ev_naive, mae_naive, rmse_naive, r2_naive))
}
