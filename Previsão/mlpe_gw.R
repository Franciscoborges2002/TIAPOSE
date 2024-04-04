mlpe_gw <- function(D, Test, W_mlpe, S, b) {
  H_mlpe <- holdout(D$y, ratio=Test, mode="incremental", iter=b, window=W_mlpe, increment=S)   
  M_mlpe <- fit(y~., D[H_mlpe$tr,], model="mlpe") 
  Pred_mlpe <- lforecast(M_mlpe, D, start=(length(H_mlpe$tr)+1), Test) 
  ev_mlpe <- mmetric(y=D$y[H_mlpe$ts], x=Pred_mlpe, metric="NMAE", val=YR)
  mae_mlpe <- mmetric(y=D$y[H_mlpe$ts], x=Pred_mlpe, metric="MAE")
  rmse_mlpe <- mmetric(y=D$y[H_mlpe$ts], x=Pred_mlpe, metric="RMSE")
  # Calculating R2
  r2_mlpe <- cor(D$y[H_mlpe$ts], Pred_mlpe)^2
  
  cat("mlpe iter:", b, "TR from:", H_mlpe$tr[1], "to:", (H_mlpe$tr[1] + length(H_mlpe$tr) - 1), "size:", length(H_mlpe$tr),
      "TS from:", H_mlpe$ts[1], "to:", H_mlpe$ts[length(H_mlpe$ts)], "size:", length(H_mlpe$ts),
      "nmae:", ev_mlpe, "mae:", mae_mlpe, "rmse:", rmse_mlpe, "r2:", r2_mlpe, "\n")
  
  return(list(Pred_mlpe, ev_mlpe, mae_mlpe, rmse_mlpe, r2_mlpe))
}
