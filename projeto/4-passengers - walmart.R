
# 4-passengers.R: script demonstration of growing window and rolling window evaluations.
library(forecast) # access forecast functions -> HoltWinters, forecast
library(rminer) # access rminer functions -> CasesSeries, fit, lforecast, mmetric, mgraph, ...

# setwd() # adjust working directory if needed.

# read data:
cat("read walmart time series:")

dados <- read.csv("walmart.csv")
d1 <- dados$WSdep1  # Vamos usar a coluna "Fuel_Price" como nossa série temporal

L=length(d1) # size of the time series, 144
K=4 # assumption for the seasonal period: test also acf(d1S)

print("incremental (growing) window training demonstration:")

Test=K # H, the number of multi-ahead steps, adjust if needed
S=round(K/3) # step jump: set in this case to 4 months, a quarter
Runs=8 # number of growing window iterations, adjust if needed

# forecast:
W=(L-Test)-(Runs-1)*S # initial training window size for the ts space (forecast methods)

# rminer:
timelags=c(1,4,8) # 1 previous month, 12 and 13 previous year months, you can test other combinations, such as 1:13
D=CasesSeries(d1,timelags) # note: nrow(D) is smaller by max timelags than length(d1)
W2=W-max(timelags) # initial training window size for the D space (CasesSeries, rminer methods)

YR=diff(range(d1)) # global Y range, use the same range for the NMAE calculation in all iterations

ev_holtwinters=vector(length=Runs) # error vector for "HoltWinters"
ev_mlpe=vector(length=Runs) # error vector for "mlpe"
ev_lm <- vector(length = Runs)  # vetor de erro para "lm"
ev_nn <- vector(length = Runs)  # vetor de erro para "nn"
ev_arima <- vector(length = Runs)  # vetor de erro para "arima"
ev_ets <- vector(length = Runs)  # vetor de erro para "ets"


# growing window:
for(b in 1:Runs)  # cycle of the incremental window training (growing window)
{
  # Holt-Winters:
  H_holtwinters_train <- holdout(d1, ratio = Test, mode = "incremental", iter = b, window = W, increment = S)
  dtr <- ts(d1[H_holtwinters_train$tr], frequency = K)
  M_holtwinters <- suppressWarnings(HoltWinters(dtr))
  
  H_holtwinters_test <- holdout(d1, ratio = Test, mode = "incremental", iter = b, window = W, increment = S)
  Pred_holtwinters <- forecast(M_holtwinters, h = length(H_holtwinters_test$ts))$mean[1:Test]
  ev_holtwinters[b] <- mmetric(y = d1[H_holtwinters_test$ts], x = Pred_holtwinters, metric = "NMAE", val = YR)
  
  # auto.arima:
  H_arima <- holdout(d1, ratio = Test, mode = "incremental", iter = b, window = W, increment = S)
  dtr_arima <- ts(d1[H_arima$tr], frequency = K)
  M_arima <- suppressWarnings(auto.arima(dtr_arima))
  Pred_arima <- forecast(M_arima, h = length(H_arima$ts))$mean[1:Test]
  ev_arima[b] <- mmetric(y = d1[H_arima$ts], x = Pred_arima, metric = "NMAE", val = YR)
  
  # nnetar:
  H_nn <- holdout(d1, ratio = Test, mode = "incremental", iter = b, window = W, increment = S)
  dtr_nn <- ts(d1[H_nn$tr], frequency = K)
  M_nn <- suppressWarnings(nnetar(dtr_nn))
  Pred_nn <- forecast(M_nn, h = length(H_nn$ts))$mean[1:Test]
  ev_nn[b] <- mmetric(y = d1[H_nn$ts], x = Pred_nn, metric = "NMAE", val = YR)
  
  # ets:
  H_ets <- holdout(d1, ratio = Test, mode = "incremental", iter = b, window = W, increment = S)
  dtr_ets <- ts(d1[H_ets$tr], frequency = K)
  M_ets <- suppressWarnings(ets(dtr_ets))
  Pred_ets <- forecast(M_ets, h = length(H_ets$ts))$mean[1:Test]
  ev_ets[b] <- mmetric(y = d1[H_ets$ts], x = Pred_ets, metric = "NMAE", val = YR)
  
  # mlpe:
  H_mlpe <- holdout(D$y, ratio = Test, mode = "incremental", iter = b, window = W2, increment = S)
  M_mlpe <- fit(y ~ ., D[H_mlpe$tr, ], model = "mlpe")
  Pred_mlpe <- lforecast(M_mlpe, D, start = (length(H_mlpe$tr) + 1), Test)
  ev_mlpe[b] <- mmetric(y = d1[H_mlpe$ts], x = Pred_mlpe, metric = "NMAE", val = YR)
  
  # lm:
  H_lm <- holdout(D$y, ratio = Test, mode = "incremental", iter = b, window = W2, increment = S)
  M_lm <- fit(y ~ ., D[H_lm$tr, ], model = "lm")
  Pred_lm <- lforecast(M_lm, D, start = (length(H_lm$tr) + 1), Test)
  ev_lm[b] <- mmetric(y = d1[H_lm$ts], x = Pred_lm, metric = "NMAE", val = YR)
  
  
  cat("iter:",b,"TR from:",trinit,"to:",(trinit+length(H$tr)-1),"size:",length(H$tr),
      "TS from:",H$ts[1],"to:",H$ts[length(H$ts)],"size:",length(H$ts),
      "nmae:",ev_holtwinters[b],",",ev_arima[b],",",ev_nn[b],",",ev_ets[b],",",ev_mlpe[b],",",ev_lm[b],"\n")
  mgraph(d1[H$ts],Pred_holtwinters,graph="REG",Grid=10,col=c("black","blue","red","green","orange","purple","brown"),leg=list(pos="topleft",leg=c("target","HW pred.","Arima pred.","nn pred.","ets pred.","mlpe","lm pred")))
  lines(Pred_arima, pch = 19, cex = 0.5, type = "b", col = "red")
  lines(Pred_nn, pch = 19, cex = 0.5, type = "b", col = "green")
  lines(Pred_ets, pch = 19, cex = 0.5, type = "b", col = "orange")
  lines(Pred_mlpe, pch = 19, cex = 0.5, type = "b", col = "purple")
  lines(Pred_lm, pch = 19, cex = 0.5, type = "b", col = "brown")
  mpause() # wait for enter
  
  }
# end of cycle

# show median of ev_holtwinters and ev_mlpe
cat("median NMAE values for HW and mlpe:\n")
cat("Holt-Winters median NMAE:",median(ev_holtwinters),"\n")
cat("mlpe median NMAE:",median(ev_mlpe),"\n")
cat("lm median NMAE:",median(ev_lm), "\n")
cat("nn median NMAE:",median(ev_nn),"\n")
cat("arima median NMAE:",median(ev_arima),"\n")
cat("ets median NMAE:",median(ev_ets),"\n")
mpause() # wait for enter

# Previsões da última iteração para todos os modelos:
mgraph(d1[H_holtwinters$ts], Pred_holtwinters, graph = "REG", Grid = 10, col = c("black", "blue", "red", "green", "orange", "purple", "brown"), leg = list(pos = "topleft", leg = c("target", "HW pred.", "ARIMA pred.", "nnetar pred.", "ETS pred.", "mlpe pred.", "lm pred.")))
lines(Pred_arima, pch = 19, cex = 0.5, type = "b", col = "red")
lines(Pred_nn, pch = 19, cex = 0.5, type = "b", col = "green")
lines(Pred_ets, pch = 19, cex = 0.5, type = "b", col = "orange")
lines(Pred_mlpe, pch = 19, cex = 0.5, type = "b", col = "purple")
lines(Pred_lm, pch = 19, cex = 0.5, type = "b", col = "brown")

