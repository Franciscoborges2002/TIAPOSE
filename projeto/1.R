library(forecast) # access forecast functions -> HoltWinters, forecast
library(rminer) # access rminer functions -> CasesSeries, fit, lforecast, mmetric, mgraph, ...

# setwd() # adjust working directory if needed.

# read data:
cat("read walmart time series:")

dados <- read.csv("walmart.csv")
d1 <- dados$Fuel_Price  # Vamos usar a coluna "Fuel_Price" como nossa série temporal
L=length(d1) # size of the time series, 144
K=4 # assumption for the seasonal period: test also acf(d1S)

print("incremental (growing) window training demonstration:")

Test=K # H, the number of multi-ahead steps, adjust if needed
S=round(K/5) # step jump: set in this case to 4 months, a quarter
Runs=8 # number of growing window iterations, adjust if needed

# forecast:
W=(L-Test)-(Runs-1)*S # initial training window size for the ts space (forecast methods)
W_arima=(L-Test)-(Runs-1)*S
W_nnetar=(L-Test)-(Runs-1)*S
W_ets=(L-Test)-(Runs-1)*S


# rminer:
timelags=c(1,4,8) # 1 previous month, 12 and 13 previous year months, you can test other combinations, such as 1:13
D=CasesSeries(d1,timelags) # note: nrow(D) is smaller by max timelags than length(d1)
W2=W-max(timelags) # initial training window size for the D space (CasesSeries, rminer methods)
W_lm=W-max(timelags)

YR=diff(range(d1)) # global Y range, use the same range for the NMAE calculation in all iterations

ev=vector(length=Runs) # error vector for "HoltWinters"
ev2=vector(length=Runs) # error vector for "mlpe"
ev_arima = vector(length=Runs) # error vector for ARIMA
ev_nnetar = vector(length=Runs) # error vector for nnetar
ev_ets = vector(length=Runs) # error vector for ETS
ev_lm = vector(length=Runs) # error vector for LM

# growing window:
for(b in 1:Runs)  # cycle of the incremental window training (growing window)
{
  # HoltWinters
  H=holdout(d1,ratio=Test,mode="incremental",iter=b,window=W,increment=S)   
  trinit=H$tr[1]
  dtr=ts(d1[H$tr],frequency=K) # create ts object, note that there is no start argument (for simplicity of the code)
  M=suppressWarnings(HoltWinters(dtr)) # create forecasting model
  Pred=forecast(M,h=length(H$ts))$mean[1:Test] # multi-step ahead forecasts
  ev[b]=mmetric(y=d1[H$ts],x=Pred,metric="NMAE",val=YR)
  
  # ARIMA
  H_arima=holdout(d1,ratio=Test,mode="incremental",iter=b,window=W_arima,increment=S) 
  M_arima = auto.arima(dtr) # Fit ARIMA model
  Pred_arima = forecast(M_arima,h=length(H_arima$ts))$mean[1:Test] # Forecast
  ev_arima[b] = mmetric(y = d1[H$ts], x = Pred_arima, metric = "NMAE", val = YR)
  
  # nnetar
  H_nnetar=holdout(d1,ratio=Test,mode="incremental",iter=b,window=W_nnetar,increment=S) 
  M_nnetar = nnetar(dtr) # Fit nnetar model
  Pred_nnetar = forecast(M_nnetar,h=length(H_nnetar$ts))$mean[1:Test] # Forecast
  ev_nnetar[b] = mmetric(y = d1[H$ts], x = Pred_nnetar, metric = "NMAE", val = YR)
  
  # ETS
  H_ets=holdout(d1,ratio=Test,mode="incremental",iter=b,window=W_ets,increment=S) 
  M_ets = ets(dtr) # Fit ETS model
  Pred_ets = forecast(M_ets,h=length(H_ets$ts))$mean[1:Test]# Forecast
  ev_ets[b] = mmetric(y = d1[H$ts], x = Pred_ets, metric = "NMAE", val = YR)
  
  # mlpe
  H2=holdout(D$y,ratio=Test,mode="incremental",iter=b,window=W2,increment=S)   
  M2=fit(y~.,D[H2$tr,],model="mlpe") # create forecasting model
  Pred2=lforecast(M2,D,start=(length(H2$tr)+1),Test) # multi-step ahead forecasts
  ev2[b]=mmetric(y=d1[H$ts],x=Pred2,metric="NMAE",val=YR)
  
  # LM
  H_lm=holdout(D$y,ratio=Test,mode="incremental",iter=b,window=W_lm,increment=S)   
  M_lm=fit(y~.,D[H_lm$tr,],model="lm") # create forecasting model
  Pred_lm=lforecast(M_lm,D,start=(length(H_lm$tr)+1),Test) # multi-step ahead forecasts
  ev_lm[b]=mmetric(y=d1[H$ts],x=Pred_lm,metric="NMAE",val=YR)
  
  
  cat("iter:", b, "TR from:", trinit, "to:", (trinit + length(H$tr) - 1), "size:", length(H$tr),
      "TS from:", H$ts[1], "to:", H$ts[length(H$ts)], "size:", length(H$ts),
      "nmae:", ev[b], ",", ev_arima[b], ",", ev_nnetar[b], ",", ev_ets[b], ",", ev2[b], ",", ev_lm[b], "\n")
  
  mgraph(d1[H$ts],Pred,graph="REG",Grid=10,col=c("black","blue","green","purple","orange","red","pink"),leg=list(pos="topleft",leg=c("target","HW pred.","arima","nn","ets","mlpe","lm")))
  lines(Pred_arima,pch=19,cex=0.5,type="b",col="green")
  lines(Pred_nnetar,pch=19,cex=0.5,type="b",col="purple")
  lines(Pred_ets,pch=19,cex=0.5,type="b",col="orange")
  lines(Pred2,pch=19,cex=0.5,type="b",col="red")
  lines(Pred_lm,pch=19,cex=0.5,type="b",col="pink")
  mpause() # wait for enter
  
  
}
# show median of each model
cat("median NMAE values for Holt-Winters, mlpe, ARIMA, nnetar, ETS, and LM:\n")
cat("Holt-Winters median NMAE:",median(ev),"\n")
cat("ARIMA median NMAE:",median(ev_arima),"\n")
cat("nnetar median NMAE:",median(ev_nnetar),"\n")
cat("ETS median NMAE:",median(ev_ets),"\n")
cat("mlpe median NMAE:",median(ev2),"\n")
cat("LM median NMAE:",median(ev_lm),"\n")

mpause() # wait for enter  