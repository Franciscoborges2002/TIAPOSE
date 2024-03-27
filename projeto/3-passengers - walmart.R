# 3-passengers.R: univariate multi-step ahead time series forecasting example: passengers.ts.
# this series contains a 12 month seasonal pattern and a growing trend.
# in the past, it was well studied by conventional time series forecasting methods (e.g., ARIMA).
#
# in this demo, there are a total of 12 forecasts: 1-ahead, 2-ahead, ..., 12-ahead (thus multi-step ahead forecasting)

# setwd("") # adjust working directory if needed.

# install.packages("forecast")
library(forecast)
library(rminer)

# simple auxiliary show function: show target values and forecasts
reshow=function(label="",Y,Pred,metric="MAE",PLOT=TRUE)
{ main=paste(label,metric,":",round(mmetric(Y,Pred,metric=metric),digits=1))
  LEG=c("target",paste(label,"pred."))
  if(PLOT)mgraph(Y,Pred,graph="REG",main=main,Grid=10,col=c("black","blue"),leg=list(pos="topleft",leg=LEG))
  else cat(main,"\n")

}


cat("read walmart time series:")
# Carregar dados do arquivo CSV
dados <- read.csv("walmart.csv")

TS <- dados$WSdep1  # Vamos usar a coluna "WSdep1" como nossa série temporal


K=4 # TS period (monthly!)
#print("show graph")
#tsdisplay(TS)
#mpause()

L=length(TS)
NTS=K # number of predictions
H=NTS # from 1 to H ahead predictions

# --- this portion of code uses forecast library, which assumes several functions, such as forecast(), and uses a ts object 
# --- note: the forecast library works differently than rminer
# time series monthly object, frequency=K 
# this time series object only includes TRAIN (older) data:
LTR=L-H
# according to the ts function documentation: frequency=7 typically assumes daily data, frequency=4 or 12 assumes quarterly and monthly data
TR=ts(TS[1:LTR],frequency=K) # do not use the start parameter!
# show the in-sample (training data) time series:
plot(TR)
mpause() # press enter

# target predictions:
Y=TS[(LTR+1):L]

# holt winters forecasting method:
print("model> HoltWinters")
HW=HoltWinters(TR)
print(HW)
plot(HW)
print("show holt winters forecasts:")
# forecasts, from 1 to H ahead:
F1=forecast(HW,h=H)
# print(F1) # ignore the month in the print, it is not correct
Pred1=F1$mean[1:H] # HolWinters format
reshow("HW",Y,Pred1,"MAE")
mpause() # press enter

# arima modeling:
print("model> auto.arima")
AR=auto.arima(TR)
print(AR) # ARIMA(3,0,1)(2,1,0)[12] 
print("show ARIMA forecasts:")
# forecasts, from 1 to H ahead:
F2=forecast(AR,h=H)
#print(F2) # ignore the month in the print, it is not correct
Pred2=F2$mean[1:H]
reshow("AR",Y,Pred2,"MAE")
mpause() # press enter

# NN from forecast:
print("model> nnetar")
NN1=nnetar(TR,P=1,repeats=3)
print(NN1)
F3=forecast(NN1,h=H)
Pred3=F3$mean[1:H] # HolWinters format
reshow("NN1",Y,Pred3,"MAE")
mpause() # press enter

# ets from forecast:
print("model> ets")
ETS=ets(TR)
print(ETS)
F4=forecast(ETS,h=H)
Pred4=F4$mean[1:H] # HolWinters format
reshow("ets",Y,Pred4,metric="MAE")
mpause() # press enter

# -- end of forecast library methods

# neural network modeling, via rminer:
print("model> mlpe (with t-1,t-4,t-8 lags)")
d=CasesSeries(TS,c(1,4,8)) # data.frame from time series (domain knowledge for the 1,12,13 time lag selection)
print(summary(d))
LD=nrow(d) # note: LD < L
hd=holdout(d$y,ratio=NTS,mode="order")
NN2=fit(y~.,d[hd$tr,],model="mlpe")
print(NN2@object)
# multi-step, from 1 to H ahead forecasts:
init=hd$ts[1] # or same as: init=LD-H+1
# for multi-step ahead prediction, the lforecast from rminer should be used instead of predict,
# since predict only performs 1-ahead predictions
F5=lforecast(NN2,d,start=hd$ts[1],horizon=H)
#print(F5)
Pred5=F5
reshow("NN2",Y,Pred5,"MAE")
mpause() # press enter

# linear regression modeling ("lm"), via rminer:
print("model> lm (with t-1,t-4,t-8 lags)")
LM=fit(y~.,d[hd$tr,],model="lm")
print(LM@object)
# multi-step, from 1 to H ahead forecasts:
init=hd$ts[1] # or same as: init=LD-H+1
# for multi-step ahead prediction, the lforecast from rminer should be used instead of predict,
# since predict only performs 1-ahead predictions
F6=lforecast(LM,d,start=hd$ts[1],horizon=H)
#print(F6)
Pred6=F6
reshow("LM",Y,Pred6,metric="MAE")
mpause() # press enter
#

cat("forecast library methods:\n")
reshow("HW >",Y,Pred1,"MAE",FALSE)
reshow("AR >",Y,Pred2,"MAE",FALSE)
reshow("NN1 >",Y,Pred3,"MAE",FALSE)
reshow("ET >",Y,Pred4,"MAE",FALSE)
cat("rminer NN methods:\n")
reshow("NN2 >",Y,Pred5,"MAE",FALSE)
reshow("LM >",Y,Pred6,"MAE",FALSE)
# end

# HW with no seasonality (gamma=0):
# Carregar dados do arquivo CSV
dados <- read.csv("walmart.csv")
TS <- dados$WSdep1  # Vamos usar a coluna "WSdep1" como nossa série temporal

K=4
L=length(TS)
NTS=K # number of predictions
H=NTS # from 1 to H ahead predictions
TR=ts(TS[1:LTR],frequency=K)
HW2=HoltWinters(TR,gamma=0.0)
print(HW)
plot(HW)
print("show holt winters forecasts:")
# forecasts, from 1 to H ahead:
FHW2=forecast(HW,h=H)
PredH2=FHW2$mean[1:H] # HolWinters format
reshow("HW (gamma=0)",Y,PredH2,metric="MAE")
