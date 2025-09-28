# ####################################################### 
#
#  Initialization for simulations and real data analysis
#
# ####################################################### 


# ################################
#
# R packages which are needed
#

# install.packages("tictoc")
# install.packages("spam")
# install.packages("speedglm")
# install.packages("glmnet")
# install.packages("mvtnorm")
# install.packages("bigstep")
# install.packages("L0Learn")

suppressPackageStartupMessages({
  library(tictoc)
  library(spam)
  library(speedglm)
  library(glmnet)
  library(mvtnorm)
  library(bigstep)
  library(L0Learn)
})

# ################################
#
# Functions to compute criteria mBIC and mBIC2 (based on bigstep)
#
this_dir <- dirname(normalizePath(sys.frames()[[1]]$ofile %||% "Init.R"))
setwd(this_dir)

my.loglik = function(y, Xm, model = "linear") 
{
  # Xm is the design matrix of the selected model excluding the intercept
  # the loglik function from bigstep cannot handle the null model
  
  if (model == "linear"){
    
    if (dim(Xm)[2] == 0){
      rss <- sum((y - mean(y))^2)
      n <- length(y)
      loglik <- -n/2 * log(rss/n)
    } else {
      loglik <- bigstep:::loglik(y,Xm)
    }
    
  } else { #logistic regression
    
    if (dim(Xm)[2] == 0){
      m.log <- glm(y ~1, family = binomial())
      loglik <- as.numeric(logLik(m.log))
    } else {
      m.log <- speedglm(y ~ Xm,  family = binomial())
      loglik <- m.log$logLik
    }
  }
  return(loglik)
}

# #################################################################
#
#  Stepwise selection procedures with 'bigstep'
#

# #####  Plain stepwise search using R-package bigstep ######

stepwise_plain = function(y, X, model = "linear")  {
  
  n = dim(X)[1]
  p = dim(X)[2]
  
  data = bigstep::prepare_data(y, X,  type = model, verbose=FALSE)
  
  # search with mBIC
  fit <- data %>%
    stepwise(crit=mbic)
  
  find.plain <- as.numeric(fit$model)
  k<-length(find.plain)
  
  loglik.bigstep = my.loglik(y,X[,find.plain,drop = FALSE], model = model)
  mBIC.plain = bigstep::mbic(loglik.bigstep, k, n, p)
  
  # Search with mBIC2
  fit <- data %>%
    stepwise(crit=mbic2)
  
  find2.plain <- as.numeric(fit$model)
  k<-length(find2.plain)
  
  loglik.bigstep = my.loglik(y,X[,find2.plain,drop = FALSE], model = model)
  mBIC2.plain = bigstep::mbic2(loglik.bigstep , k, n, p)
  
  result.list = list(mBIC.plain, mBIC2.plain, find.plain, find2.plain)
  return(result.list)
}

# #####  Stepwise search after screening using R-package bigstep ######

stepwise_reduced = function(y, X, model = "linear")  {
  
  n = dim(X)[1]
  p = dim(X)[2]
  
  data = bigstep::prepare_data(y, X,  type = model, verbose=FALSE)
  
  # search with mBIC
  fit <- data %>%
    reduce_matrix()%>%     # removes markers with marginal p-value larger than 0.15
    stepwise(crit=mbic)
  
  find.plain <- as.numeric(fit$model)
  k<-length(find.plain)
  
  loglik.bigstep = my.loglik(y,X[,find.plain,drop = FALSE], model = model)
  mBIC.plain = bigstep::mbic(loglik.bigstep, k, n, p)
  
  # Search with mBIC2
  fit <- data %>%
    reduce_matrix()%>%    # removes markers with marginal p-value larger than 0.15
    stepwise(crit=mbic2)
  
  find2.plain <- as.numeric(fit$model)
  k<-length(find2.plain)
  
  loglik.bigstep = my.loglik(y,X[,find2.plain,drop = FALSE], model = model)
  mBIC2.plain = bigstep::mbic2(loglik.bigstep , k, n, p)
  
  result.list = list(mBIC.plain, mBIC2.plain, find.plain, find2.plain)
  return(result.list)
}

# #####  Stepwise after fast forward search using R-package bigstep ######

stepwise_ff = function(y, X, model = "linear")  {
  
  n = dim(X)[1]
  p = dim(X)[2]
  
  data = bigstep::prepare_data(y, X,  type = model,  verbose=FALSE)  
  data <- data %>% bigstep::reduce_matrix(minpv = 1) # Searching in the order of marginal p-values
    
  # search with mBIC
  fit <- data %>%
    fast_forward()%>%    # Default: BIC
    multi_backward(crit=mbic)%>% 
    stepwise(crit=mbic)
  
  find <- as.numeric(fit$model)
  k<-length(find)
  
  loglik.bigstep = my.loglik(y,X[,find,drop = FALSE], model = model)
  mBIC = bigstep::mbic(loglik.bigstep, k, n, p)
  
  # Search with mBIC2
  fit <- data %>%
    fast_forward()%>%   # Searching in the order of marginal p-values
    multi_backward(crit=mbic2)%>% 
    stepwise(crit=mbic2)
  
  find2 <- as.numeric(fit$model)
  k<-length(find2)
  
  loglik.bigstep = my.loglik(y,X[,find2,drop = FALSE], model = model)
  mBIC2 = bigstep::mbic2(loglik.bigstep , k, n, p)
  
  result.list = list(mBIC, mBIC2, find, find2)
  return(result.list)
}

# #################################################################
#
#  Model selection with 'L0Learn'
#

# ##### L0opt with CD algorithm (cyclic coordinate descent) ########

L0opt_CD = function(y, X, model.loss = "SquaredError", maxSuppSize = NA)  {
  
  if (model.loss == "SquaredError")   {
    model = "linear" 
  } else model = "logistic"
  
  n = dim(X)[1]
  p = dim(X)[2]
  p.vec = 1:p
  
  if (is.na(maxSuppSize)) {
    maxSuppSize = min(c(round(p/2), round(n/2), 150))   # Restrictions on the maximal model size
  }
  
  results.CD = L0Learn.fit(X, y, loss = model.loss, penalty = "L0",
                           algorithm = "CD", maxSuppSize = maxSuppSize, nLambda = 200, nGamma = 0,
                           gammaMax = 10, gammaMin = 1e-03, partialSort = TRUE,
                           maxIters = 200,  rtol = 1e-06,
                           atol = 1e-09, activeSet = TRUE, activeSetNum = 3,
                           maxSwaps = 100, scaleDownFactor = 0.8, screenSize = 1000,
                           excludeFirstK = 0,
                           intercept = TRUE)
  
  res.CD = print(results.CD)
  mod.nr = length(res.CD$lambda)
  
  C = coef(results.CD, lambda = res.CD$lambda[1])
  Cmodel.index = as.numeric(abs(C))>0   #includes intercept
  loglik = my.loglik(y,X[,Cmodel.index[-1],drop = FALSE], model = model)
  mBIC.best  = bigstep::mbic(loglik, res.CD$suppSize[1], n, p)
  mBIC2.best = bigstep::mbic2(loglik, res.CD$suppSize[1], n, p)
  best.CD.mBIC = best.CD.mBIC2 = 1
  no.improvement.count = 0
  
  for (i in 2:mod.nr)  {
    no.improvement.count = no.improvement.count + 1
    C = coef(results.CD, lambda = res.CD$lambda[i])
    Cmodel.index = as.numeric(abs(C))>0   #includes intercept
    loglik = my.loglik(y,X[,Cmodel.index[-1],drop = FALSE], model = model)
    C.mBIC = bigstep::mbic(loglik, res.CD$suppSize[i], n, p)
    C.mBIC2 = bigstep::mbic2(loglik, res.CD$suppSize[i], n, p)
    if (C.mBIC < mBIC.best){ mBIC.best = C.mBIC; no.improvement.count = 0; best.CD.mBIC = i }
    if (C.mBIC2 < mBIC2.best){ mBIC2.best = C.mBIC2; no.improvement.count = 0; best.CD.mBIC2 = i }
    if (no.improvement.count == 5) break
  }
  
  CC = coef(results.CD, lambda = res.CD$lambda[best.CD.mBIC])
  model.index = as.numeric(abs(CC))>0   #includes intercept
  model.CD = p.vec[model.index[-1]]
  
  CC = coef(results.CD, lambda = res.CD$lambda[best.CD.mBIC2])
  model.index2 = as.numeric(abs(CC))>0   #includes intercept
  model2.CD = p.vec[model.index2[-1]]
  
  result.list = list(mBIC.best, mBIC2.best, model.CD, model2.CD)
  return(result.list)
}

# ################# L0opt with CDPSI ############################

L0opt_CDPSI = function(y, X, model.loss = "SquaredError", maxSuppSize = NA)  {
  
  if (model.loss == "SquaredError")  { 
    model = "linear"
  } else model = "logistic"
  
  n = dim(X)[1]
  p = dim(X)[2]
  p.vec = 1:p
  
  if (is.na(maxSuppSize)) {
    maxSuppSize = min(c(round(p/2), round(n/2), 150))   # Restrictions on the maximal model size
  }
  
  results.CDPSI = L0Learn.fit(X, y, loss = model.loss, penalty = "L0",
                              algorithm = "CDPSI", maxSuppSize = maxSuppSize, nLambda = 200, nGamma = 0,
                              gammaMax = 10, gammaMin = 1e-03, partialSort = TRUE,
                              maxIters = 200,  rtol = 1e-06,
                              atol = 1e-09, activeSet = TRUE, activeSetNum = 3,
                              maxSwaps = 100, scaleDownFactor = 0.8, screenSize = 1000,
                              excludeFirstK = 0,
                              intercept = TRUE)
  
  res.CDPSI = print(results.CDPSI)
  mod.nr = length(res.CDPSI$lambda)
  
  C = coef(results.CDPSI, lambda = res.CDPSI$lambda[1])
  Cmodel.index = as.numeric(abs(C))>0   #includes intercept
  loglik = my.loglik(y,X[,Cmodel.index[-1],drop = FALSE], model = model)
  mBIC.best  = bigstep::mbic(loglik, res.CDPSI$suppSize[1], n, p)
  mBIC2.best = bigstep::mbic2(loglik, res.CDPSI$suppSize[1], n, p)
  best.CDPSI.mBIC = best.CDPSI.mBIC2 = 1
  no.improvement.count = 0
  
  for (i in 2:mod.nr)  {
    no.improvement.count = no.improvement.count + 1
    C = coef(results.CDPSI, lambda = res.CDPSI$lambda[i])
    Cmodel.index = as.numeric(abs(C))>0   #includes intercept
    loglik = my.loglik(y,X[,Cmodel.index[-1],drop = FALSE], model = model)
    C.mBIC = bigstep::mbic(loglik, res.CDPSI$suppSize[i], n, p)
    C.mBIC2 = bigstep::mbic2(loglik, res.CDPSI$suppSize[i], n, p)
    if (C.mBIC < mBIC.best){ mBIC.best = C.mBIC; no.improvement.count = 0; best.CDPSI.mBIC = i }
    if (C.mBIC2 < mBIC2.best){ mBIC2.best = C.mBIC2; no.improvement.count = 0; best.CDPSI.mBIC2 = i }
    if (no.improvement.count == 5) break
  }
  
  CC = coef(results.CDPSI, lambda = res.CDPSI$lambda[best.CDPSI.mBIC])
  model.index = as.numeric(abs(CC))>0   #includes intercept
  model.CDPSI = p.vec[model.index[-1]]
  
  CC = coef(results.CDPSI, lambda = res.CDPSI$lambda[best.CDPSI.mBIC2])
  model.index2 = as.numeric(abs(CC))>0   #includes intercept
  model2.CDPSI = p.vec[model.index2[-1]]
  
  result.list = list(mBIC.best, mBIC2.best, model.CDPSI, model2.CDPSI)
  return(result.list)
}

# #################################################################
#
#  GSDAR  (Code from https://github.com/jian94/GSDAR)
#  Works only for logistic regression
#

GSDAR.source.code = file.path("GSDAR","GSDAR.R")
source(GSDAR.source.code)

Select_GSDAR = function(y, X){
  
  model = "logistic"
  n = dim(X)[1]
  p = dim(X)[2]
  p.vec = 1:p
  
  k.max = min(c(round(p/2), round(n/2), 150)) # Restrictions on the maximal model size
  loglik.0 = my.loglik(y, matrix(NA,n,0), model = model)
  mBIC.0 = bigstep::mbic(loglik.0 , 0, n, p)
  G.mbic = G.mbic2 = rep(mBIC.0, k.max)
  GSDAR.models = vector("list", k.max)
  
  # GSDAR does not work for k = 1
  data = bigstep::prepare_data(y, X,  type = model, verbose=FALSE)
  m1 = data %>% forward(crit = mbic) 
  G.mbic[1] =  G.mbic2[1] = m1$crit   #For k = 1 the two criteria are identical
  GSDAR.models[[1]] = as.integer(m1$model)
  
  mBIC.best = mBIC2.best = min(mBIC.0, m1$crit)
  no.improvement.count = 0
  
  for (i in 2:k.max)  {
    no.improvement.count = no.improvement.count + 1
    mk = GSDAR(rep(0,p),i,X,y,0.9,30)   #Selection of parameters like in demo_GSDAR.R
    GSDAR.models[[i]] = mk[[2]]
    loglik = my.loglik(y,X[,mk[[2]],drop = FALSE], model = model)
    G.mbic[i] = bigstep::mbic(loglik, i, n, p)
    G.mbic2[i] = bigstep::mbic2(loglik, i, n, p)
    if (G.mbic[i] < mBIC.best){ mBIC.best = G.mbic[i]; no.improvement.count = 0 }
    if (G.mbic2[i] < mBIC2.best){ mBIC2.best = G.mbic2[i]; no.improvement.count = 0 }
    if (no.improvement.count == 5) break
  }
  
  if (mBIC.best < mBIC.0){
    best.G.mBIC = which.min(G.mbic)
    model.G.mBIC =  GSDAR.models[[best.G.mBIC]]
  } else {
    best.G.mBIC = k.max; model.G.mBIC = vector("numeric")
  }
  if (mBIC2.best < mBIC.0){
    best.G.mBIC2 = which.min(G.mbic2)
    model.G.mBIC2 = GSDAR.models[[best.G.mBIC2]]
  } else {
    best.G.mBIC2 = k.max; model.G.mBIC2 = vector("numeric")
  }
  result.list = list(G.mbic[best.G.mBIC], G.mbic2[best.G.mBIC2], model.G.mBIC, model.G.mBIC2)
  return(result.list)
}


# --- bigstep nach außen "reichen" ---
if (!requireNamespace("bigstep", quietly = TRUE)) {
  stop("R-Paket 'bigstep' ist nicht installiert.")
}

mbic_py  <- function(loglik, n, k, p, const = 4) {
  bigstep::mbic(loglik = loglik, n = n, k = k, p = p, const = const)
}
mbic2_py <- function(loglik, n, k, p, const = 4) {
  # beachte: Signatur in bigstep::mbic2 ist (loglik, k, n, p, const)
  bigstep::mbic2(loglik = loglik, k = k, n = n, p = p, const = const)
}