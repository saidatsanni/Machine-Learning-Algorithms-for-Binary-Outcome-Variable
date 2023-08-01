
###########################################Analysis of German Credit Scoring Data using Machine Learning Techniques ##############################################
###Methods
#1. Variable Selection
#2. Generalized Linear Model (Logistic Regression)
#3. Classification Trees
#4. Bagging
#5. Random Forest
#6. Boosted Regression Trees
#7. Generalized Additive Model
#8. Neural Networks
##  Exploratory Data Analysis, Residual Diagnostics, In-sample Prediction, Out-of-sample Prediction, Predictive Performance, and Model Comparison are also included.
## The ROC curve, Confusion matrix, Misclassification rate/cost (symmetric and asymmetric) are employed.

##Load the required packages
library(Hmisc)
library(dplyr)
library(corrr)
library(tidyverse)
library(corrplot)
library(ggplot2) 
library(ROCR)
library(rpart)
library(rpart.plot)
library(glmnet)
library(boot)
library(randomForest)
library(gbm)
library(mgcv)
library(nnet)
library(ROCR)
library(ipred)
library(adabag)

#load the data, code variables, and convert to correct data types
german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
names(german_credit)
colnames(german_credit)=c("chk_acct","duration","credit_his","purpose","amount","saving_acct","present_emp","installment_rate","sex","other_debtor","present_resid","property","age","other_install","housing","n_credits","job","n_people","telephone","foreign","response")

#Recode the respone varaible to 0/1
german_credit$response = german_credit$response - 1
german_credit$response <- as.factor(german_credit$response)
str(german_credit)


##Split the data into 70% Training and 30% Testing data set
index <- sample(nrow(german_credit),nrow(german_credit)*0.70)
german_train = german_credit[index,]
german_test = german_credit[-index,]


##########################################################################Exploratory Data Analysis
str(german_credit) 
summary(german_train)
mean(as.numeric(german_train$response))
apply(german_train[,1:21], 2, sd)


#histogram and density plot for numeric variables
german_train %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2)                     

#boxplot for numeric variables
german_train %>%
  keep(is.numeric) %>% 
  gather(key = "var", value = "value") %>%
  ggplot(aes(x = '',y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  facet_wrap(~ var, scales = "free") +
  theme_bw()

#categorical variables wrt to response
par(mfrow=c(3,5))
for (i in c(1,2,4,6,7,9,10,12,14,15,17,19,20)){
  n <- names(german_train)
  counts <- table(german_train[,i])
  barplot(counts,  
          xlab=paste("Number of", n[i]))
}


############################################################## 1: Variable selection
#Backward
german.glm.back <- step(german.glm0)
summary(german.glm.back)
german.glm.back$deviance
AIC(german.glm.back) 
BIC(german.glm.back) 


#Forward variable selection
model_null<- glm(response~1, family=binomial(link="logit"),data=german_train) 
fullmodel=glm(response~., family=binomial(link="logit"),data=german_train)
german.glm_forward = step(model_null, scope=list(lower=model_null, upper=fullmodel), direction="forward")
summary(german.glm_forward)
german.glm_forward$deviance
AIC(german.glm_forward) 
BIC(german.glm_forward) 

#Stepwise selection
german.glm_both = step(german.glm0, scope=list(lower=model_null, upper=fullmodel), direction='both')
summary(german.glm_both)
german.glm_both$deviance
AIC(german.glm_both) 
BIC(german.glm_both) 



############################################################### 2: Logistic Regression Model - GLM
german.glm0<- glm(response~., family=binomial(link="logit"), data=german_train)
summary(german.glm0)
german.glm0$deviance 
AIC(german.glm0) 
BIC(german.glm0) 


######################GLM in-sample performance
pred_glm0.train <- predict(german.glm.back, newdata = german_train, type="response")

#Asymmetric Misclassification rate
costfunc = function(obs, pred.p, pcut){
  weight1 = 5   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # definition of FN/count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # definition of FP/count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function

# define a sequence from 0.01 to 1 by 0.01
p.seq = seq(0.01, 1, 0.01) 
cost = rep(0, length(p.seq)) 
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = german_train$response, pred.p = pred_glm0.train, pcut = p.seq[i])  
} 
plot(p.seq, cost)
min(cost)


#Area Under the Curve (AUC)
optimal.pcut.german = p.seq[which(cost==min(cost))] 
german_train <- as.data.frame(german_train)
pred_glm0.train <- predict(german.glm.back, newdata = german_train, type="response")
pred <- prediction(pred_glm0.train, german_train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))



#Symmetric MR
class.glm0.train.opt<- (pred_glm0.train > optimal.pcut.german)*1
table(german_train$response, class.glm0.train.opt, dnn = c("True", "Predicted"))
MR<- mean(german_train$response!= class.glm0.train.opt)
MR


############################Out-of-sample Performance
pred.german.test<- predict(german.glm.back, newdata = german_test, type="response")
class.glm1.test <-  (pred.german.test>optimal.pcut.german)*1
# get confusion matrix
table(german_test$response, class.glm1.test, dnn = c("True", "Predicted"))
#calculate MR, FPR, FNR, and cost based on the optimal cut-off.
MR_test <- mean(german_test$response!= class.glm1.test)
MR_test 

pred_glm0.test<- predict(german.glm.back,newdata=german_test, type="response")
pred <- prediction(pred_glm0.test, german_test$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values")) #AUC




###############################################3: CLASSIFICATION TREES
#Full Tree
german_credit.rpart <- rpart(formula = response ~ ., data = german_train, method = "class", parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)), cp = 0.00001) 
plotcp(german_credit.rpart)  
prp(german_credit.rpart,extra=1)

#Confusion Matrix
credit.train.pred.tree1<- predict(german_credit.rpart, german_train, type="class")
table(german_train$response, credit.train.pred.tree1, dnn=c("Truth","Predicted"))

credit.test.pred.tree1<- predict(german_credit.rpart, german_test, type="class")
table(german_test$response, credit.test.pred.tree1, dnn=c("Truth","Predicted"))

#Cost Function
cost <- function(r, pi){
  weight1 = 5
  weight0 = 1
  c1 = (r==1)&(pi==0) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi==1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}
#AMR
cost(german_train,credit.train.pred.tree1)
cost(german_test,credit.test.pred.tree1)


##Pruned tree
credit.largetree <- rpart(formula = response ~ ., data = german_train, cp=0.001)
prp(credit.largetree)
plotcp(credit.largetree)

printcp(credit.largetree)
prun_credittree <- prune(credit.largetree, cp = 0.0022) #size =11, 15 splits
prp(prun_credittree)

#Confusion Matrix
credit.train.pred.pruntree<- predict(prun_credittree, german_train, type="class")
table(german_train$response, credit.train.pred.pruntree, dnn=c("Truth","Predicted"))

credit.test.pred.pruntree<- predict(prun_credittree, german_test, type="class")
table(german_test$response, credit.test.pred.pruntree, dnn=c("Truth","Predicted"))



############AMR
cost <- function(r, pi){
  weight1 = 5
  weight0 = 1
  c1 = (r==1)&(pi==0) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi==1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}
cost(german_train,credit.train.pred.pruntree) ##in-sample
cost(german_test,credit.test.pred.pruntree)  ##out-of-sample



########################################################4: BAGGING
credit.bag<- bagging(as.factor(response)~., data = german_train, nbagg=100)
credit.bag.pred<- predict(credit.bag, newdata = german_train, type="prob")[,2]
credit.bag.pred.test<- predict(credit.bag, newdata = german_test, type="prob")[,2]

#Confusion Matrix
credit.bagtrain<- predict(credit.bag, german_train, type="class")
table(german_train$response, credit.bagtrain, dnn=c("Truth","Predicted"))

credit.bagtest<- predict(credit.bag, german_test, type="class")
table(german_test$response, credit.bagtest, dnn=c("Truth","Predicted"))


#AMR
cost(german_train,credit.bagtrain)
cost(german_test,credit.bagtest)


#AUC Values
pred = prediction(credit.bag.pred.test, german_test$response)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

##AUC FOR TRAINING DATA
pred = prediction(credit.bag.pred, german_train$response)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values")) 




################################################ 5: RANDOM FOREST
credit.rf <- randomForest(as.factor(response)~., data = german_train)
#Variable importance
varImpPlot(credit.rf)

#Confusion Matrix
credit.rftrain<- predict(credit.rf, german_train, type="class")
table(german_train$response, credit.rftrain, dnn=c("Truth","Predicted"))

credit.rftest<- predict(credit.rf, german_test, type="class")
table(german_test$response, credit.rftest, dnn=c("Truth","Predicted"))


#AMR
cost(german_train,credit.rftrain)
cost(german_test,credit.rftest)




######################################### 6: Boosting for classification trees
german_train$response= as.factor(german_train$response)
credit.boost= boosting(response~., data = german_train, boos = T)
save(credit.boost, file = "credit.boost.Rdata")

# Training AUC
pred.credit.boost= predict(credit.boost, newdata = german_train)
pred <- prediction(pred.credit.boost$prob[,2], german_train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values")) 

# Testing AUC
pred.credit.boost= predict(credit.boost, newdata = german_test)
pred <- prediction(pred.credit.boost$prob[,2], german_test$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values")) #0.7733

#Confusion Matrix
credit.boosttrain<- predict(credit.boost, german_train, type="class")
table(german_train$response, credit.boosttrain$class, dnn=c("Truth","Predicted"))

credit.boosttest<- predict(credit.boost, german_test, type="class")
table(german_test$response, credit.boosttest$class, dnn=c("Truth","Predicted"))

#AMR
cost(german_train,credit.boosttrain$class)
cost(german_test,credit.boosttest$class)



################################################### 7: GAM model
credit.gam <- gam(response~chk_acct+s(duration)+credit_his+purpose+s(amount)+
                    saving_acct+present_emp+installment_rate+sex+other_debtor+
                    present_resid+property+s(age)+other_install+housing+
                    n_credits+job+n_people+telephone+foreign, family=binomial,data=german_train)
summary(credit.gam)
plot(credit.gam,pages=1)

#age is linear
credit.gam1 <- gam(response~chk_acct+duration+credit_his+purpose+amount+
                     saving_acct+present_emp+installment_rate+sex+other_debtor+
                     present_resid+property+age+other_install+housing+
                     n_credits+job+n_people+telephone+foreign, family=binomial,data=german_train)
summary(credit.gam1)

#partial residual plot
plot(credit.gam1, shade=TRUE,seWithMean=TRUE,scale=0, pages = 1)

#model check
AIC(credit.gam1)
BIC(credit.gam1)
credit.gam1$deviance

#Misclassification rate:
#Search for optimal cut-off probability
searchgrid = seq(0.01, 0.20, 0.01)
result.gam = cbind(searchgrid, NA)
cost1 <- function(r, pi){
  weight1 = 5
  weight0 = 1
  c1 = (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}

for(i in 1:length(searchgrid))
{
  pcut <- result.gam[i,1]
  result.gam[i,2] <- cost1(german_train$response, predict(credit.gam1,type="response"))
}
plot(result.gam, ylab="Cost in Training Set")

index.min<-which.min(result.gam[,2])#find the index of minimum value
result.gam[index.min,2] 
result.gam[index.min,1] #optimal cutoff probability


#Out-of-sample performance
pcut <-  result.gam[index.min,1] 
prob.gam.out<-predict(credit.gam1,german_test,type="response")
pred.gam.out<-(prob.gam.out>=pcut)*1
table(german_test$response,pred.gam.out,dnn=c("Observed","Predicted"))
mean(ifelse(german_test$response != pred.gam.out, 1, 0))

#in-sample AMR
prob.gam.in<-predict(credit.gam1,german_train,type="response")
pred.gam.in<-(prob.gam.in>=pcut)*1
mean(ifelse(german_train$response != pred.gam.in, 1, 0))



############################################################ 8: Neural Network

credit.nnet <- nnet(response~., data=german_train, size=2, maxit=1000)
prob.nnet= predict(credit.nnet,german_test, type="class")

##out-of-sample performance
prob.nnet= predict(credit.nnet,german_test)
pred.nnet = as.numeric(prob.nnet > 1.47)
table(german_test$response,pred.nnet, dnn=c("Observed","Predicted"))
mean(ifelse(german_test$response != pred.nnet, 1, 0))


#in-sample
prob.nnettrain= predict(credit.nnet,german_train)
pred.nnet.train = as.numeric(prob.nnettrain >1.47)
table(german_train$response,pred.nnet.train, dnn=c("Observed","Predicted"))
mean(ifelse(german_train$response != pred.nnet.train, 1, 0))


