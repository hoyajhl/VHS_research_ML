install.packages("glmnet")
install.packages("pROC")
install.packages("caret")

library(Matrix)
library(glmnet)
library(pROC)
library(caret)
library(ggplot2)
library(data.table) 

# Import dataset
data = read.csv(file = "C:/Hoya/research/vhs/survival/NSA_300_recur.csv",encoding = "UTF-8")

##using glmnet
x= data.matrix(data[,1:144])
y= data$Recurrence

set.seed(356)

##glmnet (without cv version)
lasso_basic = glmnet(x, y,family = "binomial",alpha = 1)
plot(lasso_basic,xvar='lambda')

##glmnet (with cv version)
best_lambda<-cv.glmnet(x,y)$lambda.min ## best lambda: 0.05736173
lasso_cv=cv.glmnet(x, y,family = "binomial",alpha = 1,nfolds = 10) ## 10cv
plot(lasso_cv)


myCoefs <- coef(lasso_cv,s=best_lambda) ## best lambda: 0.05736173
myCoefs[which(myCoefs != 0 ), ]  ##3 variables picked


varsSelected <- myCoefs[which(myCoefs != 0 ),]
number_selected<-length(varsSelected)-1 ## intercept 제외 
varsNotSelected <- myCoefs[which(myCoefs== 0 ),]
non_selected<-length(varsNotSelected)

cat('Lasso uses', number_selected, 'variables in its model, and did not select'
    ,non_selected, 'variables.')

## Lasso uses 3 variables in its model, and did not select 110 variables.

myCoefs <- coef(lasso_basic,s=best_lambda) ## best lambda: 0.05736173
myCoefs[which(myCoefs != 0 ), ]  

coefs_random <- coef(lasso_basic,s=0.01) #lambda를 0.01로 조정 
coefs_random[which(coefs_random != 0 ), ]  

varsSelected <- coefs_random[which(coefs_random  != 0 ),]
number_selected<-length(varsSelected)-1 ## intercept 제외 
varsNotSelected <- coefs_random[which(coefs_random== 0 ),]
non_selected<-length(varsNotSelected)

cat('Lasso uses', number_selected, 'variables in its model, and did not select'
    ,non_selected, 'variables.')
## Lasso uses 34 variables in its model, and did not select 110 variables


selected_coef = data.table(lasso = varsSelected)      # build table
selected_coef_table<-selected_coef[, feature := names(varsSelected)]
selected_coef_table


to_plot_r = melt(selected_coef_table                      # label table
                 , id.vars='feature'
                 , variable.name = 'model'
                 , value.name = 'coefficient')
to_plot_r

ggplot(data=to_plot_r,                       # plot coefficients
       aes(x=feature, y=coefficient, fill=model)) +
  coord_flip() +         
  geom_bar(stat='identity', fill='brown4', color='green') +
  facet_wrap(~ model) + guides(fill=FALSE) 