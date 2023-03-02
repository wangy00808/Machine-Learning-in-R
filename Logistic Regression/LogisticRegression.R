rm(list = ls())

ebay <- read.csv("eBayAuctions2.csv")
## Include the functions required for data partitioning
source("myfunctions.R")

#########################################
## Create training and test data ##
#########################################
RNGkind (sample.kind = "Rounding") 
set.seed(0) ## set seed so that you get same partition each time
p2 <- partition.2(ebay, 0.7) ## creating 70:30 partition
training.data <- p2$data.train
test.data <- p2$data.test

#########################################
## Create plots on training data ##
#########################################
library(lessR)
BarChart(Competitive, by=currency, data=training.data, fill="grays")


lab <- c("sellerRating", "Duration", "ClosePrice", "OpenPrice")
par(mfrow = c(2,2))
for (i in 2:5){
  boxplot(training.data[,i] ~ training.data[,6], xlab = "Competitive", ylab=lab[i-1])
}


###################################
## Fit logistic regression model ##
###################################

logistic.model <- glm(Competitive ~ ., family = binomial(link='logit'), data=training.data)
summary(logistic.model)
confint.default(logistic.model) ## confidence interval for regression coefficients
exp(confint.default(logistic.model)) ## confidence interval for odds ratio

### Prediction on new data ###
x0 <- data.frame(currency="US", sellerRating = 2000, Duration = 7, 
                 ClosePrice = 13.01, OpenPrice = 9.99)
predict(logistic.model, newdata = x0, type = "response")


###confidence interval
confint.default(logistic.model)

library(caret)

# Confusion matrix for training data
pred.prob.train <- logistic.model$fitted.values
pred.y.train <- ifelse(pred.prob.train > 0.5, 1, 0) # using cutoff = 0.5
confusionMatrix(as.factor(pred.y.train), as.factor(training.data$Competitive), 
                positive = "1")

# Confusion matrix for test data
pred.prob.test <- predict(logistic.model, newdata = test.data,type = "response")
pred.y.test <- ifelse(pred.prob.test > 0.5, 1, 0) # using cutoff = 0.5
confusionMatrix(as.factor(pred.y.test), as.factor(test.data$Competitive), 
                positive = "1")



