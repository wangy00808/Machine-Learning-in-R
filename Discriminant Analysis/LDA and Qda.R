rm(list = ls())

ebay <- read.csv("eBayAuctions2.csv")
ebay$Competitive = as.factor(ebay$Competitive)
levels(ebay$Competitive) <- c("no", "yes")
## Include the functions required for data partitioning
source("partitionfunctions.R")

#########################################
## Create training and test data ##
#########################################
RNGkind (sample.kind = "Rounding") 
set.seed(0) ## set seed so that you get same partition each time
p2 <- partition.2(ebay, 0.7) ## creating 70:30 partition
training.data <- p2$data.train
test.data <- p2$data.test


################################################################
## Fit a LDA model ##
#################################################################
library(MASS)
library(caret)

lda.model <- lda(Competitive ~ ., training.data)
lda.model

### Prediction on new data ###
x0 <- data.frame(currency="US", sellerRating = 2000, Duration = 7, 
                 ClosePrice = 13.01, OpenPrice = 9.99)
predict(lda.model, newdata = x0)


# Confusion matrix for test data
pred.y.test <- predict(lda.model, newdata = test.data)$class
confusionMatrix(pred.y.test, as.factor(test.data$Competitive))



######################
## Cross validation ##
######################
library(caret)

## K-fold Cross Validation
# value of K equal to 10 
set.seed(0)
train_control <- trainControl(method = "cv", number = 10) 
modelLookup("lda")
# Fit K-fold CV model  
lda_kcv <- train(Competitive ~ ., data = training.data,  
                      method ="lda",  trControl = train_control) 
print(lda_kcv)
lda_kcv$finalModel

# Confusion matrix for test data
pred.y.test <- predict(lda_kcv, newdata = test.data)
confusionMatrix(pred.y.test, test.data$Competitive)


################################################################
## Fit a QDA model ##
#################################################################

qda.model <- qda(Competitive ~ ., training.data)
qda.model

### Prediction on new data ###
x0 <- data.frame(currency="US", sellerRating = 2000, Duration = 7, 
                 ClosePrice = 13.01, OpenPrice = 9.99)
predict(lda.model, newdata = x0)


# Confusion matrix for test data
pred.y.test <- predict(qda.model, newdata = test.data)$class
confusionMatrix(pred.y.test, as.factor(test.data$Competitive))

######################
## Cross validation ##
######################
library(caret)
# value of K equal to 10 
set.seed(0)
train_control <- trainControl(method = "cv", number = 10) 

# Fit K-fold CV model  
qda_kcv <- train(Competitive ~ ., data = training.data,  
                      method = "qda", trControl = train_control) 
print(qda_kcv)
qda_kcv$finalModel

# Confusion matrix for test data
pred.y.test <- predict(qda_kcv, newdata = test.data)
confusionMatrix(pred.y.test, test.data$Competitive)


################################################################
## Fit a RDA model ##
#################################################################
library(klaR)
######################
## Cross validation ##
######################
library(caret)
modelLookup("rda")
# value of K equal to 10 
set.seed(0)
train_control <- trainControl(method = "cv", number = 10) 

# Fit K-fold CV model  
rda_kcv <- train(Competitive ~ ., data = training.data,  
                 method = "rda",tuneLength = 4,  trControl = train_control) 
print(rda_kcv)
rda_kcv$finalModel

# Confusion matrix for test data
pred.y.test <- predict(rda_kcv, newdata = test.data)
confusionMatrix(pred.y.test, test.data$Competitive)

