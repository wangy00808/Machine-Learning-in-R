rm(list = ls())

diabetes <- read.csv("~/Desktop/teaching 2022 fall/Math 540&440 statistical learning/yang/datasets/diabetes.csv")
diabetes$Outcome = as.factor(diabetes$Outcome)

## Include the functions required for data partitioning
source("~/Desktop/teaching 2022 fall/Math 540&440 statistical learning/yang/R files/myfunctions.R")

#########################################
## Create training and test data ##
#########################################
RNGkind (sample.kind = "Rounding") 
set.seed(0) ## set seed so that you get same partition each time
p2 <- partition.2(diabetes, 0.7) ## creating 70:30 partition
training.data <- p2$data.train
test.data <- p2$data.test


################################################################
## Fit a linear svm model ##
#################################################################
library(MASS)
library(e1071) # support vector machine
library(caret) # cross validation 

liner_svm <- svm(Outcome ~ ., data = training.data, kernel = "linear", cost = 10, scale = FALSE)
summary(liner_svm)



# Confusion matrix for test data
pred.y.test <- predict(liner_svm, newdata = test.data)
confusionMatrix(pred.y.test, as.factor(test.data$Outcome))


## Cross validation ##################
set.seed(0)
train_control <- trainControl(method = "cv", number = 10) 

# Fit K-fold CV model  
linear.kcv <- train(Outcome ~., data = training.data, method = "svmLinear", 
                    trControl = train_control,  preProcess = c("center","scale"),
                    tuneGrid = expand.grid(C = seq(0.01, 10, length = 20)))
linear.kcv
linear.kcv$finalModel

# Confusion matrix for test data
pred.y.test <- predict(linear.kcv, newdata = test.data)
confusionMatrix(pred.y.test, test.data$Outcome)

################################################################
## Fit a svm with Gaussian (radial) kernel ##
#################################################################

radial_svm <- svm(Outcome ~., data = training.data, kernel = "radial",
                 gamma = 1, cost = 1)
summary(radial_svm)


# Confusion matrix for test data
pred.y.test <- predict(radial_svm, newdata = test.data)
confusionMatrix(pred.y.test, as.factor(test.data$Outcome))

## Cross validation ##################
set.seed(0)
train_control <- trainControl(method = "cv", number = 10) 

# Fit K-fold CV model  
Radial.kcv <- train(Outcome ~., data = training.data, method = "svmRadial", 
                    trControl = train_control,  preProcess = c("center","scale"),
                    tuneLength = 10)
Radial.kcv
Radial.kcv$finalModel

# Confusion matrix for test data
pred.y.test <- predict(Radial.kcv, newdata = test.data)
confusionMatrix(pred.y.test, test.data$Outcome)


################################################################
## Fit a svm with polynomial kernel ##
#################################################################

polyn_svm <- svm(Outcome ~., data = training.data, kernel = "polynomial",
                  degree = 2, cost = 10)
summary(polyn_svm)


# Confusion matrix for test data
pred.y.test <- predict(polyn_svm, newdata = test.data)
confusionMatrix(pred.y.test, as.factor(test.data$Outcome))

## Cross validation ##################
set.seed(0)
train_control <- trainControl(method = "cv", number = 10) 

# Fit K-fold CV model  
Polyn.kcv <- train(Outcome ~., data = training.data, method = "svmPoly", 
                    trControl = train_control,  preProcess = c("center","scale"),
                    tuneLength = 4)
Polyn.kcv
Polyn.kcv$finalModel

# Confusion matrix for test data
pred.y.test <- predict(Polyn.kcv, newdata = test.data)
confusionMatrix(pred.y.test, test.data$Outcome)


