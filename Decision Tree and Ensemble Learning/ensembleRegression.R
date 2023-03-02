rm(list = ls())

autompg <- read.csv("autompg.csv")

## Include the functions required for data partitioning
source("myfunctions.R")
RNGkind (sample.kind = "Rounding") 
set.seed(0)
p2 <- partition.2(autompg, 0.6)
training.data <- p2$data.train
test.data <- p2$data.test

####################################
###### Bagging #####################
####################################
library(caret)
set.seed(0)
train_control <- trainControl(method="cv", number=10)
## specify nbagg to control the number of trees. default value is 25 
bag <- train(mpg ~ . , data = training.data, method = "treebag",
               trControl = train_control, nbagg = 50)
print(bag)
plot(varImp(bag))
#dev.copy2pdf(file = "E:/Data mining/Lecture Notes/plots/ct3.pdf")
bag$finalModel

# get prediction on the test data
pred.test = predict(bag, test.data)
# get MSE on test data
rmse.test.bag <- sqrt(mean((test.data$mpg - pred.test)^2))
rmse.test.bag

####################################
###### Random Forest ###############
####################################
library(caret)
set.seed(0)
modelLookup("rf")
train_control <- trainControl(method="cv", number=10)
rf <- train(mpg ~ . , data = training.data, method = "rf",
             trControl = train_control, tuneLength = 3)
## specify tuning parameters using tuneGrid
# rf <- train(mpg ~ . , data = training.data, method = "rf", 
#                trControl = train_control, tuneGrid = expand.grid(mtry = c(2,5,8)))
print(rf)
plot(varImp(rf))
#dev.copy2pdf(file = "E:/Data mining/Lecture Notes/plots/ct3.pdf")
rf$finalModel

# get prediction on the test data
pred.test = predict(rf$finalModel, test.data)
# get MSE on test data
rmse.test.rf <- sqrt(mean((test.data$mpg - pred.test)^2))
rmse.test.rf
