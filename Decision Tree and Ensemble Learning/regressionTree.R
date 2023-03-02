rm(list = ls())

## Include the functions required for data partitioning
source("")

#######################
### Regression Tree ###
#######################
autompg <- read.csv("")

# use partition.2 function from myfunctions.R and create 60:40 partition
RNGkind (sample.kind = "Rounding") 
set.seed(0)
p2 <- partition.2(autompg, 0.6)
mydata.train <- p2$data.train
mydata.test <- p2$data.test

# fit regression tree on training data
ct1 <- rpart(mpg ~ . , data = mydata.train, method = "anova", 
             minsplit=15, minbucket = 5)
# plot tree
prp(ct1, type = 1, extra = 1, under = TRUE, split.font = 2, varlen = -10)
#dev.copy2pdf(file = "E:/Data mining/Lecture Notes/plots/rt1.pdf")

# get prediction on the test data
pred.test = predict(ct1, mydata.test)
# get MSE on test data
rmse_test <- sqrt(mean((mydata.test$mpg - pred.test)^2))

# fit regression tree using cost complexity cross validation
library(caret)
set.seed(0)
train_control <- trainControl(method="cv", number=10)
cv.ct <- train(mpg ~ . , data = mydata.train, method = "rpart",
               trControl = train_control, tuneLength = 10)
print(cv.ct)
cv.ct$finalModel
prp(cv.ct$finalModel, type = 1, extra = 1, under = TRUE, split.font = 2, varlen = -10)
#dev.copy2pdf(file = "E:/Data mining/Lecture Notes/plots/rt2.pdf")

# get prediction on the test data
pred.test = predict(cv.ct$finalModel, mydata.test)
# get MSE on test data
rmse_test_prune <- sqrt(mean((mydata.test$mpg - pred.test)^2))