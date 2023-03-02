rm(list = ls())
##########################
### knn for regression ###
##########################

autompg <- read.csv("autompg.csv")

## Include the functions required for data partitioning
source("myfunctions.R")

RNGkind (sample.kind = "Rounding") 
set.seed(0)
### call the function for creating 60:30:10 partition
p3 <- partition.3(autompg, 0.6, 0.3)
training.data <- p3$data.train
validation.data <- p3$data.val
test.data <- p3$data.test

##################################################################
### Rescale the data for the function knn.reg======================
##################################################################
training.scaled <- scale(training.data[,-1], center = TRUE, scale = TRUE)
training.scaled.wY <- cbind(training.scaled, training.data[,1])
training.scaled.attr <- attributes(training.scaled)
val.scaled <- scale(validation.data[,-1], 
                    center = training.scaled.attr$`scaled:center`, 
                    scale = training.scaled.attr$`scaled:scale`)
test.scaled <- scale(test.data[,-1], 
                     center = training.scaled.attr$`scaled:center`, 
                     scale = training.scaled.attr$`scaled:scale`)



##################################################################
### Fit kNN model on validation data with k=5======================
##################################################################


library(FNN)
Knn <- knn.reg(train = training.scaled, test = val.scaled,
               y = training.data[,1], k = 5)
Knn

# RMSE for validation data
error.val.knn <- Knn$pred - validation.data$mpg
rmse.val.knn <- sqrt(mean(error.val.knn^2))
rmse.val.knn


##################################################################
### Use K-fold Cross Validation to tune the parameter K ==========
##################################################################

# value of K equal to 10 
set.seed(0)
train_control <- trainControl(method = "cv", 
                              number = 10) 
training.data.all <- rbind(training.data, validation.data)

# Fit K-fold CV model  

# Knn_kcv <- train(mpg ~ ., data = training.data.all, method = "knn", 
#                  trControl = train_control, preProcess = c("center","scale"), 
#                  tuneLength = 20, metric = "RMSE")
# print(Knn_kcv)


# Specify tuning parameter using tuneGrid====================

tg <- data.frame(k = seq(1,350,5))
Knn_kcv <- train(mpg ~ ., data = training.data.all, method = "knn",
                 trControl = train_control, preProcess = c("center","scale"),
                 tuneGrid = tg, metric = "RMSE")
print(Knn_kcv)

plot(Knn_kcv)
Knn_kcv$finalModel


### fit k-nn model on test data with k=11
training.data.scaled.all <- rbind(training.scaled, val.scaled)
Knn <- knn.reg(train = training.data.scaled.all, test = test.scaled,
               y = training.data.all[,1], k = 11)
# RMSE for test data
error.test.knn <- Knn$pred - test.data$mpg
rmse.test.knn <- sqrt(mean(error.test.knn^2))
rmse.test.knn
