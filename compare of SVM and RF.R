install.packages("dplyr")
install.packages("e1071")
install.packages("randomForest")
install.packages("rpart.plot")
install.packages("caret")
# we need dplyr and e1071 for SVM/RF
library(dplyr)
library(e1071)
library(MASS)
# envoke RF library
library(randomForest)
# Regression/classification tree library
library(rpart)
# plotting trees
library(rpart.plot)
# enable caret for CV
library(caret)


# read the data:
path="/path/to/"
fname="trees.csv"
trees=read.csv(paste(path, fname, sep=""))

# set TreeNumber as rownames then remove TreeNumber from the dataset
rownames(trees)=trees$TreeNumber
trees[,1]=NULL
# correct factor variables:
trees$Type=as.factor(trees$Type)

set.seed(321)
# work with a random sample subset of the data to speed things up:
trees.sub = trees[sample(1:nrow(trees), 1000, replace=FALSE),]
# Problem here: because we are doing subsetting, we end up with empty classes.
# Remove rare groups (n<10) from data:
trees.sub.rm <- trees.sub %>% group_by(Type) %>% filter(n() >= 10)
# expunge empty levels from factor Type
trees.sub.rm$Type=factor(trees.sub.rm$Type)


#  fit RF to Type on all predictor vars:
# ntree = number of decision trees to grown, mtry = number of subset variables per split
set.seed(321)
fit <- randomForest(Type~.,data=trees.sub.rm,ntree=5000,mtry=2)
# make predictions of tree types and test accuracy 
oob=fit$err.rate[,1]
print(summary(oob))
pred=predict(fit,newdata=trees.sub.rm,type="class")
confusionMatrix(trees.sub.rm$Type,pred)$overall#in the bag->overfitting(100%)

fit2 <- randomForest(Type~.,data=trees.sub.rm,ntree=5000,mtry=2)
oob2=fit2$err.rate[,1]
print(summary(oob2))#random error, since each fit randomly chooses data to build the model
#so result may vary across different data selected


#train_control <- trainControl(method = "repeatedcv",number=ncv,repeats=2)   
train_control <- trainControl(method = "cv",number=10)   
# Set required parameters for the model type we are using
tune_grid = expand.grid(mtry=c(2,3,4,5,6))#mtry gives numbers of candidates to be selected and put on the split point
# Use the train() function to create the model
system.time(cv_fit <- train(Type~.,
                data=trees.sub.rm,                 # Data set
                method="rf",                     # Model type(decision tree)
                trControl= train_control,           # Model control options
                tuneGrid = tune_grid,
                ntree = 5000))               # Additional parameter
plot(varImp(cv_fit))#relative scale of importance(comparing with each other)


# tune SVM using train() under caret
C=10^(-1:5)
G=2^seq(-15,3,2)
ncv<-10
tune_grid <- expand.grid(C = C,sigma=G)
train_control <- trainControl(method = "cv",number=ncv)   
system.time(svmtry<- train(Type~.,data=trees.sub.rm,
                            method = "svmRadial",   # Choose kernel?
                            tuneGrid = tune_grid,
                            trControl=train_control)) #,epsilon = epsilon
print(svmtry$result[order(svmtry$results$Accuracy),])

grid2=expand.grid(C=10, sigma=0.03125)
system.time(fit3 <- randomForest(Type~.,data=trees.sub.rm,ntree=5000,mtry=4))
system.time(svmtry2<- train(Type~.,data=trees.sub.rm,
                           method = "svmRadial",   # Choose kernel?
                           tuneGrid = grid2,
                           trControl=train_control))










