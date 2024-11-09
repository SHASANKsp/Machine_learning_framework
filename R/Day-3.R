train = read.csv("datasets/titanic/train.csv",header = T)
dim(train)
head(train)
str(train)
train$Survived = as.factor(train$Survived)
train$Pclass = as.factor(train$Pclass)
#install.packages("randomForest")
library(randomForest)


rfModel = randomForest(Survived ~ Sex+Pclass+Embarked,
                       importance = TRUE, 
                       data = train3)
rfModel

dim(train)
length(train$Age)
length(na.omit(train$Age))
891-714

rfModel
cf = rfModel$confusion[,c(1,2)]
cf
sum(diag(cf))/sum(cf)

(523+200)/(523+26+142+200)
dim(rfModel$err.rate)
(rfModel$votes)
rfModel$importance

train2 = train[,c("Sex","Pclass","Embarked","Age", "Survived")]
dim(train2)
head(train2)   
summary(train2)

train3 = na.omit(train2)
dim(train3)
summary(train3)

rfModel = randomForest(Survived ~ Sex+Pclass+Embarked+Age,
                       importance = TRUE, 
                       ntree = 300,
                       data = train3)

acc1 = c()
for(i in 1:10){
   rfModel2 = randomForest(Survived ~ Sex+Pclass+Embarked+Age,
                           importance = TRUE, data = train3)
   cf = rfModel2$confusion[,c(1,2)]
   acc1 = c(acc1, sum(diag(cf))/sum(cf))
}
acc1

acc2 = c()
for(i in 1:10){
   rfModel3 = randomForest(Survived ~ Sex+Pclass+Age,
                           importance = TRUE, data = train3)
   cf = rfModel3$confusion[,c(1,2)]
   acc2 = c(acc2, sum(diag(cf))/sum(cf))
}
acc2


acc3 = c()
for(i in 1:10){
   rfModel4 = randomForest(Survived ~ Sex,
                           importance = TRUE, data = train3)
   cf = rfModel4$confusion[,c(1,2)]
   acc3 = c(acc3, sum(diag(cf))/sum(cf))
}
acc3

acc4 = c()
for(i in 1:10){
   rfModel5 = randomForest(Survived ~ Sex+Pclass,
                           importance = TRUE, data = train3)
   cf = rfModel5$confusion[,c(1,2)]
   acc4 = c(acc4, sum(diag(cf))/sum(cf))
}
acc3

boxplot(acc1, acc2, acc3, acc4, ylim = c(0.75, 0.85))


cf = rfModel3$confusion[,c(1,2)]
sum(diag(cf))/sum(cf)

rfModel2$importance
varImpPlot(rfModel2)

#install.packages("caret")
library(caret)
trCt = trainControl(method = "cv", n=10)
rf1 = train(Survived ~ Sex, data = train3,
            method = "rf", trControl = trCt)
rf1$resample

rf2 = train(Survived ~ Sex+Pclass, data = train3,
            method = "rf", trControl = trCt)
rf2$resample

rf3 = train(Survived ~ Sex+Embarked, data = train3,
            method = "rf", trControl = trCt)
rf3$resample

rf4 = train(Survived ~ Sex+Age, data = train3,
            method = "rf", trControl = trCt)
rf4$resample

rf5 = train(Survived ~ Sex+Pclass+Embarked, data = train3,
            method = "rf", trControl = trCt)
rf5$resample

rf6 = train(Survived ~ Sex+Pclass+Embarked, data = train,
            method = "rf", trControl = trCt)
rf6$resample

boxplot(rf1$resample$Accuracy,
        rf2$resample$Accuracy,
        rf3$resample$Accuracy,
        rf4$resample$Accuracy, 
        rf5$resample$Accuracy, 
        rf6$resample$Accuracy, ylim=c(0.65,0.9))
