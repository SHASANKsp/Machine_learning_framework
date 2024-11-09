install.packages("rpart")
library("rpart")

install.packages("randomForest")
library(randomForest)

train.data = read.csv("train.csv")
train.data$Survived = as.factor(train.data$Survived)
train.data$Embarked = as.factor(train.data$Embarked)
train.data$Sex = as.factor(train.data$Sex)
train.data$Pclass = as.factor(train.data$Pclass)
train.data$Pclass = as.factor(train.data$Pclass)

rfm1 = randomForest(Survived ~ Sex+Pclass+Embarked,
                    importance = TRUE,
                    data = train.data)
rfm1
cf = rfm1$confusion[,c(1,2)]
cf
sum(diag(cf))/sum(cf)

rfm1$importance


train2 = train.data[,c("Sex","Pclass","Embarked","Age", "Survived")]
train3 = na.omit(train2)

rfModel = randomForest(Survived ~ Sex+Pclass+Embarked+Age,
                       importance = TRUE,
                       data = train3)
rfModel
rfModel$importance
varImpPlot(rfModel)


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
acc4

boxplot(acc1, acc2, acc3, acc4, ylim = c(0.78, 0.82))


install.packages("caret")
library(caret)

trCt = trainControl(method = "cv", n=10)
rf1 = train(Survived ~ Sex, data = train3,
            method = "rf",trControl = trCt)
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

rf6 = train(Survived ~ Sex+Pclass+Embarked, data = train.data,
            method = "rf", trControl = trCt)
rf6

boxplot(rf1$resample$Accuracy,
        rf2$resample$Accuracy,
        rf3$resample$Accuracy,
        rf4$resample$Accuracy, 
        rf5$resample$Accuracy, 
        rf6$resample$Accuracy, ylim=c(0.65,0.9))



#liver_dataset

ld=read.csv("Liver_Patients.csv")
str(ld)
summary(ld)

rfld1 = randomForest(Dataset ~ Age+Total_Bilirubin+Total_Protiens,
                    importance = TRUE,
                    data = ld)

rfld1
rfld1$confusion
rfld1$importance


ld2 = train(Dataset ~ Age+Total_Bilirubin+Total_Protiens,
            data = ld,
            method = "rf", trControl = trCt)

ld2

