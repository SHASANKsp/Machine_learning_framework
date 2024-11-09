install.packages("randomForest")
#getwd()
#setwd()
train = read.csv("heart.csv")
head(train)
train$sex = as.factor(train$sex)
train$cp = as.factor(train$cp)
train$fbs= as.factor(train$fbs)
train$exng = as.factor(train$exng)
train$output = as.factor(train$output)
str(train)

library("randomForest")


rfModel = randomForest(output ~ age+exng+cp+chol+caa+oldpeak+thall,
                       importance = TRUE,
                       data = train)
rfModel
cf = rfModel$confusion[,c(1,2)]
cf
sum(diag(cf))/sum(cf)

(rfModel$importance)
varImpPlot(rfModel)
install.packages("caret")
library(caret)
trCt = trainControl(method = "cv", n=10)
rf1 = train(output ~ cp, data = train,
            method = "rf", trControl = trCt)
rf1$resample

rf2 = train(output ~ age+cp, data = train,
            method = "rf", trControl = trCt)
rf2$resample

rf3 = train(output~ oldpeak+cp, data = train,
            method = "rf", trControl = trCt)
rf3$resample

rf4 = train(output~ age+oldpeak+cp, data = train,
            method = "rf", trControl = trCt)
rf4$resample

rf5 = train(output~ age+exng+chol, data = train,
            method = "rf", trControl = trCt)
rf5$resample

rf6 = train(output ~ age+cp+caa, data = train,
            method = "rf", trControl = trCt)
rf6$resample
rf7 = train( output~ age+cp+caa+oldpeak+thall,data = train,
             method = "rf", trControl = trCt)
rf7$resample

boxplot(rf1$resample$Accuracy,
        rf2$resample$Accuracy,
        rf3$resample$Accuracy,
        rf4$resample$Accuracy, 
        rf5$resample$Accuracy, 
        rf6$resample$Accuracy,
        rf7$resample$Accuracy,
        ylim=c(0.5,1))
