setwd("/Users/binni/Documents/pankaz sir")
data2 = read.csv("water_potability.csv",header = T, stringsAsFactors = FALSE)
dim(data2)
head(data2)

table(data[,c('target', 'sex')])


data3 = na.omit(data2)


data3$Potability= as.factor(data3$Potability)


###RANDOM Forest####

library(randomForest)

rfModel1 = randomForest(Potability ~ Hardness+ph+Sulfate+Conductivity+Chloramines+Solids+Organic_carbon,
                       importance = TRUE, 
                       ntree = 300,
                       data = data3)



cf = rfModel1$confusion[,c(1,2)]
acc = sum(diag(cf))/sum(cf)*100
acc

rfModel2 = randomForest(Potability ~ Hardness+ph+Sulfate+Conductivity,
                       importance = TRUE, 
                       ntree = 200,
                       data = data3)


cf2 = rfModel2$confusion[,c(1,2)]

acc2 = sum(diag(cf2))/sum(cf2)*100
acc2
 
TP =cf2[1,1]
TN =cf2[2,2]
FN =cf2[2,1]
FP =cf2[1,2]

precision = TP/(TP+FP)
recall = TP/(TP+FN)


library(caret)

t2 = trainControl(method = "cv", n=10)

rf1 = train(Potability ~ Hardness+ph+Sulfate+Conductivity, data = data3,
            method = "rf", trControl = t2,
            tuneGrid = expand.grid(.mtry =2),ntree = 300)




rf2 = train(Potability ~ Hardness+ph+Sulfate+Conductivity, data = data3,
            method = "rf", trControl = t2,
            tuneGrid = expand.grid(.mtry =4),ntree = 500)







###K Nearest Neighbor #######

install.packages("e1071")
install.packages("caTools")
install.packages("class")

library(caTools)
split <- sample.split(data3, SplitRatio = 0.7)

train_cl <- subset(data3, split == "TRUE")
test_cl <- subset(data3, split == "FALSE")

             
train_scale <- scale(train_cl[, 1:5])
test_scale <- scale(test_cl[, 1:5])


install.packages("class",dependencies = TRUE)

library(e1071)
library(caTools)
library(class)

install.packages("e1071")
library(el071)


classifier_knn <- knn(train = train_scale,
                      test = test_scale,
                      cl = train_cl$Potability,
                      k = 8)

# Confusiin Matrix

cm <- table(test_cl$Potability, classifier_knn)
cm

TP2 =cm[1,1]
TN2 =cm[2,2]
FN2 =cm[2,1]
FP2 =cm[1,2]

acc2 = (TP2+TN2)/(TP2+TN2+FN2+FP2)
precision2 = TP2/(TP2+FP2)
recall2 = TP2/(TP2+FN2)




##tunable parameter ###

knn1 = train(Potability ~ Hardness+ph+Sulfate+Conductivity, data = data3,
             method = "knn", trControl = t2,
             tuneGrid = expand.grid(k = seq(1, 101, by = 2))
             
)

knn3 = train(Potability ~ Hardness+ph+Sulfate+Conductivity, data = data3,
             method = "knn")






plot(knn1$results$k, knn1$result$Accuracy)









boxplot(rf1$resample$Accuracy,
        rf2$resample$Accuracy,
        rf3$resample$Accuracy)














