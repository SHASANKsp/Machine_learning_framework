#setwd("/Users/pankazsingh/Library/Mobile Documents/com~apple~CloudDocs/Teaching/ML-2021/Lab")
setwd("E:/Teaching/ML-2021/Lab")

####################################################
# SVM
####################################################

library(e1071)

set.seed(10)
sample(1:100, 2)


set.seed(10)
x1 = c(sample(1:10, 10, replace = T),
       sample(30:50, 10, replace = T))

set.seed(20)
x2 = c(sample(1:10, 10, replace = T),
       sample(30:50, 10, replace = T))

plot(x1,x2)

data = data.frame(X1 = c(x1),
                  X2 = c(x2), 
                  Labels = c(rep("yes",10),rep("no",10)))
data$Labels = as.factor(data$Labels)
data
str(data)
plot(data$X1, data$X2,  col = data$Labels, type = "p", pch = 1)
#?plot()

fit.svm = svm(Labels~., data = data, kernel = "linear")
fit.svm
summary(fit.svm)
plot(fit.svm, data)


fit.svm = svm(Labels~., data = data, kernel = "linear",
              cost = 0.3)
summary(fit.svm)
plot(fit.svm, data)


fit.svm = svm(Labels~., data = data, kernel = "linear", cost = 1, gamma = 0.1)
summary(fit.svm)
plot(fit.svm, data)


fit.svm = svm(Labels~., data = data, kernel = "polynomial")
fit.svm
summary(fit.svm)
plot(fit.svm, data)


pred = predict(fit.svm,data$Labels)
# or
pred = fitted(fit.svm)

table(pred,data$Labels)
# compute decision values and probabilities:
pred <- predict(fit.svm, data[,-3], decision.values = TRUE)
attr(pred, "decision.values")

fit.svm2 = train(Labels~ ., data = data, method = "svmLinear", trControl = trcntrl)
plot((fit.svm2$finalModel), data)
kernlab::plot(fit.svm2$finalModel)




iris
iris.part = subset(iris, Species != 'setosa')
iris.part$Species = factor(iris.part$Species)
iris.part = iris.part[, c(1,2,5)]
dim(iris.part)
fiti = svm(Species ~ ., data=iris.part, type='C-classification', 
          kernel='polynomial')
?svm
plot(fiti, iris.part)


####################################################


###################################################
# Diabetes Data

#Attributes:

# Pregnancies: Number of times pregnant
# Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml)
# BMI: Body mass index (weight in kg/(height in m)^2)
# DiabetesPedigreeFunction: Diabetes pedigree function 
# (a function that represents how likely they are to get 
# the disease by extrapolating from their ancestor???s history)
# Age: Age (years)
# Outcome: Class variable (0 or 1)
####################################################


library(caret)
install.packages("caret", dependencies = T)
library(kernlab)
dbts = read.csv("datasets/diabetes.csv", header = T)
dim(dbts)
head(dbts)
summary(dbts)
dbts$Outcome = as.factor(dbts$Outcome)

model.svm = train(Outcome~., data = dbts, method ="svmLinear")
model.svm



grid = data.frame(C = c(0.01,0.05,0.2,0.5,1,1.5,2.0))
trcntrl = trainControl(method = "cv", number = 10)
#trcntrl = trainControl(method = "boot", number = 100)
model.dbts.svm = train(Outcome~ ., data = dbts, 
                       method = "svmLinear", trControl = trcntrl, 
                       tuneGrid = grid, gamma = 0.05)
model.dbts.svm
model.dbts.svm$finalModel
model.dbts.svm$results
model.dbts.svm$resample

plot(model.dbts.svm$results$C, model.dbts.svm$results$Accuracy, type = "b")

grid1 = expand.grid(C = c(0.5,1), sigma = c(.01, .015, 0.2))
#grid1 = expand.grid(degree = c(3,4,5), scale = c(.01, .015, 0.2), C = c(0.5,1))
trcntrl = trainControl(method = "cv", number = 10)
#trcntrl = trainControl(method = "boot", number = 100)
model.dbts.svm = train(Outcome~ ., data = dbts, 
                       method = "svmPoly", trControl = trcntrl, 
                       tuneGrid = grid1, gamma = 0.5)


model.dbts.svm = train(Outcome~ ., data = dbts, 
                       method = "svmRadial", trControl = trcntrl, 
                       tuneGrid = grid1, gamma = 0.5)

####################################################


fit.svm2 = train(Outcome~ ., data = dbts, method = "svmLinear", trControl = trcntrl)

names(getModelInfo())


