library(caret)
library(kernlab)
train = read.csv("lung_cancer_examples.csv", header = T)
#dim(train)
#head(train)
#summary(train)

train$Result = as.factor(train$Result)
train$Age = as.factor(train$Age)
train$Smokes = as.factor(train$Smokes)
train$AreaQ = as.factor(train$AreaQ)
train$Alkhol = as.factor(train$Alkhol)

model.svm = train(Result~., data = train, method ="svmLinear")
model.svm

plot(model.svm, train)

grid = data.frame(C = c(0.01,0.05,0.2,0.5,1,1.5,2.0))
trcntrl = trainControl(method = "cv", number = 10)
#trcntrl = trainControl(method = "boot", number = 100)
model.train.svm = train(Result~ ., data = train, 
                       method = "svmLinear", trControl = trcntrl, 
                       tuneGrid = grid, gamma = 0.05)
model.train.svm
model.train.svm$finalModel
model.train.svm$results
model.train.svm$resample

plot(model.train.svm$results$C, model.train.svm$results$Accuracy, type = "b")

grid1 = expand.grid(C = c(0.5,1), sigma = c(.01, .015, 0.2))

trcntrl = trainControl(method = "cv", number = 10)

model.train.svm = train(Result~ ., data = train, 
                       method = "svmPoly", trControl = trcntrl, 
                       tuneGrid = grid1, gamma = 0.5)


model.train.svm = train(Result~ ., data = train, 
                       method = "svmRadial", trControl = trcntrl, 
                       tuneGrid = grid1, gamma = 0.5)





##########################################################

#trying it simple

train = read.csv("lung_cancer_examples.csv", header = T)

train$Result = as.factor(train$Result)
train$Age = as.factor(train$Age)
train$Smokes = as.factor(train$Smokes)
train$AreaQ = as.factor(train$AreaQ)
train$Alkhol = as.factor(train$Alkhol)


fit = svm(Result ~ ., data=train, type='C-classification', 
          kernel='linear')
#?svm
plot(fit, train)
