library(caret)

trdat = read.csv("pml-training.csv")
testdata = read.csv("pml-testing.csv")

set.seed(29)

## partition

inTrain = createDataPartition(trdat$classe, 
        p = 3/4)[[1]]
training = trdat[inTrain,]
validation = trdat[-inTrain,]

## clean

### NA 90%

thres = 0.9
NAcols <- apply(training, 2,  
        FUN = function(x){sum(is.na(x)) > (nrow(training)*thres)})
sum(NAcols)
NNAcols <- names(subset(NAcols, NAcols == F))
training <- training[NNAcols]

### NZV

NZVcols <- nearZeroVar(training)
training <- training[-NZVcols]

### info

training <- training[-c(1:6)]

dim(training)

### apply
trvars <- names(training)
tstvars <- names(training[,-53])
validation <- validation [trvars]
testing <- testdata[tstvars]

## modelling
library(rpart) 
library(rpart.plot) 
library(rattle)
library(e1071)

set.seed(29)

mrf <- train(classe ~., data = training, method = "rf")
mgbm <- train(classe ~., data = training, method = "gbm", 
        verbose = FALSE)
mlda <- train(classe ~., data = training, method = "lda", 
        verbose = FALSE)
msvm <- svm(classe ~ ., data = training)

prf <- predict(mrf, validation)
pgbm <- predict(mgbm, validation)
plda <- predict(mlda, validation)
psvm <- predict(msvm, validation)

confusionMatrix(prf, validation$classe)$overall[1]
confusionMatrix(pgbm, validation$classe)$overall[1]
confusionMatrix(plda, validation$classe)$overall[1]
confusionMatrix(psvm, validation$classe)$overall[1]

pdf <- data.frame(prf, pgbm, plda, validation$classe)

mstacked <- train(validation$classe ~ ., data = pdf, method = "rf")

pstacked <- predict(mstacked, pdf)

confusionMatrix(pstacked, validation$classe)$overall[1]





















