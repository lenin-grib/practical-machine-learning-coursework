

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
library(caret)
library(rpart) 
library(rpart.plot) 
library(rattle)
library(e1071)

##prep  https://rpubs.com/lgreski/improvingCaretPerformance
set.seed(29)
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

tc <- trainControl(method = "cv", number = 5, 
        allowParallel=TRUE)

### modelling
mrfst <- system.time (mrf <- train(classe ~., 
        data = training, method = "rf", trControl= tc))
mgbmst <- system.time (mgbm <- train(classe ~., 
        data = training, method = "gbm", 
        verbose = FALSE, trControl= tc))
mrpst <- system.time (mrp <- train(classe ~., 
        data = training, method = "rpart", 
        trControl= tc))
msvmst <- system.time (msvm <- svm(classe ~ ., 
        data = training, trControl= tc))

### shut cluster down
stopCluster(cluster)
registerDoSEQ()

### cross-validate

prf <- predict(mrf, validation)
pgbm <- predict(mgbm, validation)
prp <- predict(mrp, validation)
psvm <- predict(msvm, validation)

### comparison

comp1 <- c("random forest", 
        confusionMatrix(prf, validation$classe)$overall[1],
        mrfst[2])
comp2 <- c("boosting", 
        confusionMatrix(pgbm, validation$classe)$overall[1],
        mgbmst[2])
comp3 <- c("decision tree", 
        confusionMatrix(prp, validation$classe)$overall[1],
        mrpst[2])
comp4 <- c("svm", 
        confusionMatrix(psvm, validation$classe)$overall[1],
        msvmst[2])

### formatting
library(knitr)
library(kableExtra)
library(dplyr)

comp <- data.frame(matrix(ncol = 3, nrow = 0))
comp <- rbind(comp, comp1, comp2, comp3, comp4)
x <- c("method", "accuracy", "running time")
colnames(comp) <- x

kable(comp) %>%
        kable_styling(bootstrap_options = c("striped", "hover"), 
                full_width = F)

##confusionMatrix(prf, validation$classe)$overall[1]
##confusionMatrix(pgbm, validation$classe)$overall[1]
##confusionMatrix(pcl, validation$classe)$overall[1]
##confusionMatrix(psvm, validation$classe)$overall[1]

## finals

finalmodel <- mrf
1-as.numeric(as.character(comp[1,2]))


## try stacking
stacked <- data.frame(prf, pgbm, psvm, training$classe)

mstackedlm <- train(validation.classe ~ ., data = stacked, method = "glm")
mstackedrf <- train(validation.classe ~ ., data = stacked, method = "rf")
pstackedlm <- predict(mstackedlm, stacked)
pstackedrf <- predict(mstackedrf, stacked)

confusionMatrix(pstackedlm, validation$classe)$overall[1]
confusionMatrix(pstackedrf, validation$classe)$overall[1]





















