---
title: "Practical Machine Learning: detecting correct barbell performance based on health trackers data"
author: "Daria Stepanyan"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---



## Executive summary

In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Several machine learning algorythms were used to choose a final model; cross-validation analysis showed that model produced by the random forest method was the most accurate.

## Loading data

The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
trdat = read.csv("pml-training.csv")
testdata = read.csv("pml-testing.csv")
```

First of all we'll separate a portion of data which will be used for cross validation


```r
library(caret)
set.seed(29)
inTrain = createDataPartition(trdat$classe, 
        p = 3/4)[[1]]
training = trdat[inTrain,]
validation = trdat[-inTrain,]
```

## Exploratory analysis and cleaning



```r
dim(training)
```

```
## [1] 14718   160
```

The data has a lot of variables, so before fitting a model it makes sense to reduce their amount and leave only relevant ones. 

After performing some exploratory analysis (outputs are moved to Appendix 1 due to significant size) we can drop columns a) with mostly NA value; b) with negligible variance (most probably not good predictors) c) informational fields, such as observation index, name, etc.

### Removing the columns with little known data

By changing the threshold below we can see that certain amount of columns have more than 95% of the values missing, whereas rest of the data is quite full. Setting the threshold to 90% just for very conservative removal.


```r
thres = 0.9
NAcols <- apply(training, 2,  
        FUN = function(x){sum(is.na(x)) > (nrow(training)*thres)})
NNAcols <- names(subset(NAcols, NAcols == F))
training <- training[NNAcols]
```

67 columns gone.

### Removing predictors with low variablility


```r
NZVcols <- nearZeroVar(training)
training <- training[-NZVcols]
```

### Removing informational fields


```r
training <- training[-c(1:6)]
```

### Tidying up

```r
dim(training)
```

```
## [1] 14718    53
```

Let's apply performed changes to our validation and test datasets for future comparison

```r
trvars <- names(training)
tstvars <- names(training[,-53])
validation <- validation [trvars]
testing <- testdata[tstvars]
```

## Finding a best-fit model

For the comparison 4 algorythms were chosen as quite popular for classification: decision tree, random forest, boosting and svm.

### Preparation for model fitting

```r
library(rpart) 
library(rpart.plot) 
library(rattle)
library(e1071)
```

Next chunk is setting the seed and preparing the machine for performing calculations more efficiently. (Thanks to the author of this article: https://rpubs.com/lgreski/improvingCaretPerformance code for modelling runs manageable amount of time)


```r
set.seed(29)
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

tc <- trainControl(method = "cv", number = 5, 
        allowParallel=TRUE)
```

### Models

Code below fits the model using different methods and gets the running time for each.


```r
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
```

Shutting the cluster down

```r
stopCluster(cluster)
registerDoSEQ()
```

### Cross-validation
For the evaluation of the fitted models' efficiency we use our wisely prepared validation set and combine it into a table.


```r
prf <- predict(mrf, validation)
pgbm <- predict(mgbm, validation)
prp <- predict(mrp, validation)
psvm <- predict(msvm, validation)

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
```

<table class="table table-striped table-hover" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> method </th>
   <th style="text-align:left;"> accuracy </th>
   <th style="text-align:left;"> running time </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> random forest </td>
   <td style="text-align:left;"> 0.992862969004894 </td>
   <td style="text-align:left;"> 1.98 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> boosting </td>
   <td style="text-align:left;"> 0.96594616639478 </td>
   <td style="text-align:left;"> 0.43 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> decision tree </td>
   <td style="text-align:left;"> 0.498776508972268 </td>
   <td style="text-align:left;"> 0.22 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> svm </td>
   <td style="text-align:left;"> 0.954323001631321 </td>
   <td style="text-align:left;"> 0.19 </td>
  </tr>
</tbody>
</table>

## Conclusion

According to our test, random forest method did the best in terms of predicting values for the validation dataset with predicted out of sample error of 0.007137. Boosting came a close second with the OOS error of 0.0340538, but MUCH better performance. Since the end goal of this excercise is to predict values in test dataset as accurate as possible, let's choose the model produced by random forest method as our final, but with the note that boosting is probably preferrable to use on large sets of data.

******

# Appendix

## Apendix 1. Training dataset structure


```r
names(trdat[inTrain,])
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```

```r
head(trdat[inTrain,])
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
## 7 7  carlitos           1323084232               368296 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 2         no         11      1.41       8.07    -94.4                3
## 3         no         11      1.42       8.07    -94.4                3
## 4         no         12      1.48       8.05    -94.4                3
## 5         no         12      1.48       8.07    -94.4                3
## 6         no         12      1.45       8.06    -94.4                3
## 7         no         12      1.42       8.09    -94.4                3
##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 2                                                         
## 3                                                         
## 4                                                         
## 5                                                         
## 6                                                         
## 7                                                         
##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 2                                                                      NA
## 3                                                                      NA
## 4                                                                      NA
## 5                                                                      NA
## 6                                                                      NA
## 7                                                                      NA
##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 2             NA                         NA             NA             
## 3             NA                         NA             NA             
## 4             NA                         NA             NA             
## 5             NA                         NA             NA             
## 6             NA                         NA             NA             
## 7             NA                         NA             NA             
##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 2                  NA                   NA                   
## 3                  NA                   NA                   
## 4                  NA                   NA                   
## 5                  NA                   NA                   
## 6                  NA                   NA                   
## 7                  NA                   NA                   
##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 2                   NA            NA               NA            NA
## 3                   NA            NA               NA            NA
## 4                   NA            NA               NA            NA
## 5                   NA            NA               NA            NA
## 6                   NA            NA               NA            NA
## 7                   NA            NA               NA            NA
##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 2             NA                NA             NA           NA
## 3             NA                NA             NA           NA
## 4             NA                NA             NA           NA
## 5             NA                NA             NA           NA
## 6             NA                NA             NA           NA
## 7             NA                NA             NA           NA
##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 2              NA           NA         0.02         0.00        -0.02
## 3              NA           NA         0.00         0.00        -0.02
## 4              NA           NA         0.02         0.00        -0.03
## 5              NA           NA         0.02         0.02        -0.02
## 6              NA           NA         0.02         0.00        -0.02
## 7              NA           NA         0.02         0.00        -0.02
##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 2          -22            4           22            -7           608
## 3          -20            5           23            -2           600
## 4          -22            3           21            -6           604
## 5          -21            2           24            -6           600
## 6          -21            4           21             0           603
## 7          -22            3           21            -4           599
##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 2          -311     -128      22.5    -161              34            NA
## 3          -305     -128      22.5    -161              34            NA
## 4          -310     -128      22.1    -161              34            NA
## 5          -302     -128      22.1    -161              34            NA
## 6          -312     -128      22.0    -161              34            NA
## 7          -311     -128      21.9    -161              34            NA
##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
## 2           NA              NA           NA            NA               NA
## 3           NA              NA           NA            NA               NA
## 4           NA              NA           NA            NA               NA
## 5           NA              NA           NA            NA               NA
## 6           NA              NA           NA            NA               NA
## 7           NA              NA           NA            NA               NA
##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
## 2            NA          NA             NA          NA        0.02
## 3            NA          NA             NA          NA        0.02
## 4            NA          NA             NA          NA        0.02
## 5            NA          NA             NA          NA        0.00
## 6            NA          NA             NA          NA        0.02
## 7            NA          NA             NA          NA        0.00
##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
## 2       -0.02       -0.02        -290         110        -125         -369
## 3       -0.02       -0.02        -289         110        -126         -368
## 4       -0.03        0.02        -289         111        -123         -372
## 5       -0.03        0.00        -289         111        -123         -374
## 6       -0.03        0.00        -289         111        -122         -369
## 7       -0.03        0.00        -289         111        -125         -373
##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
## 2          337          513                                     
## 3          344          513                                     
## 4          344          512                                     
## 5          337          506                                     
## 6          342          513                                     
## 7          336          509                                     
##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
## 2                                                                       
## 3                                                                       
## 4                                                                       
## 5                                                                       
## 6                                                                       
## 7                                                                       
##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
## 2           NA            NA          NA           NA            NA
## 3           NA            NA          NA           NA            NA
## 4           NA            NA          NA           NA            NA
## 5           NA            NA          NA           NA            NA
## 6           NA            NA          NA           NA            NA
## 7           NA            NA          NA           NA            NA
##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
## 2          NA                 NA                  NA                NA
## 3          NA                 NA                  NA                NA
## 4          NA                 NA                  NA                NA
## 5          NA                 NA                  NA                NA
## 6          NA                 NA                  NA                NA
## 7          NA                 NA                  NA                NA
##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
## 2      13.13074      -70.63751    -84.71065                       
## 3      12.85075      -70.27812    -85.14078                       
## 4      13.43120      -70.39379    -84.87363                       
## 5      13.37872      -70.42856    -84.85306                       
## 6      13.38246      -70.81759    -84.46500                       
## 7      13.12695      -70.24757    -85.09961                       
##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
## 2                                                                     
## 3                                                                     
## 4                                                                     
## 5                                                                     
## 6                                                                     
## 7                                                                     
##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
## 2                                                              NA
## 3                                                              NA
## 4                                                              NA
## 5                                                              NA
## 6                                                              NA
## 7                                                              NA
##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
## 2                 NA                                 NA                 NA
## 3                 NA                                 NA                 NA
## 4                 NA                                 NA                 NA
## 5                 NA                                 NA                 NA
## 6                 NA                                 NA                 NA
## 7                 NA                                 NA                 NA
##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
## 2                                       NA                       NA
## 3                                       NA                       NA
## 4                                       NA                       NA
## 5                                       NA                       NA
## 6                                       NA                       NA
## 7                                       NA                       NA
##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
## 2                                          37                 NA
## 3                                          37                 NA
## 4                                          37                 NA
## 5                                          37                 NA
## 6                                          37                 NA
## 7                                          37                 NA
##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
## 2                NA                   NA                NA
## 3                NA                   NA                NA
## 4                NA                   NA                NA
## 5                NA                   NA                NA
## 6                NA                   NA                NA
## 7                NA                   NA                NA
##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
## 2                 NA                    NA                 NA
## 3                 NA                    NA                 NA
## 4                 NA                    NA                 NA
## 5                 NA                    NA                 NA
## 6                 NA                    NA                 NA
## 7                 NA                    NA                 NA
##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
## 2               NA                  NA               NA                0
## 3               NA                  NA               NA                0
## 4               NA                  NA               NA                0
## 5               NA                  NA               NA                0
## 6               NA                  NA               NA                0
## 7               NA                  NA               NA                0
##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
## 2            -0.02             0.00             -233               47
## 3            -0.02             0.00             -232               46
## 4            -0.02            -0.02             -232               48
## 5            -0.02             0.00             -233               48
## 6            -0.02             0.00             -234               48
## 7            -0.02             0.00             -232               47
##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
## 2             -269              -555               296               -64
## 3             -270              -561               298               -63
## 4             -269              -552               303               -60
## 5             -270              -554               292               -68
## 6             -269              -558               294               -66
## 7             -270              -551               295               -70
##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
## 2         28.3         -63.9        -153                      
## 3         28.3         -63.9        -152                      
## 4         28.1         -63.9        -152                      
## 5         28.0         -63.9        -152                      
## 6         27.9         -63.9        -152                      
## 7         27.9         -63.9        -152                      
##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
## 2                                                                  
## 3                                                                  
## 4                                                                  
## 5                                                                  
## 6                                                                  
## 7                                                                  
##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
## 2                                                           NA
## 3                                                           NA
## 4                                                           NA
## 5                                                           NA
## 6                                                           NA
## 7                                                           NA
##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
## 2                NA                               NA                NA
## 3                NA                               NA                NA
## 4                NA                               NA                NA
## 5                NA                               NA                NA
## 6                NA                               NA                NA
## 7                NA                               NA                NA
##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
## 2                                     NA                      NA
## 3                                     NA                      NA
## 4                                     NA                      NA
## 5                                     NA                      NA
## 6                                     NA                      NA
## 7                                     NA                      NA
##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
## 2                                        36                NA
## 3                                        36                NA
## 4                                        36                NA
## 5                                        36                NA
## 6                                        36                NA
## 7                                        36                NA
##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 2               NA                  NA               NA                NA
## 3               NA                  NA               NA                NA
## 4               NA                  NA               NA                NA
## 5               NA                  NA               NA                NA
## 6               NA                  NA               NA                NA
## 7               NA                  NA               NA                NA
##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 2                   NA                NA              NA
## 3                   NA                NA              NA
## 4                   NA                NA              NA
## 5                   NA                NA              NA
## 6                   NA                NA              NA
## 7                   NA                NA              NA
##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 2                 NA              NA            0.02            0.00
## 3                 NA              NA            0.03           -0.02
## 4                 NA              NA            0.02           -0.02
## 5                 NA              NA            0.02            0.00
## 6                 NA              NA            0.02           -0.02
## 7                 NA              NA            0.02            0.00
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 2           -0.02             192             203            -216
## 3            0.00             196             204            -213
## 4            0.00             189             206            -214
## 5           -0.02             189             206            -214
## 6           -0.03             193             203            -215
## 7           -0.02             195             205            -215
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 2              -18              661              473      A
## 3              -18              658              469      A
## 4              -16              658              469      A
## 5              -17              655              473      A
## 6               -9              660              478      A
## 7              -18              659              470      A
```

## Appendix 2. Environment data


```r
sessionInfo()
```

```
## R version 3.5.1 (2018-07-02)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 10 x64 (build 17134)
## 
## Matrix products: default
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] parallel  stats     graphics  grDevices utils     datasets  methods  
## [8] base     
## 
## other attached packages:
##  [1] dplyr_0.7.8       kableExtra_1.1.0  knitr_1.20       
##  [4] doParallel_1.0.14 iterators_1.0.10  foreach_1.4.4    
##  [7] e1071_1.7-1       rattle_5.2.0      rpart.plot_3.0.7 
## [10] rpart_4.1-13      caret_6.0-84      ggplot2_3.1.0    
## [13] lattice_0.20-38  
## 
## loaded via a namespace (and not attached):
##  [1] Rcpp_0.12.19        lubridate_1.7.4     class_7.3-14       
##  [4] assertthat_0.2.0    rprojroot_1.3-2     digest_0.6.18      
##  [7] ipred_0.9-9         R6_2.3.0            plyr_1.8.4         
## [10] backports_1.1.2     stats4_3.5.1        evaluate_0.12      
## [13] highr_0.7           httr_1.3.1          pillar_1.3.0       
## [16] rlang_0.3.0.1       lazyeval_0.2.1      rstudioapi_0.8     
## [19] data.table_1.11.8   Matrix_1.2-15       rmarkdown_1.10     
## [22] splines_3.5.1       webshot_0.5.1       gower_0.2.1        
## [25] readr_1.3.1         stringr_1.3.1       munsell_0.5.0      
## [28] compiler_3.5.1      pkgconfig_2.0.2     gbm_2.1.5          
## [31] htmltools_0.3.6     nnet_7.3-12         tidyselect_0.2.5   
## [34] tibble_1.4.2        gridExtra_2.3       prodlim_2018.04.18 
## [37] codetools_0.2-15    randomForest_4.6-14 viridisLite_0.3.0  
## [40] crayon_1.3.4        withr_2.1.2         MASS_7.3-51.1      
## [43] recipes_0.1.5       ModelMetrics_1.2.2  grid_3.5.1         
## [46] nlme_3.1-137        gtable_0.2.0        magrittr_1.5       
## [49] scales_1.0.0        stringi_1.2.4       reshape2_1.4.3     
## [52] bindrcpp_0.2.2      timeDate_3043.102   xml2_1.2.0         
## [55] generics_0.0.2      lava_1.6.5          tools_3.5.1        
## [58] glue_1.3.0          purrr_0.2.5         hms_0.4.2          
## [61] survival_2.43-1     yaml_2.2.0          colorspace_1.3-2   
## [64] rvest_0.3.2         bindr_0.1.1
```


