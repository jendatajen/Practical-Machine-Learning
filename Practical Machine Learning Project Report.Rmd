---
title: "Practical Machine Learning Course Project Report"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```
## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data Preprocessing  

```{r}
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(rattle)
library(randomForest)
library(RColorBrewer)
```
###Getting Data

```{r}
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile = trainFile, method = "curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile = testFile, method = "curl")
}
rm(trainUrl)
rm(testUrl)
```
###Reading Data
read the two csv files into two data frames.

```{r}
trainRaw <- read.csv(trainFile)
testRaw <- read.csv(testFile)
dim(trainRaw)
```
```{r}
dim(testRaw)
```
The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The `classe` variable in the training set is the outcome to predict. 

###Cleaning Data

```{r}
sum(complete.cases(trainRaw))
```
First, we remove columns that contain NA missing values.

```{r}
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0]
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0]
```
Next, we get rid of some columns that do not contribute much to the accelerometer measurements.

```{r}
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```

What we see is a lot of data with NA / empty values. Let's remove those
```{r}
maxNAPerc = 20
maxNACount <- nrow(trainRaw) / 100 * maxNAPerc
removeColumns <- which(colSums(is.na(trainRaw) | trainRaw=="") > maxNACount)
training.cleaned01 <- trainRaw[,-removeColumns]
testing.cleaned01 <- testRaw[,-removeColumns]
```
Also remove all time related data, since we won't use those
```{r}
removeColumns <- grep("timestamp", names(training.cleaned01))
training.cleaned02 <- training.cleaned01[,-c(1, removeColumns )]
testing.cleaned02 <- testing.cleaned01[,-c(1, removeColumns )]
```
Then convert all factors to integers
```{r}
classeLevels <- levels(training.cleaned02$classe)
training.cleaned03 <- data.frame(data.matrix(training.cleaned02))
training.cleaned03$classe <- factor(training.cleaned03$classe, labels=classeLevels)
testing.cleaned03 <- data.frame(data.matrix(testing.cleaned02))
```
Finally set the dataset to be explored
```{r}

training.cleaned <- training.cleaned03
testing.cleaned <- testing.cleaned03

```

Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

###Partitioning Training Set
```{r}
set.seed(22519) # For reproducibile purpose
classeIndex <- which(names(training.cleaned) == "classe")

partition <- createDataPartition(y=training.cleaned$classe, p=0.75, list=FALSE)
training.subSetTrain <- training.cleaned[partition, ]
validation <- training.cleaned[-partition, ]
```

###Correlation Matrix Visualization 

```{r}
corMatrix <- cor(training.subSetTrain[, -length(names(training.subSetTrain))])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

##Data Modelling
We fit a predictive model for activity recognition using **Random Forest** algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use **5-fold cross validation** when applying the algorithm.  

```{r}
treeModel <- rpart(classe ~ ., data=training.subSetTrain, method="class")
prp(treeModel) 
```

```{r}
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=training.subSetTrain, method="rf", trControl=controlRf, ntree=250)
modelRf
```
Then, we estimate the performance of the model on the validation data set.

```{r}
predictRF <- predict(modelRf, validation)
confusionMatrix(validation$classe, predictRF)
```

```{r}
accuracy <- postResample(predictRF, validation$classe)
ose <- 1 - as.numeric(confusionMatrix(validation$classe, predictRF)$overall[1])
accuracy
```

##Predicting for Test Data Set
Now, we apply the model to the original testing data set downloaded from the data source.
```{r}
rm(accuracy)
rm(ose)
predict(modelRf, testCleaned[, -length(names(testCleaned))])
```


