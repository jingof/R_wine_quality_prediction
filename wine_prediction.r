# PREDICTING WINE QUALITY
# project available at https://github.com/jingof/R_wine_quality_prediction.git
# We are going to try and predict the quality of wine basing on the ingredients that are used in the manufactury process. 
# We are going to use two types of wine that is red wine and white wine.

# The ingredients used in this notebook include: ['wine_type','fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
# 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'] 
# 'and these are the predictor features. The output feature is 'quality'.

# We start with loading the data sets containing the two wine types.

if(!require(tidyverse)) install.packages("tidyverse")

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# if(!require(psych)) install.packages('psych')
if(!require(glmnet)) install.packages('glmnet', dependencies=TRUE, type="binary")

library(dplyr)
library(tidyverse)
library(caret)
library(tidyr)
library(psych)
library(ggplot2)

options(warn=-1)

setwd('D:/predicting_wine_quality_R')

wine_data=read_csv("./datasets/winequality.csv")
head(wine_data)

# Next we look at the structure of the dataset
# Since we have two wine types, it is better to dig into both of them to understand the dataset better.
# We start with the information to know how many null values we have as well as the data types.
colSums(is.na(wine_data))



# looking at the shapes of the white wine and the red wine contained in the dataset
dim(wine_data)
red_wine <- wine_data %>% filter(wine_type=='Red') %>% select(-wine_type)
dim(red_wine)
white_wine <- wine_data %>% filter(wine_type=='White') %>% select(-wine_type)
dim(white_wine)

#plotting the distribution count of the wine dataset
#dev.off()
wine_data %>% ggplot(aes(wine_type)) + geom_histogram(binwidth = 0.1, stat='count', color=rainbow(2))

# looking at the quality summary for the datasets
summary(wine_data$quality)
summary(red_wine$quality)
summary(white_wine$quality)

# From the above, we can see that the white wine dataset has more rows meaning it has more observations.
# We also see that the data has no null values and has 12 predictor columns and one dependent column.
# Further, we can tell that all predictor columns are numerical except for wine type, to make this numerical, 
# we shall have to create two categorical features having 1 or 0, one for red wine, another for white wine.
# We can also see that no column is categorical except for the two we create below.

wine_data <- wine_data %>% mutate(red=ifelse(wine_type=='Red',1,0),white=ifelse(wine_type=='White',1,0)) %>% select(-wine_type)
y=wine_data$quality
wine_data <- wine_data %>% select(-quality) %>% mutate(quality=y)
# Next, we look at the dependent column to determine the nature of the values in the column. 
# From the below, we can tell that it is an integer rank ranging from 0 of 10.
unique(wine_data$quality)

# We now look at correlation, we could print out the correlation matrix of the columns 
# however i think using a heatmap might be prefferable since it is easier to spot very highly 
# correlated and very lowly correlated columns than on the matrix, lets have a look.

#dev.off()
corPlot(wine_data,scale=FALSE,xlas=2,upper=FALSE,main='')

# Calculate the average of each feature for the red and white wines separately using mean() function. 
# Plot bar graph to show comparison.
red_means <- red_wine %>% colMeans()
white_means <- white_wine %>% colMeans()
wine_data2 <- wine_data %>% select(-red,-white)


# This will help us understand the basic average between the quantities of ingredients used for the 
# two different wine datasets. From below we can see that most features have an almost similar mean 2
# for the two datasets except for free sulfurdioxide and total sulfur dioxide which have a significant difference 
# between the two means. We can also plot these to have a visualization.



wine_means <- structure(list(columns = c(colnames(wine_data2)), White = c(white_means),Red = c(red_means)), 
                 class = "data.frame", row.names = c('1','2','3','4','5','6','7','8','9','10','11', '12'))

#dev.off()
wine_means %>%
  pivot_longer(cols = -columns) %>%
  ggplot(aes(x = columns, y = value, fill = name)) + 
  geom_col(position = 'dodge')+
  theme(axis.text.x = element_text(angle = 70, hjust = 1))+
  ggtitle('Mean values of features for red and white wine')



# Graph Description
# White wine generally has a higher mean for most of the values meaning that it requires more ingredients and 
# therefore is more expensive to make. However, the cost does not necessarily lead to better quality since the 
# mean value for the quality is the same.

## Outliers 
# These are biased observations that may make our data analysis inaccurate. They are 3 standard deviations 
# away from the median. We can perform a box plot to check if there are any outliers.

boxplot(x = wine_data2, las=2)

# We can also have the outliers count is a data frame to correctly see how many outliers each column has. 
# From the below we can see that 8 columns have less than 100 rows with outliers.  


#method appends items from one list to another list
addElements <- function(list1,list2){
  len1 <- length(list1)
  i <- len1+1
  a<-1
  len2 <- length(list2)
  while(a <= len2){
    list1[i] <- list2[a]
    a <-a+1
    i<-i+1
  }
  list1
}


# method finds the indexes of the outliers in the column
FindOutliers <- function(dt,col){
    sd <- as.numeric(sapply(dt[col],sd))
    med <- as.numeric(sapply(dt[col],median))
    lowerq <- med - (sd*3)
    upperq <- med + (sd*3)
    which(dt[col]>upperq | dt[col]<lowerq)
}


cols <- c(colnames(wine_data2))
outlier_data <- data.frame(columns = cols)
outlier_count <- c()
outlier_indexes <- c()
i <-1
for(col in cols) {
  lis <- FindOutliers(wine_data2,col)
  outlier_indexes <- addElements(outlier_indexes, c(lis))
  outlier_count[i] <- length(lis)
  i <- i+1
}
outlier_indexes <- unique(outlier_indexes)
outlier_data <- outlier_data %>% mutate(outliers = outlier_count)
outlier_data

#looking at the outlier_index to determine how many unique values are there to determine 
# how many rows will actually fall under outliers.
outlier_len <- length(outlier_indexes)
outlier_len
percentage <- outlier_len*100/nrow(wine_data)
percentage
# We have 714 rows as outliers, this is 10.99% percent of the data, which is a huge number
# Instead of dropping these rows, we can have it as a categorical feature.
outlier_col <- rep(0,nrow(wine_data))
for(ind in outlier_indexes){
  outlier_col[ind] = 1
}
wine_data <- wine_data %>% mutate(is_outlier = outlier_col)

# splitting the dataset
set.seed(27)
n <- nrow(wine_data)
train_rows <- sample(1:n,.66*n)
train <- wine_data[train_rows,]

X_train <- train %>% select(-quality)
y_train <- train %>% select(quality)

test <- wine_data[-train_rows,]
X_test <- test %>% select(-quality)
y_test <- test %>% select(quality)
y_test <-as.numeric(unlist(y_test))


library(glmnet)
#library(MASS)
#for plot on left 
fit.lasso <- glmnet(as.matrix(X_train), as.matrix(y_train), family='gaussian', alpha=c(0:2,0.1))

# 10-fold Cross validation for each alpha = 0, 0.1, ... , 0.9, 1.0
fit.lasso.cv <- cv.glmnet(as.matrix(X_train), as.matrix(y_train), type.measure='mse', alpha=1, family='gaussian', standardize=TRUE)

#par(mfrow=c(1,1))
# For plotting options, type '?plot.glmnet' in R console
plot(fit.lasso, xvar="lambda")
plot(fit.lasso.cv)

# Feature Importance and selection
###*******************************************************************************************
###*
###*
###*
'%ni%'<-Negate('%in%')
glmnet1<-cv.glmnet(
      x=as.matrix(X_train),
      y=as.matrix(y_train),
      type.measure='mse',nfolds=20,alpha=1,
      standardize=TRUE)

c<-coef(glmnet1,s='lambda.1se',exact=TRUE)
inds<-which(c!=0)
variables<-row.names(c)[inds]
variables<-variables[variables %ni% '(Intercept)']
variables


lasso_imp <- c(as.matrix(c))[-1]
barplot(height=abs(lasso_imp),names=colnames(X_train), las=2, col=rainbow(15))

corr <- as.numeric(cor(train)['quality',])
corr<-corr[!corr==1]
barplot(height=abs(corr),names=colnames(X_train), las=2, col=rainbow(15))

importance <- data.frame(Features=colnames(X_train),lasso=abs(lasso_imp),correlation=abs(corr))
importance

# Model Choice

###******************************************************************************************
###*
###*
###*
train <- train %>% select(variables)%>% mutate(quality=unlist(y_train))
X_test <- X_test %>% select(variables)

test_accuracy <- function(pred,actual)
{
  val<-(actual - pred)
  len<-length(which(val==0))
  len/length(pred)
}

accuracys <- c()
control <- trainControl(method = "cv",  
                           number = 5,
                           repeats = 3,
                           classProbs = T)

# lasso
set.seed(27)
lasso <- train(quality ~ ., method = "lasso", data = train, trControl = control)
y_hat_lasso <- round(predict(lasso, X_test))
lasso_accuracy <- test_accuracy(y_hat_lasso, y_test)
accuracys[1]<-lasso_accuracy
lasso_accuracy


# random forest
rf <- train(quality ~ ., method = "rf", data = train, trControl = control)
y_hat_rf <- round(predict(rf, X_test))
rf_accuracy <- test_accuracy(y_hat_rf, y_test)
accuracys[2] <- rf_accuracy
rf_accuracy


# knn cv
knn <- train(quality ~ ., method='knn', data=train,tuneGrid=data.frame(k=seq(3, 51, 2)), trControl=control)
y_hat_knn <- round(predict(knn, X_test))
knn_accuracy <- test_accuracy(y_hat_knn, y_test)
accuracys[3] <- knn_accuracy
knn_accuracy



# decision tree
dt <- train(quality ~ ., method = 'rpart', data = train,tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)), trControl=control)
y_hat_dt <- round(predict(dt, X_test))
dt_accuracy <- test_accuracy(y_hat_dt, y_test)
accuracys[4] <- dt_accuracy
dt_accuracy

models <- c('lasso','random forest','knn cv','decision tree')

models_accuracy <- data.frame(Models=models,Accuracys=accuracys)
models_accuracy








