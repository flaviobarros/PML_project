## Loading packages
library(readr)
library(caret)

############################### DATA PREPARATION ##########################################
## Reading training and testing 
treino <- read_csv(file = 'pml-training.csv')
teste <- read_csv(file = 'pml-testing.csv')

## Removing variables with too much NA
## Function to count NA per column
conta_na <- function(x) {sum(is.na(x))}

## Apply function to get NA fraction
sapply(treino, conta_na)

## Selecting variables with no more then 20% of NA
seletor <- sapply(treino, conta_na) < 0.2

## Selecting variables
treino <- subset(treino, select = seletor)
treino <- treino[,-1]
nzv <- nearZeroVar(treino, saveMetrics= TRUE)
treino <- treino[,!nzv$nzv]
treino$classe <- as.factor(treino$classe)

## Retrieving variables names to select the same from test set
colunas <- colnames(treino)
teste <- subset(teste, select = colnames(teste) %in% colunas)

################################# FIRST MODEL ############################################
library(rpart)

arvore <- rpart(classe ~ ., data = treino)
predito <- predict(arvore, teste, 'class')

## With just data cleaning and this simple model i got 17/20.

##########################################################################################
## Applying PCA and removing noisy variables like timestamp
treino$classe <- as.factor(treino$classe)
## Saving user name that can be a predictive power
user_name = treino$user_name

## Removing user name and timestamp variables
treino <- treino[,-c(1:4)]

## Finding correlations
correlated = findCorrelation(cor(treino[,-53]), cutoff = 0.85, verbose = T)

## Removing correlated variables
treino <- treino[,-correlated]

## Add user_name again
treino$user_name <- user_name

## Saving column names to remove from test set too
colunas <- colnames(treino)
teste <- subset(teste, select = colnames(teste) %in% colunas)

################################ Tree Model ######################################
## Evaluationg the tree with cross validation
## Fitting a multinomial logistic regression model with 90% PCA components
ctrl <- trainControl(method = 'cv', repeats = 1, 
                     verboseIter = T)

## Multinomial logistic regression model
rfFit <- train(classe ~., data = treino, method = 'rf', trainControl = ctrl)
predito <- predict(rfFit, teste)

################################ PCA #############################################
## PCA with princomp
pc <- princomp(treino[,-53], cor = T)
treino_pc <- pc$scores[1:47]

## Including class variable
treino_pc$classe <- treino$classe

## Fitting a multinomial logistic regression model with 90% PCA components
ctrl <- trainControl(method = 'cv', repeats = 1, 
                     verboseIter = T, classProbs = T)

## Multinomial logistic regression model
multnomlogistic <- train(classe ~., data = treino, method = 'multinom', trainControl = ctrl)
predito <- predict(multnomlogistic, teste)

######################## SAVING FILES TO SUBMIT ##########################################

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(x = predito)
