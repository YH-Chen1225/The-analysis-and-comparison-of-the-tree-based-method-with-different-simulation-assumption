install.packages("randomForest")
install.packages('adabag')
install.packages('forecast')
install.packages('fastAdaboost')
install.packages('tictoc')
library(rpart)
library(MASS)
library(randomForest)
library(adabag)
library(partykit)
library(mlr)
library(devtools)
library(forecast)
library(pROC)
library(cvms)
library(tidyverse)
library(fastAdaboost)
library(tictoc)
##Alpha graph for Ada boost explanation(corresponding to the figure 2,3,4 in term paper)
total_error <- seq(0,1,by=0.005)
alpha <- c()
alpha_cal <- function(total_error){
  alpha <- 1/2*log((1-total_error)/total_error)
}
alpha <- alpha_cal(total_error)
plot(total_error,alpha)

new_weight_01 <- exp(alpha)
plot(alpha,new_weight_01)
new_weight_02 <- exp(-alpha)
plot(alpha,new_weight_02)


#Mutivariate dataset simulation_01
# set seed and create data vectors
set.seed(666)
sample_size <- 600                                       
sample_meanvector <- c(2, 2)                                   
sample_covariance_matrix <- matrix(c(2, 0, 0, 2),
                                   ncol = 2)

# create bivariate normal distribution
sample_distribution <- mvrnorm(n = sample_size,
                               mu = sample_meanvector, 
                               Sigma = sample_covariance_matrix)

head(sample_distribution)

positive <- data.frame(sample_distribution)
positive['y'] <- rep(1,600)

sample_size <- 600                                       
sample_meanvector <- c(8, 8)                                   
sample_covariance_matrix <- matrix(c(2, 0, 0, 2),
                                   ncol = 2)
sample_minus <- mvrnorm(n = sample_size,
                               mu = sample_meanvector, 
                               Sigma = sample_covariance_matrix)
head(sample_minus)
negative <- data.frame(sample_minus)
negative['y'] <- rep(-1,600)
data <- rbind(positive,negative)


#data['y'] <- as.factor(data['y'])

###############Plot the data(corresponding to the figure 5 in term paper)
library(ggplot2)
ggplot(data = data, aes(x=X1, y=X2))+
  geom_point(aes(shape = factor(y),color = factor(y)))+
  labs(shape = 'y',color = 'y')+
  ggtitle("Distribution in case one")

########Data generation process##The sample size can be change here according to my needs(Now I need 25000 obs)
dgp_01 <- function(mean1,mean2){
  sample_size <- 25000                                       
  sample_meanvector <- c(mean1, mean1)                                   
  sample_covariance_matrix <- matrix(c(2, 0, 0, 2),
                                     ncol = 2)
  sample_distribution <- mvrnorm(n = sample_size,
                                 mu = sample_meanvector, 
                                 Sigma = sample_covariance_matrix)
  positive <- data.frame(sample_distribution)
  positive['y'] <- rep(1,25000)
  sample_meanvector <- c(mean2, mean2)#just need to change the mean, all the other are the same
  sample_minus <- mvrnorm(n = sample_size,
                          mu = sample_meanvector, 
                          Sigma = sample_covariance_matrix)
  negative <- data.frame(sample_minus)
  negative['y'] <- rep(-1,25000)
  data <- rbind(positive,negative)
}
##############################################################################





#Take the sample just once and makes prediction
data_roc <- dgp_01(2,8)
sample <- sample.int(n = nrow(data_roc),size = floor(0.7*nrow(data_roc)),replace = F)
train <- data_roc[sample,]
test <- data_roc[-sample,]
#See how does these algorithms work in first case
#decision tree
rpart_train <- rpart(y ~ .,data = train, control = c(cp = 0, minsplit = 0), method ="class")
rpart_predict <- predict(rpart_train,test,type = "prob")
rpart_predict_cl <- predict(rpart_train,test, type = 'class')
prob <- rpart_predict[,2]

#random forest_50
rf_train_50 <- randomForest(factor(y) ~ . , data = train, 
                            ntree = 50)# take all the predictors as sampled
rf_predict_50 <- predict(rf_train_50,test,type = 'prob')
rf_predict_50_cl <- predict(rf_train_50,test)
prob_rf <- rf_predict_50[,2]
#random forest_500
rf_train_500 <- randomForest(factor(y) ~ . , data = train, 
                            ntree = 500)# take all the predictors as sampled
rf_predict_500 <- predict(rf_train_500,test,type = 'prob')
rf_predict_500_cl <- predict(rf_train_500,test)
prob_rf_500 <- rf_predict_500[,2]

#adaboost
train$y <- as.factor(train$y)#Need to change y as factor in advance
ada_train <- boosting(y ~ ., train, boos=TRUE, mfinal=100)#Create 100 tree and using the weight to drawn
ada_predict <- predict(ada_train,test)
ada_predict_cl <- ada_predict$class 
prob_ada <- ada_predict$prob[,2]
#prune_tree
model_pruned <- prune(rpart_train,cp=rpart_train$cptable[which.min(rpart_train$cptable[,"xerror"]),"CP"])
pruned_predict <- predict(model_pruned,test,type = 'prob')
pruned_predict_cl <- predict(model_pruned,test,type = 'class')
prob_pruned <- pruned_predict[,2]


roc.res <- roc(test$y,prob,levels = c(-1,1),direction = "<",identity.lty = 1,identity=TRUE)
plot.roc(roc.res,print.auc =TRUE)

dev.off()
par(pty = "s")

#dev.set(dev.next())
#par(mar=c(2.5,2.5,2,1),xpd=TRUE)
#par(mar = c(4, 4, 4, 4)+.1,xpd =FALSE)
#png("test.png", width = 480, height = 480)


####Plot the ROC graph here(Corresponding to the figure 8 in term paper)
plot(roc(test$y,prob,levels = c(-1,1),direction = "<",identity.lty = 3),print.auc = TRUE, col = "blue",identity.lty = 2,
    grid = TRUE,lwd = 4,xlim = c(1,0),xlab = "Sensitivity")
plot(roc(test$y, prob_rf,levels = c(-1,1),direction = "<"), print.auc = TRUE,
                 col = "green", print.auc.y = .4, add = TRUE,lty=2,identity.lty = 2,lwd = 2)
plot(roc(test$y, prob_ada,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "red", print.auc.y = .3, add = TRUE,lty=3,identity.lty = 2,lwd = 2)
plot(roc(test$y, prob_pruned,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "black", print.auc.y = .2, add = TRUE,lty=4,identity.lty = 2,lwd = 4)
plot(roc(test$y, prob_rf_500,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "brown", print.auc.y = .1, add = TRUE,lty=5,identity.lty = 2,lwd = 2)
legend("topright", c("DT", "RF50","Ada","Pruned",'RF500'), lty = c(1,2,3,4,5), 
       col = c("blue", "green",'red','black','brown'), bty="n", lwd = c(4,2,2,4,2),inset=c(0,0.05),y.intersp = 0.5)
title('ROC Curve', line = 2.5)

##Plot the confusion matrix here(Corresponding to the figure 9,10,11,12,13 in term paper)
confusion_matrix_train_tree <- table(actual = test$y, predicted = rpart_predict_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_rf50 <- table(actual = test$y, predicted = rf_predict_50_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_rf500 <- table(actual = test$y, predicted = rf_predict_500_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_prune <- table(actual = test$y, predicted = pruned_predict_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_ada <- table(actual = test$y, predicted = ada_predict_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_tree), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of DT ")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_prune), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of Pruned")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_rf50), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of rf50")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_rf500), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of rf500")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_ada), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of ada")
)





#write.csv(train,"C:\\Users\\ua896\\OneDrive\\桌面\\Comp_Stats\\Term paper\\train_data.csv", row.names = FALSE)
#write.csv(test,"C:\\Users\\ua896\\OneDrive\\桌面\\Comp_Stats\\Term paper\\test_data.csv", row.names = FALSE)




###### 1000 times simulation for case one with 1200 obs
T_iter <- 1000
mis_class_tree <- c()
mis_class_prune <- c()
mis_class_ada <- c()
mis_class_rf_50 <- c()
mis_class_rf_500 <- c()
for (t in 1:T_iter){
  data <- dgp_01(2,8)
  sample <- sample.int(n = nrow(data), size = floor(.7*nrow(data)), replace = F)
  train <- data[sample, ]
  test  <- data[-sample, ]
  #Decision Tree
  rpart_train <- rpart(y ~ .,data = train, control = c(cp = 0, minsplit = 0))
  rpart_predict <- predict(rpart_train,test)
  #Random Forest_50
  rf_train_50 <- randomForest(factor(y) ~ . , data = train, 
                              mtry = 2 ,ntree = 50)# take all the predictors as sampled
  rf_predict_50 <- predict(rf_train_50,test)
  #Random Forest_500
  rf_train_500 <- randomForest(factor(y) ~ . , data = train, 
                              mtry = 2 ,ntree = 500)# take all the predictors as sampled
  rf_predict_500 <- predict(rf_train_500,test)
  #Adaboost
  train$y <- as.factor(train$y)#Need to change y as factor in advance
  ada_train <- boosting(y ~ ., train, boos=TRUE, mfinal=100)#Create 100 tree and using the weight to drawn
  ada_predict <- predict(ada_train,test)
  ada_predict$class
  #Pruned Tree
  d.tree.param <- makeClassifTask(
    data = train, 
    target="y")
  
  param_grid_multi <- makeParamSet( 
    makeNumericParam("cp", lower = 0.001, upper = 0.01))
  
  control_grid = makeTuneControlGrid()
  resample = makeResampleDesc("CV", iters = 2L)
  measure = acc
  
  dt_tuneparam <- tuneParams(learner="classif.rpart", 
                             task=d.tree.param, 
                             resampling = resample,
                             measures = measure,
                             par.set=param_grid_multi, 
                             control=control_grid, 
                             show.info = TRUE)
  best_parameters = setHyperPars(
    makeLearner("classif.rpart"), 
    par.vals = dt_tuneparam$x
  )
  
  
  best_model = train(best_parameters, d.tree.param)
  test$y <- as.factor(test$y)
  d.tree.mlr.test <- makeClassifTask(
    data=test, 
    target="y")
  results <- predict(best_model, task = d.tree.mlr.test)$data
  accuracy_prune <- mean(results$response == results$truth)
  
  accuracy_tree <- c()
  accuracy_rf_50 <- c()
  accuracy_rf_500<-c()
  accuracy_ada<-c()
  
  
  for (i in 1:360){
    if (rpart_predict[i] == test[i,'y']){
      accuracy_tree[i] <- 1}
    else{accuracy_tree[i] <- 0
    }
    if (rf_predict_50[i] == test[i,"y"]){
      accuracy_rf_50[i] <- 1}
    else{accuracy_rf_50[i] <- 0}
    if (ada_predict$class[i] == test[i,"y"]){
      accuracy_ada[i] <- 1}
    else{accuracy_ada[i] <- 0}
    if (rf_predict_500[i] == test[i,"y"]){
      accuracy_rf_500[i] <- 1}
    else{accuracy_rf_500[i]<-0}
    }
  
  accuracy_rate_tree <- mean(accuracy_tree)
  mis_class_tree[t] <- 1-accuracy_rate_tree
  accuracy_rate_rf <- mean(accuracy_rf_50)
  mis_class_rf_50[t] <- 1-accuracy_rate_rf
  accuracy_rate_ada <- mean(accuracy_ada)
  mis_class_ada[t] <- 1-accuracy_rate_ada
  accuracy_rate_rf_500 <- mean(accuracy_rf_500)
  mis_class_rf_500[t] <- 1-accuracy_rate_rf_500
  mis_class_prune[t] <- 1-accuracy_prune
}

mean_mis_class_tree <-mean(mis_class_tree)
mean_mis_class_rf_50 <- mean(mis_class_rf_50)



mis_data <- cbind(mis_class_tree,mis_class_rf_50,mis_class_rf_500,mis_class_prune,mis_class_ada)
mis_data <- as.data.frame(mis_data)


mis_data$mis_class_tree <- as.numeric(mis_data$mis_class_tree)
x <- seq(1,1000,by = 1)

#ggplot(mis_data, aes(x))+
#geom_line(aes(y = 'mis_class_tree'), color = 'green')
#par(mar=c(2.5,2.5,2,1),xpd=TRUE)  


#Plot the simulation outcome here
#dev.off()
#plot(x,mis_class_tree,type = 'o',col = 'blue',xlab = 'Experiment Times',
     #ylab = 'mis classification rate',ylim = c(0,0.06))
#lines(mis_class_prune, type ="o", col = "black")
#lines(mis_class_rf_50, type = "o",col = 'green')
#lines(mis_class_rf_500, type = "o",col = 'brown')
#lines(mis_class_ada, type = "o", col = 'red')
#legend("topright", c("DT", "Pruned","RF50","RF500",'Ada'), lty = c(1,1,1,1,1),y.intersp = 0.3,
       #col = c("blue", "black",'green','brown','red'), bty="n",inset=c(0,0.0001))
#title("Misclassification Rate Comparison")

#Box plot (Corresponding to figure 14 in the term paper)
boxplot(mis_class_tree, mis_class_rf_50, mis_class_rf_500,mis_class_ada,
        mis_class_prune,names = c('DT','RF50','RF500','Ada','pruned'),
        col=c('green','blue','red','brown','yellow'),ylim = c(0,0.05))
title('simulation result')

##########################Calculation the mean and s.d for creating the table(Corresponding to table 9 in the term paper)
mean(mis_class_tree)
mean(mis_class_rf_50)
mean(mis_class_rf_500)
mean(mis_class_ada)
mean(mis_class_prune)
sd(mis_class_tree)
sd(mis_class_rf_50)
sd(mis_class_rf_500)
sd(mis_class_ada)
sd(mis_class_prune)

###########################################  
###########################################  
##############Experiment_2#################
###########################################

set.seed(666)
sample_size <- 600                                       
sample_meanvector <- c(2, 2)                                   
sample_covariance_matrix <- matrix(c(2, 0, 0, 2),
                                   ncol = 2)

# create bivariate normal distribution
positive_distribution <- mvrnorm(n = sample_size,
                               mu = sample_meanvector, 
                               Sigma = sample_covariance_matrix)


positive <- data.frame(positive_distribution)
positive['y'] <- rep(1,600)

sample_size <- 600                                       
sample_meanvector <- c(3, 3)                                   
sample_covariance_matrix <- matrix(c(2, 0, 0, 2),
                                   ncol = 2)
sample_minus <- mvrnorm(n = sample_size,
                        mu = sample_meanvector, 
                        Sigma = sample_covariance_matrix)

negative <- data.frame(sample_minus)
negative['y'] <- rep(-1,600)
data_2 <- rbind(positive,negative)


#plot the distribution of data again for case 2(corresponding to figure 6 in the term paper)
library(ggplot2)
ggplot(data = data_2, aes(x=X1, y=X2))+
  geom_point(aes(shape = factor(y),color = factor(y)))+
  labs(shape = 'y',color = 'y')+
  ggtitle("Distribution of case two")

#Make prediciton once with 50000 obs
set.seed(111)
data_roc <- dgp_01(2,3)
sample <- sample.int(n = nrow(data_roc),size = floor(0.7*nrow(data_roc)),replace = F)
train <- data_roc[sample,]
test <- data_roc[-sample,]
#See how does these algorithms work in first case"type = "prob"
#decision tree
rpart_train <- rpart(y ~ .,data = train, control = c(cp = 0, minsplit = 0),method ="class")
rpart_predict <- predict(rpart_train,test,type = 'prob')
rpart_predict_cl <- predict(rpart_train,test,type = 'class')
length(rpart_predict)
prob <- rpart_predict[,2]

#random forest_50
rf_train_50 <- randomForest(factor(y) ~ . , data = train, mtry = 2,
                            ntree = 50)# take all the predictors as sampled
rf_predict_50 <- predict(rf_train_50,test,type = 'prob')
rf_predict_50_cl <- predict(rf_train_50,test)

prob_rf <- rf_predict_50[,2]
#random forest_500
tic("sleeping")
rf_train_500 <- randomForest(factor(y) ~ . , data = train, mtry = 2,
                            ntree = 500)# take all the predictors as sampled
rf_predict_500 <- predict(rf_train_500,test,type = 'prob')
rf_predict_500_cl <- predict(rf_train_500, test)
prob_rf_500 <- rf_predict_500[,2]
toc()
#adaboost
tic("sleeping")
train$y <- as.factor(train$y)#Need to change y as factor in advance
ada_train <- boosting(y ~ ., train, boos=TRUE, mfinal=100)#Create 100 tree and using the weight to drawn
ada_predict <- predict(ada_train,test)
prob_ada <- ada_predict$prob[,2]
prob_ada_cl <- ada_predict$class
toc()
#prune_tree
model_pruned <- prune(rpart_train,cp=rpart_train$cptable[which.min(rpart_train$cptable[,"xerror"]),"CP"])
pruned_predict <- predict(model_pruned,test,type = 'prob')
pruned_predict_cl <- predict(model_pruned,test,type = 'class')
prob_pruned <- pruned_predict[,2]

roc.res <- roc(test$y,prob,levels = c(-1,1),direction = "<",identity.lty = 1,identity=TRUE)
plot.roc(roc.res,print.auc =TRUE)

######Plot the ROC graph(corresponding to the figure 15 in the term paper)
dev.off()
par(pty = "s")
#dev.set(dev.next())
#par(mar=c(2.5,2.5,2,1),xpd=TRUE)
#par(mar = c(4, 4, 4, 4)+.1,xpd =FALSE)
#png("test.png", width = 480, height = 480)
plot(roc(test$y,prob,levels = c(-1,1),direction = "<",identity.lty = 3),print.auc = TRUE, col = "blue",identity.lty = 2,
     grid = TRUE,lwd = 2,xlim = c(1,0))
plot(roc(test$y, prob_rf,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "green", print.auc.y = .4, add = TRUE,lty=2,identity.lty = 2,lwd = 2)
plot(roc(test$y, prob_ada,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "red", print.auc.y = .3, add = TRUE,lty=3,identity.lty = 2,lwd = 2)
plot(roc(test$y, prob_pruned,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "black", print.auc.y = .2, add = TRUE,lty=4,identity.lty = 2,lwd = 2)
plot(roc(test$y, prob_rf_500,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "brown", print.auc.y = .1, add = TRUE,lty=5,identity.lty = 2,lwd = 2)
legend("bottomleft", c("DT", "RF50","Ada","Pruned",'RF500'), lty = c(1,2,3,4,5), 
       col = c("blue", "green",'red','black','brown'), bty="n", lwd = c(2,2,2,2,2),inset=c(0.55,0.03),y.intersp = 0.5,x.intersp = 0.05)
title('ROC Curve', line = 2.5)

#Plot confusion matrix for case two(corresponding to the figure 16,17,18,19,20)
confusion_matrix_train_tree <- table(actual = test$y, predicted = rpart_predict_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_rf50 <- table(actual = test$y, predicted = rf_predict_50_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_rf500 <- table(actual = test$y, predicted = rf_predict_500_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_prune <- table(actual = test$y, predicted = pruned_predict_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_ada <- table(actual = test$y, predicted = prob_ada_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_tree), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of DT ")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_prune), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of Pruned")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_rf50), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of rf50")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_rf500), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of rf500")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_ada), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of ada")
)






#####simulation study
set.seed(666)
T_iter <- 1000
mis_class_tree_02 <- c()
mis_class_prune_02 <- c()
mis_class_ada_02 <- c()
mis_class_rf_50_02 <- c()
mis_class_rf_500_02 <- c()
for (t in 1:T_iter){
  data <- dgp_01(2,3)##Same dgp is used in dgp1 but change parameter
  sample <- sample.int(n = nrow(data), size = floor(.7*nrow(data)), replace = F)
  train <- data[sample, ]
  test  <- data[-sample, ]
  #Decision Tree
  rpart_train <- rpart(y ~ .,data = train, control = c(cp = 0, minsplit = 0))
  rpart_predict <- predict(rpart_train,test)
  #Random Forest_50
  rf_train_50 <- randomForest(factor(y) ~ . , data = train, 
                              mtry = 2 ,ntree = 50)# take all the predictors as sampled
  rf_predict_50 <- predict(rf_train_50,test)
  #Random Forest_500
  rf_train_500 <- randomForest(factor(y) ~ . , data = train, 
                               mtry = 2 ,ntree = 500)# take all the predictors as sampled
  rf_predict_500 <- predict(rf_train_500,test)
  #Adaboost
  train$y <- as.factor(train$y)#Need to change y as factor in advance
  ada_train <- boosting(y ~ ., train, boos=TRUE, mfinal=100)#Create 100 tree and using the weight to drawn
  ada_predict <- predict(ada_train,test)
  ada_predict$class
  #Pruned Tree
  d.tree.param <- makeClassifTask(
    data = train, 
    target="y")
  
  param_grid_multi <- makeParamSet( 
    makeNumericParam("cp", lower = 0.001, upper = 0.01))
  
  control_grid = makeTuneControlGrid()
  resample = makeResampleDesc("CV", iters = 2L)
  measure = acc
  
  dt_tuneparam <- tuneParams(learner="classif.rpart", 
                             task=d.tree.param, 
                             resampling = resample,
                             measures = measure,
                             par.set=param_grid_multi, 
                             control=control_grid, 
                             show.info = TRUE)
  best_parameters = setHyperPars(
    makeLearner("classif.rpart"), 
    par.vals = dt_tuneparam$x
  )
  
  
  best_model = train(best_parameters, d.tree.param)
  test$y <- as.factor(test$y)
  d.tree.mlr.test <- makeClassifTask(
    data=test, 
    target="y")
  results <- predict(best_model, task = d.tree.mlr.test)$data
  accuracy_prune <- mean(results$response == results$truth)
  
  accuracy_tree <- c()
  accuracy_rf_50 <- c()
  accuracy_rf_500<-c()
  accuracy_ada<-c()
  
  
  for (i in 1:360){
    if (rpart_predict[i] == test[i,'y']){
      accuracy_tree[i] <- 1}
    else{accuracy_tree[i] <- 0
    }
    if (rf_predict_50[i] == test[i,"y"]){
      accuracy_rf_50[i] <- 1}
    else{accuracy_rf_50[i] <- 0}
    if (ada_predict$class[i] == test[i,"y"]){
      accuracy_ada[i] <- 1}
    else{accuracy_ada[i] <- 0}
    if (rf_predict_500[i] == test[i,"y"]){
      accuracy_rf_500[i] <- 1}
    else{accuracy_rf_500[i]<-0}
  }
  
  accuracy_rate_tree <- mean(accuracy_tree)
  mis_class_tree_02[t] <- 1-accuracy_rate_tree
  accuracy_rate_rf <- mean(accuracy_rf_50)
  mis_class_rf_50_02[t] <- 1-accuracy_rate_rf
  accuracy_rate_ada <- mean(accuracy_ada)
  mis_class_ada_02[t] <- 1-accuracy_rate_ada
  accuracy_rate_rf_500 <- mean(accuracy_rf_500)
  mis_class_rf_500_02[t] <- 1-accuracy_rate_rf_500
  mis_class_prune_02[t] <- 1-accuracy_prune
}

#####Plot the 1000 times simulation outcome
#dev.off()
#x = seq(1,1000,by = 1)
#plot(x,mis_class_tree_02,type = 'o',col = 'blue',xlab = 'Experiment Times',
    # ylab = 'mis classification rate',ylim = c(0,0.85))
#lines(mis_class_prune_02, type ="o", col = "black")
#lines(mis_class_rf_50_02, type = "o",col = 'green')
#lines(mis_class_rf_500_02, type = "o",col = 'brown')
#lines(mis_class_ada_02, type = "o", col = 'red')
#legend("bottomright", c("DT", "Pruned","RF50","RF500",'Ada'), lty = c(1,1,1,1,1),y.intersp = 0.3,
       #col = c("blue", "black",'green','brown','red'), bty="n",inset=c(0,0.0001))
#title("Misclassification Rate Comparison")

#Correspond to table 10 in the term paper
mean(mis_class_tree_02)
mean(mis_class_rf_50_02)
mean(mis_class_rf_500_02)
mean(mis_class_ada_02)
mean(mis_class_prune_02)

#Correspond to figure 21 in the term paper
boxplot(mis_class_tree_02, mis_class_rf_50_02, mis_class_rf_500_02,mis_class_ada_02,
        mis_class_prune_02,names = c('DT','RF50','RF500','Ada','pruned'),
        col=c('green','blue','red','brown','yellow'))
title('simulation result')
sd(mis_class_tree_02)
sd(mis_class_rf_50_02)
sd(mis_class_rf_500_02)
sd(mis_class_ada_02)
sd(mis_class_prune_02)




###########################################  
###########################################  
##############Experiment_3#################
###########################################

set.seed(111)
sample_size <- 400                                       
sample_meanvector <- c(1, 1)                                   
sample_covariance_matrix <- matrix(c(1, 0.7, 0.7, 1),
                                   ncol = 2)

# create bivariate normal distribution
positive_distribution <- mvrnorm(n = sample_size,
                                 mu = sample_meanvector, 
                                 Sigma = sample_covariance_matrix)


positive <- data.frame(positive_distribution)
positive['y'] <- rep(1,400)

sample_size <- 400                                       
sample_meanvector <- c(1, 3)                                   
sample_covariance_matrix <- matrix(c(1, 0.7, 0.7, 1),
                                   ncol = 2)
sample_minus_01 <- mvrnorm(n = sample_size,
                        mu = sample_meanvector, 
                        Sigma = sample_covariance_matrix)

negative <- data.frame(sample_minus_01)
negative['y'] <- rep(-1,400)

sample_size <- 400                                       
sample_meanvector <- c(3, 1)                                   
sample_covariance_matrix <- matrix(c(1, 0.7, 0.7, 1),
                                   ncol = 2)
sample_minus_02 <- mvrnorm(n = sample_size,
                           mu = sample_meanvector, 
                           Sigma = sample_covariance_matrix)

negative_02 <- data.frame(sample_minus_02)
negative_02['y'] <- rep(-1,400)

data_3 <- rbind(positive,negative,negative_02)


#################################Plot the dataset#corresponding to figure 7 in the term paper
library(ggplot2)
ggplot(data = data_3, aes(x=X1, y=X2))+
  geom_point(aes(shape = factor(y),color = factor(y)))+
  labs(shape = 'y',color = 'y')+
  ggtitle('Distribution in case three')

##################################data generating process
dgp_03 <- function(N){
  sample_size <- N                                       
  sample_meanvector <- c(1, 1)                                   
  sample_covariance_matrix <- matrix(c(1, 0.7, 0.7, 1),
                                     ncol = 2)
  positive_distribution <- mvrnorm(n = sample_size,
                                   mu = sample_meanvector, 
                                   Sigma = sample_covariance_matrix)
  positive <- data.frame(positive_distribution)
  positive['y'] <- rep(1,N)
  sample_meanvector <- c(1, 3)
  sample_minus_01 <- mvrnorm(n = sample_size,
                             mu = sample_meanvector, 
                             Sigma = sample_covariance_matrix)
  negative <- data.frame(sample_minus_01)
  negative['y'] <- rep(-1,N)
  sample_meanvector <- c(3, 1)
  sample_minus_02 <- mvrnorm(n = sample_size,
                             mu = sample_meanvector, 
                             Sigma = sample_covariance_matrix)
  negative_02 <- data.frame(sample_minus_02)
  negative_02['y'] <- rep(-1,N)
  data <- rbind(positive,negative,negative_02)
}

#Making prediction with 49998 obs
#enableJIT(3)
library(fastAdaboost)
set.seed(3)
data_roc <- dgp_03(16666)
sample <- sample.int(n = nrow(data_roc),size = floor(0.7*nrow(data_roc)),replace = F)
train <- data_roc[sample,]
test <- data_roc[-sample,]
#See how does these algorithms work in first case
#decision tree
rpart_train <- rpart(y ~ .,data = train, control = c(cp = 0, minsplit = 0), method ="class")
rpart_predict <- predict(rpart_train,test,type = "prob")
rpart_predict_cl <- predict(rpart_train,test, type = "class")
prob <- rpart_predict[,2]

#random forest_50
rf_train_50 <- randomForest(factor(y) ~ . , data = train, 
                            mtry = 2 ,ntree = 50)# take all the predictors as sampled
rf_predict_50 <- predict(rf_train_50,test,type = 'prob')
rf_predict_50_cl <- predict(rf_train_50, test)
prob_rf <- rf_predict_50[,2]
#random forest_500
rf_train_500 <- randomForest(factor(y) ~ . , data = train, 
                             mtry = 2 ,ntree = 500)# take all the predictors as sampled
rf_predict_500 <- predict(rf_train_500,test,type = 'prob')
rf_predict_500_cl <- predict(rf_train_500, test)
prob_rf_500 <- rf_predict_500[,2]

#adaboost 
train$y <- as.factor(train$y)#Need to change y as factor in advance
ada_train <- adaboost(y ~ ., train, nIter = 100)#Create 100 tree and using the weight to drawn
ada_predict <- predict(ada_train,test)
prob_ada <- ada_predict$prob[,2]
prob_ada_cl <- ada_predict$class
#prune_tree
model_pruned <- prune(rpart_train,cp=rpart_train$cptable[which.min(rpart_train$cptable[,"xerror"]),"CP"])
pruned_predict <- predict(model_pruned,test,type = 'prob')
pruned_predict_cl <- predict(model_pruned,test,type = "class")
prob_pruned <- pruned_predict[,2]

roc.res <- roc(test$y,prob,levels = c(-1,1),direction = "<",identity.lty = 1,identity=TRUE)
plot.roc(roc.res,print.auc =TRUE)


#plot the ROC for case3, correspond to figure22 in the term paper
dev.off()
#dev.set(dev.next())
par(pty = "s")
#par(mar=c(2.5,2.5,2,1),xpd=TRUE)
#par(mar = c(4, 4, 4, 4)+.1,xpd =FALSE)
#png("test.png", width = 480, height = 480)
plot(roc(test$y,prob,levels = c(-1,1),direction = "<",identity.lty = 3),print.auc = TRUE, col = "blue",identity.lty = 2,
     grid = TRUE,lwd = 2,xlim = c(1,0))
plot(roc(test$y, prob_rf,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "green", print.auc.y = .4, add = TRUE,lty=2,identity.lty = 2,lwd = 2)
plot(roc(test$y, prob_ada,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "red", print.auc.y = .3, add = TRUE,lty=3,identity.lty = 2,lwd = 2)
plot(roc(test$y, prob_pruned,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "black", print.auc.y = .2, add = TRUE,lty=4,identity.lty = 2,lwd = 2)
plot(roc(test$y, prob_rf_500,levels = c(-1,1),direction = "<"), print.auc = TRUE,
     col = "brown", print.auc.y = .1, add = TRUE,lty=5,identity.lty = 2,lwd = 2)
legend("bottomleft", c("DT", "RF50","Ada","Pruned",'RF500'), lty = c(1,2,3,4,5), 
       col = c("blue", "green",'red','black','brown'), bty="n", lwd = c(2,2,2,2,2),inset=c(0.6,0.05),y.intersp = 0.5,x.intersp = 0.05)
title('ROC Curve', line = 2.5)

#Confusion matrix for case 3 once simulation(Correspond to figure 23,24,25,26,27 in the term paper)
confusion_matrix_train_tree <- table(actual = test$y, predicted = rpart_predict_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_rf50 <- table(actual = test$y, predicted = rf_predict_50_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_rf500 <- table(actual = test$y, predicted = rf_predict_500_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_prune <- table(actual = test$y, predicted = pruned_predict_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 
confusion_matrix_train_ada <- table(actual = test$y, predicted = prob_ada_cl)
options(repr.plot.res = 250, repr.plot.height = 5, repr.plot.width = 5) 

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_tree), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of DT ")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_prune), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of Pruned")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_rf50), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of rf50")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_rf500), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of rf500")
)

suppressWarnings(
  plot_confusion_matrix(as_tibble(confusion_matrix_train_ada), 
                        target_col = "actual", 
                        prediction_col = "predicted",
                        counts_col = "n") +
    ggtitle("Confusion matrix of ada")
)
###############################################simulation study for case3
set.seed(666)
T_iter <- 1000
mis_class_tree_03 <- c()
mis_class_prune_03 <- c()
mis_class_ada_03 <- c()
mis_class_rf_50_03 <- c()
mis_class_rf_500_03 <- c()
for (t in 1:T_iter){
  data <- dgp_03(400)
  sample <- sample.int(n = nrow(data), size = floor(.7*nrow(data)), replace = F)
  train <- data[sample, ]
  test  <- data[-sample, ]
  #Decision Tree
  rpart_train <- rpart(y ~ .,data = train, control = c(cp = 0, minsplit = 0))
  rpart_predict <- predict(rpart_train,test)
  #Random Forest_50
  rf_train_50 <- randomForest(factor(y) ~ . , data = train, 
                              mtry = 2 ,ntree = 50)# take all the predictors as sampled
  rf_predict_50 <- predict(rf_train_50,test)
  #Random Forest_500
  rf_train_500 <- randomForest(factor(y) ~ . , data = train, 
                               mtry = 2 ,ntree = 500)# take all the predictors as sampled
  rf_predict_500 <- predict(rf_train_500,test)
  #Adaboost
  train$y <- as.factor(train$y)#Need to change y as factor in advance
  ada_train <- boosting(y ~ ., train, boos=TRUE, mfinal=100)#Create 100 tree and using the weight to drawn
  ada_predict <- predict(ada_train,test)
  ada_predict$class
  #Pruned Tree
  d.tree.param <- makeClassifTask(
    data = train, 
    target="y")
  
  param_grid_multi <- makeParamSet( 
    makeNumericParam("cp", lower = 0.001, upper = 0.01))
  
  control_grid = makeTuneControlGrid()
  resample = makeResampleDesc("CV", iters = 2L)
  measure = acc
  
  dt_tuneparam <- tuneParams(learner="classif.rpart", 
                             task=d.tree.param, 
                             resampling = resample,
                             measures = measure,
                             par.set=param_grid_multi, 
                             control=control_grid, 
                             show.info = TRUE)
  best_parameters = setHyperPars(
    makeLearner("classif.rpart"), 
    par.vals = dt_tuneparam$x
  )
  
  
  best_model = train(best_parameters, d.tree.param)
  test$y <- as.factor(test$y)
  d.tree.mlr.test <- makeClassifTask(
    data=test, 
    target="y")
  results <- predict(best_model, task = d.tree.mlr.test)$data
  accuracy_prune <- mean(results$response == results$truth)
  
  accuracy_tree <- c()
  accuracy_rf_50 <- c()
  accuracy_rf_500<-c()
  accuracy_ada<-c()
  
  
  for (i in 1:360){
    if (rpart_predict[i] == test[i,'y']){
      accuracy_tree[i] <- 1}
    else{accuracy_tree[i] <- 0
    }
    if (rf_predict_50[i] == test[i,"y"]){
      accuracy_rf_50[i] <- 1}
    else{accuracy_rf_50[i] <- 0}
    if (ada_predict$class[i] == test[i,"y"]){
      accuracy_ada[i] <- 1}
    else{accuracy_ada[i] <- 0}
    if (rf_predict_500[i] == test[i,"y"]){
      accuracy_rf_500[i] <- 1}
    else{accuracy_rf_500[i]<-0}
  }
  
  accuracy_rate_tree <- mean(accuracy_tree)
  mis_class_tree_03[t] <- 1-accuracy_rate_tree
  accuracy_rate_rf <- mean(accuracy_rf_50)
  mis_class_rf_50_03[t] <- 1-accuracy_rate_rf
  accuracy_rate_ada <- mean(accuracy_ada)
  mis_class_ada_03[t] <- 1-accuracy_rate_ada
  accuracy_rate_rf_500 <- mean(accuracy_rf_500)
  mis_class_rf_500_03[t] <- 1-accuracy_rate_rf_500
  mis_class_prune_03[t] <- 1-accuracy_prune
}

#####################Plot the simulation result
#dev.off()
#x = seq(1,50,by = 1)
#plot(x,mis_class_tree_02,type = 'o',col = 'blue',xlab = 'Experiment Times',
     #ylab = 'mis classification rate',ylim = c(0,0.85))
#lines(mis_class_prune_02, type ="o", col = "black")
#lines(mis_class_rf_50_02, type = "o",col = 'green')
#lines(mis_class_rf_500_02, type = "o",col = 'brown')
#lines(mis_class_ada_02, type = "o", col = 'red')
#legend("bottomright", c("DT", "Pruned","RF50","RF500",'Ada'), lty = c(1,1,1,1,1),y.intersp = 0.3,
       #col = c("blue", "black",'green','brown','red'), bty="n",inset=c(0,0.0001))
#title("Misclassification Rate Comparison")

#simulation result for case3 1000 times simulation, correspond to table 11 in the term paper
mean(mis_class_tree_03)
mean(mis_class_rf_50_03)
mean(mis_class_rf_500_03)
mean(mis_class_ada_03)
mean(mis_class_prune_03)
sd(mis_class_tree_03)
sd(mis_class_rf_50_03)
sd(mis_class_rf_500_03)
sd(mis_class_ada_03)
sd(mis_class_prune_03)
#box plot(correspond to the figure28 in the term paper)
boxplot(mis_class_tree_03, mis_class_rf_50_03, mis_class_rf_500_03,mis_class_ada_03,
        mis_class_prune_03,names = c('DT','RF50','RF500','Ada','pruned'),
        col=c('green','blue','red','brown','yellow'))
title('simulation result')

####Correspond to the figure1 in the term paper
data_test <- dgp_03(600)
sample <- sample.int(n = nrow(data_test), size = floor(.7*nrow(data_test)), replace = F)
train <- data_test[sample, ]
test  <- data_test[-sample, ]
rf_outcome <- c()
accuracy_rf <- c()
for (i in 1:200){
  rf_train <- randomForest(factor(y) ~ . , data = train, 
                              mtry = 2 ,ntree = i*10)# take all the predictors as sampled
  rf_predict<- predict(rf_train,test)
  for (k in 1:540){
    if (rf_predict[k] == test[k,"y"]){
      accuracy_rf[k] <- 1}
    else{accuracy_rf[k] <- 0}
  }
  rf_outcome[i] <- mean(accuracy_rf)
}

test_x <- seq(1,200,by = 1)
plot(test_x,rf_outcome,type = 'o', xlab = 'Number of Trees',ylab = "Accuracy rate")
title('Bagging experiment')




