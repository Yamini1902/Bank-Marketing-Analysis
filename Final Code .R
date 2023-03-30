
#loading library
require(JOUSBoost)
require(pROC)
library(gbm)
library(leaps)
library(lattice)
library(bestglm)
require(glmnet)
library(Matrix)
library(glmnet)
require(doParallel)
library(doParallel)
library(foreach)
library(iterators)
library(parallel)
library(fastDummies)
library(randomForest)
library(dplyr)
library(ggplot2)
library(tidyr)
library(corrplot)
library(caret)
library(car)
citation("pROC")
library(pROC)
library(e1071)
library(xgboost)



#Load the data
# read table
dat <- read.csv("train.csv")
unseen <- read.csv("test.csv")

#creating a duplicate copy
dat0 <- dat
unseen0 <- unseen

#basic checks
View(dat)
dim(dat)
str(dat)

View(unseen)
dim(unseen)
str(unseen)

#Basic statistics 
summary(dat)

#Data Pre-Preprocessing 
#checking the correlation of variables and relation with outcome.

#1. Age
sum(dat$age < 25)
sum(dat$age < 25 & dat$outcome == 1)/sum(dat$outcome == 1)*100

sum(dat$age >= 25 & dat$age <= 58)
sum(dat$age >= 25 & dat$age <= 58 & dat$outcome == 1)/sum(dat$outcome == 1)*100

sum(dat$age > 58)
sum(dat$age > 58 & dat$outcome == 1)/sum(dat$outcome == 1)*100
#customers with age 25 to 58 have more possibility of opening savings account.

#2. Civil
sum(dat$civil ==  "single")
sum(dat$civil == "single" & dat$outcome == 1)/sum(dat$outcome == 1)*100

sum(dat$civil ==  "married")
sum(dat$civil == "married" & dat$outcome == 1)/sum(dat$outcome == 1)*100

sum(dat$civil ==  "divorced")
sum(dat$civil == "divorced" & dat$outcome == 1)/sum(dat$outcome == 1)*100

#customers who are married have more possibility of opening savings account followed by customers who are single.

#3. Credit
sum(dat$credit ==  "unknown")
sum(dat$credit ==  "yes")
sum(dat$credit ==  "no")
#majority of the customers in the data have default as unknown or no thus excluding the same from the data.

#4.hloan & Plaon
sum(dat$hloan ==  "yes")
sum(dat$hloan == "yes" & dat$outcome == 1)/sum(dat$outcome == 1)*100

sum(dat$ploan ==  "yes")
sum(dat$ploan == "yes" & dat$outcome == 1)/sum(dat$outcome == 1)*100

#customers having hloan are more prone to opening savings account, thus kept hloan and remvoed ploan from the data.

#5. Lcdays
sum(dat$lcdays ==  999)
sum(dat$lcdays == 999 & dat$outcome == 1)/sum(dat$outcome == 1)*100
#majority of the customers have lcdays with 999, thus converted the variable into binary 

#6. Ccontact & pcontact

sum(dat$ccontact <=  4)
sum(dat$ccontact <=  4 & dat$outcome == 1)/sum(dat$outcome == 1)*100


sum(dat$ccontact >  4)
sum(dat$ccontact >  4 & dat$outcome == 1)/sum(dat$outcome == 1)*100

sum(dat$pcontact <=  2)
sum(dat$pcontact <=  2 & dat$outcome == 1)/sum(dat$outcome == 1)*100

#customers who have ccontact 4+ have lesser impact on campaign and similar with pcontact >2

#7.presult
sum(dat$presult ==  "success")
sum(dat$presult ==  "success" & dat$outcome == 1)/sum(dat$outcome == 1)*100
#Customer have opted for previous campaigns have higher chance of opening savings account.

#Data Visualisation.

#outcome
ggplot(dat0, aes(x = outcome)) + 
  geom_bar() + 
  labs(x = "Outcome", y = "Count", title = "Opening Savings account")


#job
ggplot(dat, aes(x = job)) + 
  geom_bar() + 
  labs(x = "Job", y = "Count", title = "Distribution of Jobs")

#job
ggplot(dat, aes(x = civil)) + 
  geom_bar() + 
  labs(x = "Civil", y = "Count", title = "Distribution of Marital status")


#job
ggplot(dat, aes(x = month)) + 
  geom_bar() + 
  labs(x = "month", y = "Count", title = "Distribution of Month")

#age
boxplot(dat$age, main = "Box plot of Age")

#Variable reduction and treatments

#1. Making lcdays binary
dat$lcdays <- ifelse(dat$lcdays == 999, 0, 1)
unseen$lcdays <- ifelse(unseen$lcdays == 999, 0, 1)

#2. make outcome y
dat$y <- dat$outcome
dat$outcome <- NULL

#3. drop variables
dat$id <- NULL
dat$credit <- NULL
dat$ploan <- NULL

unseen$id <- NULL
unseen$credit <- NULL
unseen$ploan <- NULL

#4. Treating unknowns, imputing mode.

#replacing unknown
dat[dat == "unknown"] <- NA
sum(is.na(dat))

unseen[unseen == "unknown"] <- NA
sum(is.na(unseen))


get_mode <- function(x) {
  ux <- unique(x[!is.na(x)])
  ux[which.max(tabulate(match(x, ux)))]
}

# Replace NAs with mode of each column
for (col in colnames(dat)) {
  dat[is.na(dat[, col]), col] <- get_mode(dat[, col])
}

get_mode <- function(x) {
  ux <- unique(x[!is.na(x)])
  ux[which.max(tabulate(match(x, ux)))]
}

# Replace NAs with mode of each column
for (col in colnames(unseen)) {
  unseen[is.na(unseen[, col]), col] <- get_mode(unseen[, col])
}

#5. Fast dummies
dat <- dummy_cols(dat, select_columns = c("job","civil", "edu","hloan","ctype","presult",
                                          "month","day"),
                  remove_first_dummy = TRUE, remove_selected_columns = TRUE)

unseen <- dummy_cols(unseen, select_columns = c("job","civil", "edu","hloan","ctype","presult",
                                                "month","day"),
                     remove_first_dummy = TRUE, remove_selected_columns = TRUE)

colnames(dat)
colnames(unseen)

#6. Scaling the data

dat2 <- dat
dat2$y <- NULL
dat1_scaled <- scale(dat2)
View(dat1_scaled)
#combine id and outcome with scaled values
dat3 <- cbind(dat1_scaled, y = dat$y )


unseen_scaled <- scale(unseen)
View(unseen_scaled)
unseen1 <- unseen_scaled
unseen1 <- as.data.frame(unseen1)

colnames(unseen)


unseen1$y <- 0
View(unseen1)
colnames(unseen1)
colnames(dat3)

dat6 <- dat3

#------------ XXXXX- ---------------- XXXXX -----------XXXXXXX-------------- XXXXX- ---------------- XXXXX -----------XXXXXXX
#Split into test and train #dat 3

set.seed(42)
train <- sample(1:nrow(dat6) ,nrow(dat6)*0.7)
# Print the first few observations in your “train” sample using the code below.
# Compare it with my output to ensure they s they match up.
head(train)

#We can then use the new variable train to select the according rows in our dataset and store this selection into a new variable dat_train.
dat_train <- dat6[train, ]
dim(dat_train)
colnames(dat_train)
## Check out the train dataset you just created.
nrow(dat_train)

## If you want to test that we actually got 70% of our original data, you can run:
nrow(dat_train) / nrow(dat6) * 100

#We build this test dataset by deleting all the rows that are identified in the “train” list.
dat_test <- dat6[-train, ]
nrow(dat_test)
# head(dat_test)
nrow(dat_test) / nrow(dat6) * 100
colnames(dat_test)

dat_train <- data.frame(dat_train)
dat_test <- data.frame(dat_test)


X <- dat_train
Xtest <- dat_test
colnames(X)
colnames(Xtest)

# adaboost wants us to have class 1 and -1
y1 <- ifelse(X[,45] == 1, 1, -1)


#------------ XXXXX- ---------------- XXXXX -----------XXXXXXX-------------- XXXXX- ---------------- XXXXX -----------XXXXXXX
#Running Models
#1. Ada
ada <- adaboost(X= as.matrix(X[,1:44]), y = y1, tree_depth = 1, n_rounds = 1000)
ada.prob <- predict(ada, as.matrix(Xtest), type = "prob")
ada.roc <- roc(Xtest$y, ada.prob, plot = FALSE, quiet = TRUE)
auc(ada.roc)



#2.Gradient boosting model 
set.seed(42)
model_gbm <- gbm(y ~ ., data = dat_train, distribution = "bernoulli", n.trees = 1000, interaction.depth = 4, shrinkage = 0.01, bag.fraction = 0.5, train.fraction = 1)
pred_gbm <- predict(model_gbm, newdata = dat_test, n.trees = 1000, type = "response")
roc_gbm <- roc(dat_test$y, pred_gbm, plot = TRUE, grid = TRUE, col = "red")
auc(roc_gbm)

#check imp variables
var_importance <- summary(model_gbm, plot = FALSE)

#3. Random forest
rf_model <- randomForest(y ~ ., data = dat_train)
pred_prob <- predict(rf_model, newdata = dat_test, type = "response")
roc_obj <- roc(dat_test$y, pred_prob)
auc <- auc(roc_obj)
auc

#4.Logistic model1 
log.model_full <- glm(y ~ ., data = dat_train, family = binomial)
summary(log.model_full)
pred_full <- predict(log.model_full, newdata = dat_test, type = "response")
roc_full <- roc(dat_test$y, pred_full, plot = TRUE, grid = TRUE, col = "blue")
auc(roc_full) 


#5. forward selection
# Estimate model with no predictors
null <- glm(y ~ 1, data = dat_train, family = binomial)
null

#BIC auc 
forward_model_BIC <- step(null, scope = formula(log.model_full), direction = "forward", k=log(nrow(dat_train)))
summary(forward_model_BIC)
pred_forward_BIC <- predict(forward_model_BIC, newdata = dat_test, type = "response")
roc_forward_BIC <- roc(dat_test$y, pred_forward_BIC, plot = TRUE, grid = TRUE, col = "green")
auc(roc_forward_BIC)

#AIC 
forward_model_AIC <- step(null, scope = formula(log.model_full), direction = "forward")
summary(forward_model_AIC)
pred_forward_AIC <- predict(forward_model_AIC, newdata = dat_test, type = "response")
roc_forward_AIC <- roc(dat_test$y, pred_forward_AIC, plot = TRUE, grid = TRUE, col = "green")
auc(roc_forward_AIC)


#6. XGBoost
xgb_params = list(
  objective = "binary:logistic",
  eta = 0.01,
  gamma = 0,
  max.depth = 4,
  min_child_weight = 1,
  eval_metric = "auc"
)

xgb <- xgboost(data = as.matrix(X[,1:44]), label = as.vector(X$y),
               params = xgb_params, nthread = 4, nrounds = 1000, verbose = FALSE,
               early_stopping_rounds = 10)

# evaluate performance
xgb.prob <- predict(xgb, as.matrix(Xtest[, 1:44]), type = "prob")
xgb.roc <- roc(Xtest$y, xgb.prob, plot=FALSE, quiet = TRUE)
auc(xgb.roc)


# Look into feature importance
impMat <-xgb.importance(colnames(X[,1:44]), model = xgb)
# head gives us the top 10 variables
head(impMat, 10)
# Nice graph of top 10 variables
xgb.plot.importance(impMat[1:10,])

a <- tail(impMat, 10)
# Nice graph of top 10 variables
xgb.plot.importance(a[1:10,])

#------------ XXXXX- ---------------- XXXXX -----------XXXXXXX-------------- XXXXX- ---------------- XXXXX -----------XXXXXXX
#Analysing the AUC
# Plot ROC curves
plot(roc_full, col = "blue", main = "Analysing ROC")
lines(roc_forward_BIC, col = "green")
lines(roc_forward_AIC, col = "red")
lines(xgb.roc, col = "black")
lines(roc_obj, col = "pink")
lines(roc_gbm, col = "grey")
lines(ada.roc, col = "orange")

# Add legend
legend("bottomright", legend=c("Full Model", "Forward Selection (BIC)", "Forward Selection (AIC)", "XGB", "Random Forest", "Gradient Boosting Model", "AdaBoost"), col=c("blue", "green", "red", "black", "pink", "grey", "orange"), lty=1)



# Calculate AUC values and store them in separate variables
auc_log <- auc(roc_full)
auc_forward_BIC <- auc(roc_forward_BIC)
auc_forward_AIC <- auc(roc_forward_AIC)
auc_xgb <- auc(xgb.roc)
auc_rf <- auc(roc_obj)
auc_gbm <- auc(roc_gbm)
auc_ada <- auc(ada.roc)

# Find the maximum AUC value and the corresponding method
max_auc <- max(auc_full, auc_forward_BIC, auc_forward_AIC, auc_xgb, auc_rf, auc_gbm, auc_ada)

#------------ XXXXX- ---------------- XXXXX -----------XXXXXXX-------------- XXXXX- ---------------- XXXXX -----------XXXXXXX
#outcome file

colnames(dat_test)
colnames(unseen1)
unseen1$y <- 0
pred_gbm_unseen <- predict(model_gbm, newdata = unseen1, n.trees = 1000, type = "response")
unseen1$outcome <- pred_gbm_unseen
unseen1$id <- unseen0$id
selected_cols <- c("id", "outcome")
output <- unseen1[selected_cols]
# Export the selected columns to a CSV file
write.csv(output, "outcomeGBM.csv", row.names = FALSE)

#------------ XXXXX- ---------------- XXXXX -----------XXXXXXX-------------- XXXXX- ---------------- XXXXX -----------XXXXXXX

