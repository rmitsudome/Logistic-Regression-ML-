#1. Sample code number: id number 
#2. Clump Thickness: 1 - 10 
#3. Uniformity of Cell Size: 1 - 10 
#4. Uniformity of Cell Shape: 1 - 10 
#5. Marginal Adhesion: 1 - 10 
#6. Single Epithelial Cell Size: 1 - 10 
#7. Bare Nuclei: 1 - 10 
#8. Bland Chromatin: 1 - 10 
#9. Normal Nucleoli: 1 - 10 
#10. Mitoses: 1 - 10 
#11. Class: (2 for benign, 4 for malignant)

BreastCancer = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                          sep = ",")

dim(BreastCancer)
head(BreastCancer)
names(BreastCancer) = c("ID", "Clump_Thick", "Cell_Size", "Cell_Shape"
                        , "Adhesion", "Epithelial_Size", "Nuclei",
                        "Chromatin", "Nucleoli", "Mitoses", "Class")

sum(is.na(BreastCancer))

#delete all ? in data set
Question_row = which(BreastCancer == "?", arr.ind = TRUE)
BreastCancer = BreastCancer[-c(Question_row[1:16,]), ]
nrow(BreastCancer)

#change all predictors to numeric
for(i in 2:10) {
  BreastCancer[, i] <- as.numeric(as.character(BreastCancer[, i]))
}

#make Class out of 0 (benign) and 1 (malignant)
BreastCancer$Class <- ifelse(BreastCancer$Class == "4", 1, 0)

#change dependent variable to factor
BreastCancer[,11] = as.factor(BreastCancer[,11])

#need to randomly split the data into training and test samples
#Since response variable is binary categorical variable need to
#make sure training data has approximately = proportion of classes.

table(BreastCancer$Class)

library(caret)
'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.


#randomly put 70% of orig. data in train, the rest in test
set.seed(100)
trainIndex = createDataPartition(BreastCancer$Class, 
              p=0.7, list = F)  # 70% training data
train = BreastCancer[trainIndex, ]
test = BreastCancer[-trainIndex, ]

table(train$Class) #around 2x of 0 than 1

#Down sampling
#Majority class randomly down sampled to same size as smaller class

set.seed(100)
#Selects all columns but Class for x
#y must be factor variable
down_train <- downSample(x = train[, colnames(train) %ni% "Class"],
                         y = train$Class)
table(down_train$Class)

#Up Sampling
#rows from minority class repeatedly sampled till reaches 
#equal size as majority class

set.seed(100)
up_train <- upSample(x = train[, colnames(train) %ni% "Class"],
                     y = train$Class)
table(up_train$Class)

#for this example, will use down_train as training data
train = down_train

#graphing data 
par(mfrow=c(3,3))
barplot(table(train$Class, train$Clump_Thick), main = "Clump Thick to Class",
     ylab = "Count", xlab = "Clump_Thick")
barplot(table(train$Class, train$Cell_Size), main = "Cell Size to Class",
     ylab = "Count", xlab = "Cell_Size")
barplot(table(train$Class, train$Cell_Shape), main = "Cell Shape to Class",
     ylab = "Count", xlab = "Cell_Shape")
barplot(table(train$Class, train$Adhesion), main = "Adhesion to Class",
     ylab = "Count", xlab = "Adhesion")
barplot(table(train$Class, train$Epithelial_Size), main = "Epithelial Size to Class",
     ylab = "Count", xlab = "Epithelial_size")
barplot(table(train$Class, train$Nuclei), main = "Nuclei to Class",
     ylab = "Count", xlab = "Nuclei")
barplot(table(train$Class, train$Nucleoli), main = "Nucleoli to Class",
     ylab = "Count", xlab = "Nucleoli")
barplot(table(train$Class, train$Chromatin), main = "Chromatin to Class",
     ylab = "Count", xlab = "Chromatin")
barplot(table(train$Class, train$Mitoses), main = "Mitoses to Class",
     ylab = "Count", xlab = "Mitoses")


#logistic regression model
head(train)
logit.model =  glm(Class ~ Clump_Thick + Cell_Size + Cell_Shape
                    + Adhesion + Epithelial_Size + Nuclei
                    + Chromatin + Nucleoli + Mitoses, 
                   family = binomial(link = logit), data = train)
logit.nullmodel =  glm(Class ~ 1, family = binomial(logit), data = train)
summary(logit.model)

#model selection
library(bestglm)

best.subset.AIC = bestglm(Xy = train, family = binomial(link=logit),IC = "AIC",method = "exhaustive")
best.subset.AIC
best.subset.BIC = bestglm(Xy = train, family = binomial(link=logit),IC = "BIC",method = "exhaustive")
best.subset.BIC

forward.model2 = step(logit.nullmodel, scope = list(lower = logit.nullmodel, upper = logit.model), direction = "forward",trace = FALSE)
backward.model2 = step(logit.model, scope = list(lower = logit.nullmodel, upper = logit.model), direction = "backward",trace = FALSE)
FB.model2 = step(logit.nullmodel, scope = list(lower = logit.nullmodel, upper = logit.model), direction = "both",trace = FALSE)
BF.model2 = step(logit.model, scope = list(lower = logit.nullmodel, upper = logit.model), direction = "both",trace = FALSE)

#Best model: Class~Clump_Thick +Cell_Shape+Nucleoli+Nuclei+Chromatin
#other best: Class~Clump_Thick +Cell_Shape+Nucleoli+Nuclei+Epithelial_Size
#3 vs 2

small.model = glm(Class~Clump_Thick +Cell_Shape+Nucleoli+Nuclei+Chromatin, 
                  family = binomial(logit), data = train) #best model based off model selection
summary(small.model)

#likelihood ratio test to test small model to full model
L0 = logLik(small.model)
L1 = logLik(logit.model)
LR.test = as.numeric(-2*(L0 - L1))
LR.pval = pchisq(LR.test, df = 5 - 3,lower.tail = F )
LR.pval #pval = 0.2294315 > 0.05, fail to reject null
#high value means obs. result more likely to occur under H0

#confidence interval
confint(small.model)
#general interpretation of model
#With every one increase in Nuclei, the odds of malignant 
#increases by between exp(0.2825  1.147).

#hosmer lemeshow test to see if model was good fit
install.packages("ResourceSelection")
library(ResourceSelection)
HL.test2 = hoslem.test(small.model$y, small.model$fitted.values,g = 10)
HL.test2 #high pval, fail to reject, model good

#predictive power
r = cor(small.model$y,small.model$fitted.values)
r

#Prediction with test data using model

p <- predict(small.model, newdata = test, type = "response")

y_pred_num <- ifelse(p > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1)) #result from algo
y_act <- test$Class

mean(y_pred == y_act)

#double check prediction

#error matrix
pi0 =0.50
my.table = table(truth = small.model$y,predict = ifelse(fitted(small.model)>pi0,1,0))
my.table

#percentage of total error/ overall error rate
error.rate2 = 1-sum(diag(my.table))/sum(my.table)
error.rate2


