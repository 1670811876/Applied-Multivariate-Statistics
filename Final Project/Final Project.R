library(mvstats)

data <- read.csv("data.csv")
dim(data)
head(data)

wdbc.data <- as.matrix(data[, c(3:32)])
row.names(wdbc.data) <- data$id
diagnosis <- as.numeric(data$diagnosis == "M")

table(data$diagnosis)

library(corrplot)
cor.matrix <- data[, c(3:32)]
correlation <- round(cor(cor.matrix), 2)
cor(cor.matrix)
corrplot(correlation, diag=FALSE, method="color", order="FPC", tl.srt=90)

drop.list = c('perimeter_mean', 'area_mean', 'compactness_mean', 'concave points_mean', 'radius_se', 'perimeter_se', 'radius_worst',
              'perimeter_worst', 'compactness_worst', 'concave points_worst', 'compactness_se', 'concave points_se', 'texture_worst', 
              'area_worst')
wdbc.data <- as.data.frame(wdbc.data)
droped.data <- wdbc.data[, !names(wdbc.data) %in% drop.list]

## Principal Component Analysis
wdbc.pr <- princomp(droped.data, cor=TRUE)
screeplot(wdbc.pr, type="lines")
summary(wdbc.pr)
wdbc.pr$loadings
wdbc.pr$scores[, 1:6]  # Principal component score

## Discriminant analysis
wdbc.pcs <- wdbc.pr$scores[, 1:6]
head(wdbc.pcs, 10)
wdbc.pcst <- cbind(wdbc.pcs, diagnosis)
head(wdbc.pcst)

library(MASS)

# convert matrix to a dataframe
wdbc.pcst.df <- as.data.frame(wdbc.pcst)

# Linear Discriminant Model
wdbc.lda <- lda(diagnosis~Comp.1+Comp.2+Comp.3+Comp.4+Comp.5+Comp.6, data=wdbc.pcst.df)
wdbc.lda
wdbc.lda.predict <- predict(wdbc.lda, newdata = wdbc.pcst.df)
wdbc.lda.predict.class <- wdbc.lda.predict$class
wdbc.lda.predict.class <- as.data.frame(wdbc.lda.predict.class)
cbind(wdbc.pcst.df$diagnosis, wdbc.lda.predict.class)
tab <- table(wdbc.pcst.df$diagnosis, wdbc.lda.predict$class)
tab
sum(diag(prop.table(tab)))

# Quadratic Discriminant Model
wdbc.qda <- qda(diagnosis~Comp.1+Comp.2+Comp.3+Comp.4+Comp.5+Comp.6, data=wdbc.pcst.df)
wdbc.qda
wdbc.qda.predict <- predict(wdbc.qda, newdata = wdbc.pcst.df)
wdbc.qda.predict.class <- wdbc.qda.predict$class
wdbc.qda.predict.class <- as.data.frame(wdbc.qda.predict.class)
cbind(wdbc.pcst.df$diagnosis, wdbc.qda.predict.class)
tab <- table(wdbc.pcst.df$diagnosis, wdbc.qda.predict$class)
tab
sum(diag(prop.table(tab)))

## Factor Analysis
wdbc.fa1 <- factpc(droped.data, 6)  # Factor analysis
wdbc.fa1
wdbc.fa1$Vars
wdbc.fa2 <- factpc(droped.data, 6, rot="varimax")  # Use rotation factor analysis
wdbc.fa2
wdbc.fa2$Vars

## fa1 discriminant analysis
wdbc.fcs.fa1 <- wdbc.fa1$scores[, 1:6]
head(wdbc.fcs.fa1, 10)
wdbc.fcst.fa1 <- cbind(wdbc.fcs.fa1, diagnosis)
head(wdbc.fcst.fa1)

# convert matrix to a dataframe
wdbc.fcst.fa1.df <- as.data.frame(wdbc.fcst.fa1)

# fa1 linear discriminant analysis
wdbc.lda.fa1 <- lda(diagnosis~Factor1+Factor2+Factor3+Factor4+Factor5+Factor6, data=wdbc.fcst.fa1.df)
wdbc.lda.fa1
wdbc.lda.fa1.predict <- predict(wdbc.lda.fa1, newdata = wdbc.fcst.fa1.df)
wdbc.lda.fa1.predict.class <- wdbc.lda.fa1.predict$class
wdbc.lda.fa1.predict.class <- as.data.frame(wdbc.lda.fa1.predict.class)
cbind(wdbc.fcst.fa1.df$diagnosis, wdbc.lda.fa1.predict.class)
tab <- table(wdbc.fcst.fa1.df$diagnosis, wdbc.lda.fa1.predict$class)
tab
sum(diag(prop.table(tab)))

# fa1 quadratic discrimant analysis
wdbc.qda.fa1 <- qda(diagnosis~Factor1+Factor2+Factor3+Factor4+Factor5+Factor6, data=wdbc.fcst.fa1.df)
wdbc.qda.fa1
wdbc.qda.fa1.predict <- predict(wdbc.qda.fa1, newdata = wdbc.fcst.fa1.df)
wdbc.qda.fa1.predict.class <- wdbc.qda.fa1.predict$class
wdbc.qda.fa1.predict.class <- as.data.frame(wdbc.qda.fa1.predict.class)
cbind(wdbc.fcst.fa1.df$diagnosis, wdbc.qda.fa1.predict.class)
tab <- table(wdbc.fcst.fa1.df$diagnosis, wdbc.qda.fa1.predict$class)
tab
sum(diag(prop.table(tab)))

## fa1 discriminant analysis
wdbc.fcs.fa2 <- wdbc.fa2$scores[, 1:6]
head(wdbc.fcs.fa2, 10)
wdbc.fcst.fa2 <- cbind(wdbc.fcs.fa2, diagnosis)
head(wdbc.fcst.fa2)

# convert matrix to a dataframe
wdbc.fcst.fa2.df <- as.data.frame(wdbc.fcst.fa2)

# fa2 linear discriminant analysis
wdbc.lda.fa2 <- lda(diagnosis~Factor1+Factor2+Factor3+Factor4+Factor5+Factor6, data=wdbc.fcst.fa2.df)
wdbc.lda.fa2
wdbc.lda.fa2.predict <- predict(wdbc.lda.fa2, newdata = wdbc.fcst.fa2.df)
wdbc.lda.fa2.predict.class <- wdbc.lda.fa2.predict$class
wdbc.lda.fa2.predict.class <- as.data.frame(wdbc.lda.fa2.predict.class)
cbind(wdbc.fcst.fa2.df$diagnosis, wdbc.lda.fa2.predict.class)
tab <- table(wdbc.fcst.fa2.df$diagnosis, wdbc.lda.fa2.predict$class)
tab
sum(diag(prop.table(tab)))

# fa2 quadratic discrimant analysis
wdbc.qda.fa2 <- qda(diagnosis~Factor1+Factor2+Factor3+Factor4+Factor5+Factor6, data=wdbc.fcst.fa2.df)
wdbc.qda.fa2
wdbc.qda.fa2.predict <- predict(wdbc.qda.fa2, newdata = wdbc.fcst.fa2.df)
wdbc.qda.fa2.predict.class <- wdbc.qda.fa2.predict$class
wdbc.qda.fa2.predict.class <- as.data.frame(wdbc.qda.fa2.predict.class)
cbind(wdbc.fcst.fa2.df$diagnosis, wdbc.qda.fa2.predict.class)
tab <- table(wdbc.fcst.fa2.df$diagnosis, wdbc.qda.fa2.predict$class)
tab
sum(diag(prop.table(tab)))
