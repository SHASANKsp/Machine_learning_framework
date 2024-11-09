#PCA
geneExpData=read.table("PCASampleData.csv", sep=",", header = T)
head(geneExpData)
summary(geneExpData)
scale(geneExpData)
plot(geneExpData,cex=1,col="blue",main="Gene 1 vs Gene 2",
     xlim = c(-1.5,1.5),ylim = c(-1.5,1.5))
abline(h=0,col="blue")
abline(v=0,col="blue")


#standardize <- function(x) {(x - mean(x))}
#scaledExp = apply(geneExpData, FUN = standardize, MARGIN = 2)
#head(geneExpData)
#head(scaledExp)
covMAt = cov(scaledExp)
eg <- eigen(covMAt)

rownames(eg$vectors) <- c("G1","G2")

colnames(eg$vectors) <- c("PC1","PC2")
loadings = eg$vectors

pc1Slope = eg$vectors[1,1]/eg$vectors[2,1]
pc2Slope = eg$vectors[1,2]/eg$vectors[2,2]

abline(0,pc1Slope, col="red")
abline(0,pc2Slope, col="green")

# See how much variation each eigenvector accounts for

pc1Var <- 100*round(eg$values[1]/sum(eg$values),digits=2)
pc2Var <- 100*round(eg$values[2]/sum(eg$values),digits=2)
xlab=paste("PC1 - ",pc1Var," % of variation",sep="")
ylab=paste("PC2 - ",pc2Var," % of variation",sep="")

# Multiply the scaled data by the eigen vectors (principal components)
dim(scaledExp)
scores <- scaledExp %*% loadings
head(scores)
head(geneExpData)


rownames(loadings) = colnames(geneExpData)
plot(scores,ylim=c(-1,1),main="Data in terms of EigenVectors / PCs")
abline(h=0,col="red")
abline(v=0,col="green")

cor(geneExpData[,1],geneExpData[,2])
cor(scores[,1],scores[,2])
biplot(pca)

# BiPlot
plot(scores[,1],scores[,2],main="My First BiPlot",
     ylim = c(-2,2), xlim = c(-2,2), type = "n")
text(scores[,1],scores[,2], rownames(scores),col="blue", cex=0.7)
rownames(scores)=seq(1:nrow(scores))
abline(h=0,col="red")
abline(v=0,col="green")

# This is to make the size of the lines more apparent
factor <- 2

# First plot the variables as vectors
arrows(0,0,loadings[,1],loadings[,2], col="red")
text(loadings[,1],loadings[,2],rownames(loadings), col="red", cex=1.2)

# Second plot the scores as points



# Using princomp
dim(data)
pca = princomp(geneExpData)
pca$loadings
pca$scores
head(pca$scores)
plot(pca)
biplot(pca)
