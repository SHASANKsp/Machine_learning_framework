#Clustering

setwd("/Users/pankazsingh/Library/Mobile Documents/com~apple~CloudDocs/Teaching/ML-2021/Lab")

data = read.csv("datasets/expressionData.csv", header = T)
head(data)
cnames = data$Gene
expData = data[,-1]
rownames(expData) = cnames
head(expData)

expDataT = t(expData)
expDataT1 = expDataT[,1:30]
head(expDataT1)
correalations = cor(expDataT1)
dim(correalations)

is.na(expDataT1)

# correalations_abs = ifelse(correalations<0, (correalations*(-1)), correalations)
#correalations_abs = abs(correalations)

cluster = corrplot(correalations, order = "hclust", hclust.method = "ward.D", 
                   addrect = 5)

?corrplot


# kmeans clustering

dim(expData)
head(expData)
expData = na.omit(expData)
dim(expData)

clusters = kmeans(expData, 4)
clusters
clusters$cluster
write.csv(clusters$cluster, "clusters.csv")

table(clusters$cluster)

clusters$size
clusters$iter

clusters$centers
clusters$totss
clusters = kmeans(expData, 4)
clusters$betweenss
clusters$withinss
sum(clusters$withinss) + clusters$betweenss


# heirarchical clustering
dim(expData)
class(expData)
expData = as.matrix(expData[1:30,])
distMAt = dist(expData)
head(distMAt)
hclusters = hclust(distMAt, method = "single") 
plot(hclusters)
plot(hclusters, hang = -1)


?hclust
plot(hclusters)
rect.hclust(hclusters, h = 4)

?rect.hclust
grps = cutree(hclusters,h = 4)
grps
table(grps)
cltable = cbind(expData,grps)
head(cltable)


# heatmap
cp = colorRampPalette(c("red","yellow","green"))
heatmap.2(as.matrix(expData), cexRow = 0.3, cexCol = 1,
          col=cp)
?heatmap.2



dist(expData)
library(dbscan)
dbs = dbscan(expData[,c(1,2)], eps = 0.3,minPts = 3)
plot(expData[,1], expData[,2], col = ((dbs$cluster)+1))
dbs$cluster
table(dbs$cluster)

kClus = kmeans(expData[,c(1,2)], 3)
table(kClus$cluster)
plot(expData[,1], expData[,2], col = ((kClus$cluster)))

expDataT = t(expData)

dbs = dbscan(expData, eps = 1,minPts = 3)
dbs$cluster
table(dbs$cluster)
plot(expData, col = ((dbs$cluster)+1))

plot(expData, col = ((kClus$cluster)))
