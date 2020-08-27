TreeGenerate_missing <- function(X, Y, AttributesSet, AttributeValue=NULL, X.globel=X,w){
    # 功能：
    # 生成缺失值决策树
    
    # 输入：
    # X：样本的协变量
    # Y：样本的分类变量
    # AttributeSet：属性集
    # AttributeValue：该参数为方便递归设置，在建立决策树时不需要输入该参数
    # w：权重
    
    # 输出：
    # node：以node为根节点的决策树，决策树是以list形式构造的
    
    # 生成节点
    node <- CreateNode(X,Y, AttributeValue=AttributeValue)
    # 情形(1)
    if (length(unique(Y)) == 1){
        node <- LeafNode(node,as.character(unique(Y)), AttributeValue=AttributeValue)
        return(node)
    }
    # 情形(2)
    if ( (length( AttributesSet) == 0) || (dim(X[!duplicated(X),AttributesSet])[1] == 1) ){
        C <- GetMostCategories(Y)
        C <- as.character(C)
        node <- LeafNode(node,C,AttributeValue=AttributeValue)
        return(node)
    }
    # 找到最优划分属性
    PartitionAttributes <- SelectPartitionAttributes_missing(X, Y, AttributesSet, w)
    # 将最优划分属性储存在节点的PartitionAttributes中
    node$PartitionAttributes <- colnames(X)[PartitionAttributes]
    # 根据最优划分属性的取值进行分支操作
    AttributesSet <- AttributesSet[which(AttributesSet != PartitionAttributes)]
    for (i in unique(X.globel[!is.na(X.globel[,PartitionAttributes]),PartitionAttributes])){
        SubSamplesIndex <- X[,PartitionAttributes] == i # 返回子集索引
        SubSamplesIndex[is.na(X[,PartitionAttributes])] <- TRUE
        # 进行分支
        node <- Branch(node, X[SubSamplesIndex,], Y[SubSamplesIndex])
        # 情形（3）
        if (sum(X[!is.na(X[,PartitionAttributes]),PartitionAttributes] == i) == 0){
            C <- GetMostCategories(Y)
            C <- as.character(C)
            node$SubNode[[length(node$SubNode)]] <- LeafNode(node$SubNode[[length(node$SubNode)]], C,i)
            return(node)
        }else{
            # 计算r
            r <- sum(X[!is.na(X[,node$PartitionAttributes]),node$PartitionAttributes] == i)/length(X[!is.na(X[,node$PartitionAttributes]),node$PartitionAttributes])
            # 计算权重
            w[is.na(X[,node$PartitionAttributes])] <- w[is.na(X[,node$PartitionAttributes])]*r
            # 递归到下一层
            SubNode_new <- TreeGenerate_missing(X[SubSamplesIndex,], Y[SubSamplesIndex], AttributesSet, AttributeValue=i,X.globel=X.globel,w=w[SubSamplesIndex])
            # 将w还原回去以便重新进行权重赋值
            w[is.na(X[,node$PartitionAttributes])] <- w[is.na(X[,node$PartitionAttributes])]/r
            node$SubNode[[length(node$SubNode)]] <- SubNode_new
        }
    }
    return(node)
}









SelectPartitionAttributes_missing <- function(X, Y, AttributesSet,w){
    # 生成一个2列的矩阵，其行数为当前可选属性集中属性的个数，用该矩阵来存放不同属性对应的信息增益
    Gain <- matrix(nrow=length(AttributesSet),ncol=2)
    Gain[,1] <- AttributesSet # 设置该矩阵的第一列为属性集中的属性
    for(i in AttributesSet){
        if(sum(is.na(X[,i]))==length(X[,i])){
            Gain[which(Gain[,1]==i),2] <- 1e-5
        }else{
            # 计算按属性i进行分类时的Ent(D)
            # 先找到属性i下没有缺失值的样本
            Index <- !is.na(X[,i])
            # 根据无缺失值样本计算父节点信息熵
            No_missing_samples_X <- X[Index,]
            No_missing_samples_Y <- Y[Index]
            K_Class <- length(unique(No_missing_samples_Y))
            p_k <- 1:K_Class
            # 计算各个类别的比例
            for(k in 1:K_Class){
                p_k[k] <- ((unique(No_missing_samples_Y)[k] == No_missing_samples_Y) %*% w[Index])/sum(w[Index])
            }
            #计算父节点信息熵
            Ent <- -(p_k %*% log2(p_k))
            # 计算信息增益
            # 按公式计算各个属性对应的信息熵
            pho <- (Index %*% w)/sum(w)
            v <- length(unique(X[Index,i]))
            Sub_Ent <- 0
            for(j in 1:v){
                v_Index <- (X[,i] == unique(X[Index,i])[j])
                v_Index[is.na(v_Index)] <- FALSE
                r<- (v_Index %*% w)/sum(w[Index])
                for(k in 1:K_Class){
                    Sub_Index <- v_Index
                    Sub_Index[!(Y == unique(Y)[k])] <- FALSE
                    p_k[k] <- (Sub_Index %*% w)/sum(w[v_Index])
                }
                log_pk <- log2(p_k)
                log_pk[which(log_pk == -Inf)] <- 0
                Sub_Ent <- r*(-(p_k %*% log_pk)) + Sub_Ent # 子节点对应的信息熵
            }
            Gain[which(Gain[,1]==i),2] <- pho * (Ent - Sub_Ent) #设置矩阵的第二列为不同属性对应的信息增益
        }
    }
    PartitionAttributes <- Gain[which.max(Gain[,2]),1] # 返回让信息增益达到最大的属性
    return(PartitionAttributes)
}

setwd("E:/projects/决策树/")
library(readxl)
mydata <- readxl::read_excel('E:/projects/决策树/西瓜书数据3.xlsx')
mydata <- as.data.frame(mydata)
X <- mydata[,-dim(mydata)[2]]
Y <- mydata[,dim(mydata)[2]]
AttributesSet <- 1:(dim(mydata)[2]-1)
w <- rep(1,each=length(Y))
mytree_missing <- TreeGenerate_missing(X, Y, AttributesSet, AttributeValue=NULL, X.globel=X,w=w)
plot.tree(mytree_missing)
