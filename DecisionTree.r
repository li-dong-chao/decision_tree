# -*- coding:utf-8 -*-

# 目标：实现周志华老师《机器学习》书中决策树相关代码
# Time: 2020-03-09


#生成中间节点[生成决策树用]
CreateNode <- function(X, Y, PartitionAttributes=NULL, AttributeValue=NULL,SubNode=NULL){
    # 功能：
    # 生成节点
    
    # 输入：
    # X：样本的协变量
    # Y：样本的类别变量
    # PartitionAttributes：表示该节点最优的划分属性
    # AttributeValue：表示该节点在其对应父节点的最优划分属性上的取值
    # SubNode：表示该节点的子节点，子节点用列表表示
    
    # 输出：
    # node：返回一个list类型的节点
    
    node <- list(X=X, Y=Y, PartitionAttributes=PartitionAttributes,AttributeValue=AttributeValue, SubNode=SubNode) # 定义节点
    return(node)
}

# 返回当前样本集中数量最多的类[生成决策树用]
GetMostCategories <- function(Y){
    # 功能：
    # 返回样本数量最多的类
    
    # 输入：
    # Y：当前数据集的类别变量
    
    # 输出：
    # C：返回当前数据样本中占比最大的类别
    
    df <- as.data.frame(table(Y)) # 将table转化为data.frame
    C <- df[(which.max(df$Freq)),1] # 找到数据样本中占比最大的类别
    return(C)
}

# 设置为叶节点[生成决策树用]
LeafNode <- function(node, class, AttributeValue=NULL){
    # 功能：
    # 将输入函数的节点设置为叶节点
    
    # 输入：
    # node：该函数之前生成的一个节点
    # class：所要设置的叶节点的类别
    # AttributeValue：表示该节点在其对应父节点的最优划分属性上的取值
    
    # 输出：
    # node：返回输入的那个节点，但是这个节点已经b被设置成了叶节点，列表中的元素发生了一些改变
    
    node <- list(X=node$X, Y=node$Y, class=class, AttributeValue=AttributeValue) # 定义叶节点
    return(node)
}

# 进行分支[生成决策树用]
Branch <- function(node, X, Y){
    # 功能：
    # 对函数输入的节点进行分支操作
    
    # 输入：
    # node：需要进行分支的节点
    # X：需要划分入新生成节点的样本对应的协变量
    # Y：需要划分入新生成节点的样本对应的类别变量
    
    #输出：
    # node：返回输入的那个节点，但是这个节点已经进行了分支，多了一个子节点
    
    SubNode <-  CreateNode(X=X, Y=Y, PartitionAttributes=NULL, AttributeValue=NULL, SubNode = NULL) # 生成子节点
    node$SubNode[[length(node$SubNode)+1]] <- SubNode
    return(node)
}

# 找最优划分属性[生成决策树用]
SelectPartitionAttributes <- function(X, Y, AttributesSet, method='ID3'){
    # 功能：
    # 该函数提供了三种寻找最优划分属性的方法，分别为'ID3'、'C4.5'、'CART'，默认方法为'ID3'，对含有连续属性的数据生成决策树的方法集成在了'ID3'中
    
    # 输入：
    # X：当前样本集中样本的协变量
    # Y：当前样本集中样本的类别变量
    # AttributeSet：当前可供选择的属性集
    # method：选择最优划分属性的方法，共有三个可选参数，分别为'ID3'、'C4.5'、'CART'，对应三种决策树
    
    # PartitionAttributes：返回最优划分属性
    
    # ID3
    if (method == 'ID3'){
        K_Class <- length(unique(Y))
        # 计算各个类别的比例
        df <- as.data.frame(table(Y))
        p_k <- df[,2]/length(Y)
        #计算父节点信息熵
        Ent <- -(p_k %*% log2(p_k))
        # 生成一个3列的矩阵，其行数为当前可选属性集中属性的个数，用该矩阵第二列来存放不同属性对应的信息增益，第三列存放连续属性的最优分割点
        Gain <- matrix(nrow=length(AttributesSet),ncol=3)
        Gain[,1] <- AttributesSet # 设置该矩阵的第一列为属性集中的属性
        # 计算各个属性对应的信息熵
        for(i in AttributesSet){
            # 处理有连续属性的情况
            if(class(X[,i])=='numeric'){
                # 利用OptimalPartition函数找到连续属性的最优分割点
                Optimal.Partition <- OptimalPartition(X[,i],Y)
                # 把最优分割点放进Gain的第三列中的对应位置上
                Gain[which(Gain[,1]==i),3] <- Optimal.Partition[1]
                # 把该连续属性的信息熵放在Gain的第二列对应位置上
                Sub_Ent <- Optimal.Partition[2]
            }else{
                # 处理离散属性的情况
                # 下面几行代码可以计算离散属性的信息熵
                Sub_Ent <- 0
                v <- length(unique(X[,i]))
                for(j in 1:v){
                    Index <- which(X[,i] == unique(X[,i])[j])
                    Sub_Y <- Y[Index]
                    df <- as.data.frame(table(Sub_Y))
                    p_k <- df[,2]/length(Sub_Y)
                    log_pk <- log2(p_k)
                    log_pk[which(log_pk == -Inf)] <- 0
                    Sub_Ent <- -(p_k %*% log_pk) *length(Sub_Y)/length(Y) + Sub_Ent # 子节点对应的信息熵
                }
            }
            Gain[which(Gain[,1]==i),2] <- Ent - Sub_Ent #设置矩阵的第二列为不同属性对应的信息增益
        }
        PartitionAttributes <- Gain[which.max(Gain[,2]),c(1,3)] # 返回让信息增益达到最大的属性
        
    }
    # C4.5
    if (method == 'C4.5'){
        K_Class <- length(unique(Y))
        # 计算各个类别的比例
        df <- as.data.frame(table(Y))
        p_k <- df[,2]/length(Y)
        #计算父节点信息熵
        Ent <- -(p_k %*% log2(p_k))
        # 生成一个2列的矩阵，其行数为当前可选属性集中属性的个数，用该矩阵来存放不同属性对应的增益率
        Gain_ratio <- matrix(0,nrow=length(AttributesSet),ncol=3)
        Gain_ratio[,1] <- AttributesSet # 设置该矩阵的第一列为属性集中的属性
        # 计算各个属性对应的信息熵以及权重
        for(i in AttributesSet){
            Sub_Ent <- 0
            IV <- 0
            v <- length(unique(X[,i]))
            for(j in 1:v){
                Index <- which(X[,i] == unique(X[,i])[j])
                Sub_Y <- Y[Index]
                df <- as.data.frame(table(Sub_Y))
                p_k <- df[,2]/length(Sub_Y)
                log_pk <- log2(p_k)
                log_pk[which(log_pk == -Inf)] <- 0
                IV_v <- length(Sub_Y)/length(Y) 
                IV <- -(IV_v*log2(IV_v)) + IV # 信息增益率计算公式中的分母
                Sub_Ent <- -(p_k %*% log_pk) *IV_v  + Sub_Ent # 信息熵
            }
            Gain_ratio[which(Gain_ratio[,1]==i),3] <- (Ent - Sub_Ent) # Gain_ratio的第三列为信息增益
            Gain_ratio[which(Gain_ratio[,1]==i),2] <- (Ent - Sub_Ent)/IV # Gain_ratio第二列为信息增益率
        }
        Gain_ratio <- Gain_ratio[which(Gain_ratio[,3] >= mean(Gain_ratio[,3])),] # C4.5的启发式所在（高于平均值）
        if(length(as.matrix(Gain_ratio)) == dim(as.matrix(Gain_ratio))[1]){
            Gain_ratio <- t(as.matrix(Gain_ratio))
        }
        PartitionAttributes <- Gain_ratio[which.max(Gain_ratio[,2]),1] # 返回使信息增益率达到最大的属性
    }
    # CART
    if (method == 'CART'){
        # 生成一个2列的矩阵，其行数为当前可选属性集中属性的个数，用该矩阵来存放不同属性对应的基尼指数
        Gini_index <- matrix(0,ncol = 2,nrow = length(AttributesSet))
        Gini_index[,1] <- AttributesSet # 设置该矩阵的第一列为属性集中的属性
        # 计算各个属性对应的基尼指数
        for(i in AttributesSet){
            Gini_index_v <- 0
            v <- length(unique(X[,i]))
            for(j in 1:v){
                Index <- which(X[,i] == unique(X[,i])[j])
                Sub_Y <- Y[Index]
                df <- as.data.frame(table(Sub_Y))
                p_k <- df[,2]/length(Sub_Y)
                Gini <- 1 - sum(p_k**2)
                Gini_index_v <- length(Sub_Y)/length(Y)*Gini + Gini_index_v # 基尼指数
            }
            Gini_index[which(Gini_index[,1]==i),2] <- Gini_index_v
        }
        PartitionAttributes <- Gini_index[which.min(Gini_index[,2]),1] # 返回使基尼指数达到最小的属性
    }
    return(PartitionAttributes)
}

# 生成决策树
TreeGenerate <- function(X, Y, AttributesSet, method='ID3', AttributeValue=NULL, X.globel=X){
    # 功能：
    # 生成决策树
    
    # 输入：
    # X：样本的协变量
    # Y：样本的分类变量
    # AttributeSet：属性集
    # AttributeValue：该参数为方便递归设置，在建立决策树时不需要输入该参数
    # method：建立决策树的类型，共有三个可选参数，分别为'ID3'、'C4.5'、'CART'，对应三种决策树
    
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
    PartitionAttributes <- SelectPartitionAttributes(X, Y, AttributesSet, method=method)
    # 对离散型属性的处理(离散属性返回的PartitionAttributes中第二列为NA)
    if(is.na(PartitionAttributes[2])){
        # 得到最优属性索引
        PartitionAttributes <- PartitionAttributes[1]
        # 将最优划分属性储存在节点的PartitionAttributes中
        node$PartitionAttributes <- colnames(X)[PartitionAttributes]
        # 在属性集中删掉刚刚选出的最优划分属性
        AttributesSet <- AttributesSet[which(AttributesSet != PartitionAttributes)]
        # 根据最优划分属性的取值进行分支操作
        for (i in unique(X.globel[,PartitionAttributes])){
            SubSamplesIndex <- which(X[,PartitionAttributes] == i) # 返回子集索引
            # 进行分支
            node <- Branch(node, X[SubSamplesIndex,], Y[SubSamplesIndex])
            # 情形(3)
            if (sum(X[,PartitionAttributes] == i) == 0){
                C <- GetMostCategories(Y)
                C <- as.character(C)
                node$SubNode[[length(node$SubNode)]] <- LeafNode(node$SubNode[[length(node$SubNode)]], C,i)
                return(node)
            }else{
                # 进行递归，往下继续生成树
                SubNode_new <- TreeGenerate(X[SubSamplesIndex,], Y[SubSamplesIndex], AttributesSet, method = method, AttributeValue=i,X.globel=X.globel)
                node$SubNode[[length(node$SubNode)]] <- SubNode_new
            }
        }
        return(node)
    }else{
        # 对连续型属性的处理(连续属性返回的PartitionAttributes中第二列不是NA，而是最优的分割点)
        Optimal.Partition <- PartitionAttributes[2]
        # 最优划分属性索引
        PartitionAttributes <- PartitionAttributes[1]
        # 将最优划分属性储存在节点的PartitionAttributes中
        node$PartitionAttributes <- colnames(X)[PartitionAttributes]
        
        # 连续型属性不用剔除在属性集中剔除
        
        # 由于对连续属性进行分割后，只会生成两个分支，这里没有用循环，直接写出了两种结果的分支
        # 生成分支一
        # 找到样本集中当前连续属性取值小于分割点的子集索引
        SubSamplesIndex <- which(X[,PartitionAttributes] <= Optimal.Partition) # 返回子集索引
        # 将'<=最优分割点'设置为第一个子节点的属性取值
        AttributeValue <- paste('<=',Optimal.Partition,sep = '')
        # 生成分支
        node <- Branch(node, X[SubSamplesIndex,], Y[SubSamplesIndex])
        # 情形(3)
        if(sum(X[,PartitionAttributes] <= Optimal.Partition) == 0){
            C <- GetMostCategories(Y)
            C <- as.character(C)
            node$SubNode[[length(node$SubNode)]] <- LeafNode(node$SubNode[[length(node$SubNode)]], C,AttributeValue)
            return(node)
        }else{
            # 递归长树
            SubNode_new <- TreeGenerate(X[SubSamplesIndex,], Y[SubSamplesIndex], AttributesSet, method = method, AttributeValue=AttributeValue,X.globel=X.globel)
            node$SubNode[[length(node$SubNode)]] <- SubNode_new
        }
        
        # 生成分支二
        SubSamplesIndex <- which(X[,PartitionAttributes] > Optimal.Partition) # 返回子集索引
        AttributeValue <- paste('> ',Optimal.Partition,sep = '')
        node <- Branch(node, X[SubSamplesIndex,], Y[SubSamplesIndex])
        if(sum(X[,PartitionAttributes] <= Optimal.Partition) == 0){
            C <- GetMostCategories(Y)
            C <- as.character(C)
            node$SubNode[[length(node$SubNode)]] <- LeafNode(node$SubNode[[length(node$SubNode)]], C,AttributeValue)
            return(node)
        }else{
            SubNode_new <- TreeGenerate(X[SubSamplesIndex,], Y[SubSamplesIndex], AttributesSet, method = method, AttributeValue=AttributeValue,X.globel=X.globel)
            node$SubNode[[length(node$SubNode)]] <- SubNode_new
        }
        return(node)
    }
}

# 针对一维变量进行预测，得到其所属类别[预测用]
xTreePredict <- function(x,node){
    # 功能：
    # 根据生成的树，对输入的样本进行预测
    
    # 输入：
    # x：一个样本对应的协变量
    # node：样本分类时每一层所属节点
    
    
    # 输出：
    # y：预测的分类结果
    
    # 处理只有根节点的情况
    if(is.null(node$SubNode)){
        y <- as.character(node$class)
        return(y)
    }
    
    
    
    # 得到当前节点的子节点个数
    n_subnode <- length(node$SubNode)
    # 找到本样本的属性取值将划分至哪个子节点
    for(i in 1:n_subnode){
        # 判断划分属性是离散属性还是连续属性
        # 连续
        if(class(x[node$PartitionAttributes]) == 'numeric'){
            # 判断子节点的属性取值是'<='还是'> '
            if(substr(node$SubNode[[i]]$AttributeValue,1,1) == '<'){
                if(x[node$PartitionAttributes] <= as.numeric(substr(node$SubNode[[i]]$AttributeValue,3,nchar(node$SubNode[[i]]$AttributeValue)))){
                    break()
                }else{
                    i <- n_subnode # 这是n_subnode一定为2，因为不是'<='就是'> '了，只有两种可能的属性取值
                    break()
                }
            }
        }else{
        # 离散
        # 找到当前节点的属性取值
        if(node$SubNode[[i]]$AttributeValue == x[node$PartitionAttributes]){break()}
        }
    }
    # 若为叶节点，则输出类别，否则，迭代到下一层
    node <- node$SubNode[[i]]
    if(!is.null(node$class)){
        return(node$class)
    }else{
        y <- xTreePredict(x,node)
    }
}

# 预测函数
TreePredict <- function(X,tree){
    # 功能：
    # 根据生成的树，对输入的样本进行预测
    
    # 输入：
    # X：待分类样本的协变量
    # tree：已经建立的树模型

    
    # 输出：
    # Y_hat：预测的分类结果
    
    # 对X进行矩阵化，防止X为一维向量时程序运行出错
    X <- as.matrix(X)
    n <- dim(X)[1]
    # 创建一个存放预测结果的容器
    Y_hat <- matrix(nrow = 1,ncol = n)
    # 利用xTreePredict函数进行预测
    for(i in 1:n){
        x <- X[i,]
        y <- xTreePredict(x,tree)
        Y_hat[1,i] <- as.character(y)
    }
    return(Y_hat)
}

# 求当前节点不分支时正确分类的样本数量[进行预剪枝用]
n_no_Branch <- function(Node,X.test,Y.test){
    # 功能：
    # 计算划分前验证集中被正确分类的样本
    
    # 输入：
    # Node：需要进行预剪枝处理的节点
    # X.test：验证集样本的协变量数据
    # Y.test：验证集样本的响应变量数据
    
    # 输出：
    # n：划分前验证集中被正确分类的个数
    
    # 对X进行矩阵化，防止X为一维向量程序报错
    C <- as.character(GetMostCategories(Node$Y))
    n <- sum(as.character(Y.test) == C)
    return(n)
}

# 当前节点分支后正确分类的数量[进行预剪枝用]
n_Branch <- function(Node,X.test,Y.test){
    # 功能：
    # 计算划分后验证集中被正确分类的样本
    
    # 输入：
    # Node：需要进行预剪枝处理的节点
    # X.test：验证集样本的协变量数据
    # Y.test：验证集样本的响应变量数据
    
    
    # 输出：
    # n：划分后验证集中被正确分类的个数
    n <- 0
    # 得到当前节点的子节点个数
    n_SubNode <- length(Node$SubNode)
    for(i in 1:n_SubNode){
        # 找到当前节点的属性取值
        C <- as.character(GetMostCategories(Node$SubNode[[i]]$Y))
        n_new <- sum(Y.test[Node$SubNode[[i]]$AttributeValue == X.test[,Node$PartitionAttributes]] == C)
        n <- n + n_new
    }
    return(n)
}

# 对树进行预剪枝
PrePruning <- function(tree,X.test,Y.test){
    # 功能：
    # 基于验证集数据对树进行预剪枝操作
    
    # 输入：
    # tree：需要进行预剪枝处理的树
    # X.test：验证集样本的协变量数据
    # Y.test：验证集样本的响应变量数据
    
    
    # 输出：
    # tree：完成预剪枝的树
    
    # 计算不分支的正确预测数量
    n_no_Branch <- n_no_Branch(tree,X.test,Y.test)
    # 计算分支后的正确预测数量
    n_Branch <- n_Branch(tree,X.test,Y.test)
    # 若不分支正确预测数量多，将该节点设置为叶节点
    if(n_no_Branch >= n_Branch){
        df <- as.data.frame(table(tree$Y)) # 将table转化为data.frame
        Class <- as.character(df[which.max(df[,2]),1])
        tree <- LeafNode(tree, Class, AttributeValue=tree$AttributeValue)
    }else{
        # 若分支后正确预测数量多，迭代到子节点
        n_SubNode <- length(tree$SubNode)
        for(i in 1:n_SubNode){
            # 若是子节点不是叶节点才进行处理，若是叶节点，不进行处理
            if(!is.null(tree$SubNode[[i]]$SubNode)){
                SubIndex <- (tree$SubNode[[i]]$AttributeValue == X.test[,tree$PartitionAttributes])
                tree$SubNode[[i]] <- PrePruning(tree$SubNode[[i]],X.test[SubIndex,],Y.test[SubIndex])
            }
        }
    }
    return(tree)
}

# 生成剪枝后的树[进行后剪枝用]
newtree <- function(tree){
    # 之所以设置本函数，是因为源于对数进行后剪枝的实现思路。
    # 后剪枝可以先对全为叶子节点的分支进行剪枝操作，然后直接利用预测函数预测精度即可
    # 根据剪枝前后两棵树的效果，修改下一次迭代时使用的是剪枝后的树还未剪枝的树。
    
    # 功能：
    # 将输入的树中全为叶节点的分支进行剪枝处理，本函数只进行一次剪枝处理
    
    # 输入：
    # tree：需要进行剪枝处理的树
    
    # 输出：
    # tree：完成剪枝的树
    
    # 计算叶子节点个数S
    S <- 0
    n_SubNode <- length(tree$SubNode)
    if(!is.null(tree$SubNode)){
        # 判断叶子节点的个数
        for(i in 1:n_SubNode){
            S <- S + is.null(tree$SubNode[[i]]$SubNode)
        }
    }
   
    # 若子节点都是叶子节点，则将该节点设置为叶子节点，即进行剪枝操作
    if(S == n_SubNode){
        df <- as.data.frame(table(tree$Y)) # 将table转化为data.frame
        Class <- as.character(df[which.max(df[,2]),1])
        tree <- LeafNode(tree, Class, AttributeValue=tree$AttributeValue)
        return(tree)
    }else{
        # 若不是叶子节点，迭代到下一层
        for(j in 1:n_SubNode){
            # 判断是不是叶子节点
            if(!is.null(tree$SubNode[[j]]$SubNode)){
                tree$SubNode[[j]] <- newtree(tree$SubNode[[j]])
                return(tree)
            }
        }
    }
}

# 对树进行后剪枝操作
PostPruning <- function(tree,X.test,Y.test){
    # 功能：
    # 基于验证集数据对树进行后剪枝操作
    
    # 输入：
    # tree：需要进行后剪枝处理的树
    # X.test：验证集样本的协变量数据
    # Y.test：验证集样本的响应变量数据
    
    
    # 输出：
    # tree：完成后剪枝的树
    
    Prune.tree <- tree
    # 只要没有迭代到根节点，就一直迭代
    while(!is.null(Prune.tree$SubNode)){
        # 剪枝前树的正确预测个数
        pred.before <- sum(TreePredict(X.test,tree) == Y.test)
        # 对树进行剪枝操作
        Prune.tree <- newtree(Prune.tree)
        # 剪枝后树的正确预测个数
        pred.after <- sum(TreePredict(X.test,Prune.tree)== Y.test)
        # 若剪枝后效果好，将原来的树更新为剪枝后的树
        if(pred.after>pred.before){ #若考虑奥卡姆剃刀原则，应该将'>'改成'>='，这里为跟书中保持一直，没有更改
            tree <- Prune.tree
        }
    }
    return(tree)
}

# 返回树的深度(即最多分支层数)[绘制决策树用]
depth <- function(tree,d = 0){
    # 功能：
    # 计算输入节点的深度
    
    # 输入：
    # tree：待计算深度的节点
    # d：为方便递归设置的一个参数，使用该函数时不用设置
    
    # 输出：
    # d：节点的深度
    
    # 子节点的个数
    n_SubNode <- length(tree$SubNode)
    # 判断是不是有子节点
    if(n_SubNode != 0){
        d <- d+1 #有子节点，深度加1
        # 创建一个容器来存放不同子节点的深度，以便得到最大深度
        d.new <- matrix(ncol = n_SubNode,nrow = 1)
        for(i in 1:n_SubNode){
            # 递归调用函数，返回不同子节点的深度
            d.new[,i] <- depth(tree$SubNode[[i]],d)
        }
        # 返回最大深度
        return(max(d.new))
    }else{
        # 没有子节点，返回当前深度
        return(d)
    }
}

# 返回树的宽度(即叶子节点的个数)[绘制决策树用]
width <- function(tree,w = 0){
    # 功能：
    # 计算输入节点的宽度
    
    # 输入：
    # tree：待计算宽度的节点
    # w：为方便递归设置的一个参数，使用该函数时不用设置
    
    # 输出：
    # w：节点的宽度
    
    # 子节点的数量
    n_SubNode <- length(tree$SubNode)
    # 判断是不是有子节点
    if(n_SubNode != 0){
        # 有子节点
        for(i in 1:n_SubNode){
            # 返回子节点的深度
            w <- width(tree$SubNode[[i]],w)
        }
        return(w)
    }else{
        # 没有子节点即为叶节点，宽度加1
        w <- w+1
        return(w)
    }
}

# 绘制当前节点(绘制决策树时中间的一个过程)[绘制决策树用]
plot_tree_progress <- function(tree,root_x=1,root_y=w/2,m){
    # 功能：
    # 绘制输入的节点
    
    # 输入：
    # tree：需要绘制的节点
    # root_x：根节点的横坐标
    # root_y：根节点的纵坐标
    # m：方便递归设置的参数
    
    # 输出：
    # 完成了图像的绘制
    # m：方便递归设置的参数
    
    # 绘制根节点划分属性
    text(root_x,root_y,tree$PartitionAttributes)
    # 子节点个数
    n_SubNode <- length(tree$SubNode)
    # 有子节点，进行绘制
    if(n_SubNode!=0){
        # 逐个对子节点进行绘制
        for(i in 1:n_SubNode){
            # 返回当前子节点的宽度
            w.new <- width(tree$SubNode[[i]])
            # 子节点有划分属性，说明是中间节点，不是叶节点
            if(!is.null(tree$SubNode[[i]]$PartitionAttributes)){
                # 根节点的横纵坐标
                x <- root_x + 1
                y <- m[x-1,2] + w.new/2
                # m的设置可以解决图像挤在一块的问题
                m[x-1,2] <- m[x-1,2] + w.new
                # 绘制当前节点
                text(x,y,tree$SubNode[[i]]$PartitionAttributes,col=1)
                # 绘制箭头
                arrows(root_x,root_y,x,y,length = 0.01)
                # 添加属性取值
                text((root_x+x)/2,(root_y+y)/2,tree$SubNode[[i]]$AttributeValue,col=3)
                m <- plot_tree_progress(tree$SubNode[[i]],root_x=x,root_y=y,m)
            }else{
                # 绘制叶子节点
                # 具体过程和上面绘制中间节点类似
                x <- root_x + 1
                y <- m[x-1,2] + w.new/2
                m[x-1,2] <- m[x-1,2] + w.new
                arrows(root_x,root_y,x,y,length = 0.1)
                text(x,y,tree$SubNode[[i]]$class,col=2)
                text((root_x+x)/2,(root_y+y)/2,tree$SubNode[[i]]$AttributeValue,col=3)
            }
        }
    }
    return(m)
}

# 绘制决策树
plot.tree <- function(tree,...){
    # 功能：
    # 绘制决策树
    
    # 输入：
    # tree：决策树
    
    # 输出：
    # 完成决策树的绘制
    
    # 计算决策树的深度和宽度
    d <- depth(tree)
    w <- width(tree)
    # 生成一个空白图像，以便后续在图像上绘制决策树
    plot(0,0,xlab = '',ylab = '',xlim = c(0,d+2),ylim = c(0,w),type = 'n',axes = FALSE,...)
    # 防止图像挤在一起，设置参数m来对层中间节点及叶节点个数进行计数
    m <- matrix(0,ncol = 2,nrow = d)
    m[,1] <- 1:d
    # 绘制树
    plot_tree_progress(tree,root_x=1,root_y=w/2,m)
}

# 求候选划分点集合[处理连续数据用]
CandidatePartition <- function(x){
    # 功能：
    # 找到一组连续数据所有可能的分割点
    
    # 输入：
    # x：需要寻找分割点的数据
    
    # 输出：
    # Attr：所有可能分割点的集合
    
    # 对数据进排序
    x <- sort(x)
    # 生成一个存放分割点集合的容器
    Attr <- 1:(length(x)-1)
    # 逐个找到分割点，并放在Attr中
    for(i in 1:(length(x)-1)){
        Attr[i] <- (x[i] + x[i+1])/2
    }
    return(Attr)
}

# 找到最优划分点[处理连续数据用]
OptimalPartition <- function(X,Y){
    # 功能：
    # 找到一组连续数据的最优分割点
    
    # 输入：
    # X：连续变量数据
    # Y：分类结果
    
    # 输出：
    # Optimal.Partition：最优分割点
    
    # 计算所有可能的分割点
    Attr <- CandidatePartition(X)
    # 生成一个容器存放各个分割点对应的信息熵
    Ent_Matrix <- matrix(0,ncol = 2,nrow = length(Attr))
    Ent_Matrix[,1] <- Attr
    # 逐个对分割点计算信息熵
    for(i in 1:length(Attr)){
        # 以下就是按公式进行计算的过程
        Median <- Attr[i]
        Index <- which(X<=Median)
        Sub_Y_less <- Y[Index]
        df_less <- as.data.frame(table(Sub_Y_less))
        p_k_less <- df_less[,2]/length(Sub_Y_less)
        log_pk_less <- log2(p_k_less)
        log_pk_less[which(log_pk_less == -Inf)] <- 0
        Sub_Y_bigger <- Y[-Index]
        df_bigger <- as.data.frame(table(Sub_Y_bigger))
        p_k_bigger <- df_bigger[,2]/length(Sub_Y_bigger)
        log_pk_bigger <- log2(p_k_bigger)
        log_pk_bigger[which(log_pk_bigger == -Inf)] <- 0
        
        Ent <- (p_k_less %*% log_pk_less) *length(Sub_Y_less)/length(Y) + (p_k_bigger %*% log_pk_bigger) *length(Sub_Y_bigger)/length(Y)
        Ent_Matrix[i,2] <- -Ent
    }
    # 信息增益最大，就是子节点的信息熵最小，找到使信息熵达到最小的分割点，即为最优分割点
    Index <- which.min(Ent_Matrix[,2])
    Optimal.Partition <- Ent_Matrix[Index,]
    return(Optimal.Partition)
}

# 生成缺失值决策树
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

# 在含有缺失值的数据中找到最优划分属性
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


# 供使用者的函数主要有以下几个
# TreeGenerate：生成决策树
# TreePredict：利用决策树预测
# PrePruning：进行预剪枝
# PostPruning：进行后剪枝
# plot.tree：绘制决策树
# TreeGenerate_missing：生成有缺失值的决策树



##### 主函数代码
# 引入相关包
library(readxl)
# 设置工作路径
setwd("E:/projects/决策树/")
# 读取只有离散属性的数据
mydata <- readxl::read_excel('E:/projects/决策树/西瓜书数据1.xlsx')
mydata <- as.data.frame(mydata)
X <- mydata[,-dim(mydata)[2]]
Y <- mydata[,dim(mydata)[2]]
# 分割数据
train.index <- c(1,2,3,6,7,10,14,15,16,17)
X.train <- X[train.index,]
Y.train <- Y[train.index]
X.test <- X[-train.index,]
Y.test <- Y[-train.index]
# 给定属性集
AttributesSet <- 1:(dim(mydata)[2]-1)
# 生成《机器学习》78页图4.4决策树
tree_4.4 <- TreeGenerate(X, Y, AttributesSet,method='ID3')
plot.tree(tree_4.4,main='图4.4决策树')
# 生成《机器学习》81页图4.5决策树
tree_4.5 <- TreeGenerate(X.train, Y.train, AttributesSet,method='ID3')
plot.tree(tree_4.5,main='图4.5决策树')
# 生成《机器学习》81页图4.6决策树
tree_4.6 <- PrePruning(tree_4.5,X.test,Y.test)
plot.tree(tree_4.6,main='图4.6决策树')
# 生成《机器学习》83页图4.7决策树
tree_4.7 <- PostPruning(tree_4.5,X.test,Y.test)
plot.tree(tree_4.7,main='图4.7决策树')
# 利用树tree_4.7进行预测
Y_pred <- TreePredict(X.test,tree_4.7)

# 读取有连续属性的数据
mydata <- readxl::read_excel('E:/projects/决策树/西瓜书数据2.xlsx')
mydata <- as.data.frame(mydata)
X <- mydata[,-dim(mydata)[2]]
Y <- mydata[,dim(mydata)[2]]
# 给定属性集
AttributesSet <- 1:(dim(mydata)[2]-1)
# 生成《机器学习》85页图4.8决策树
tree_4.8 <- TreeGenerate(X, Y, AttributesSet,method='ID3')
plot.tree(tree_4.8,main='图4.8决策树')

# 读取有缺失值的数据
mydata <- readxl::read_excel('E:/projects/决策树/西瓜书数据3.xlsx')
mydata <- as.data.frame(mydata)
X <- mydata[,-dim(mydata)[2]]
Y <- mydata[,dim(mydata)[2]]
# 给定属性集
AttributesSet <- 1:(dim(mydata)[2]-1)
# 给定初始权重
w <- rep(1,each=length(Y))
# 生成《机器学习》89页图4.9决策树
tree_4.9 <- TreeGenerate_missing(X, Y, AttributesSet, AttributeValue=NULL, X.globel=X,w=w)
plot.tree(tree_4.9,main='图4.9决策树')