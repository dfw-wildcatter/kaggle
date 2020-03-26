setwd("/Users/michiel/Documents/Kaggle/Titanic/")
chosen.columns <- c("survived","sex","pclass","fare","embarked","sibsp","parch","age","nfs")

train.data <- read.csv("train.csv", stringsAsFactors = F)
test.data <- read.csv("test.csv", stringsAsFactors = F)
test.data.with.prediction <- cbind(NA, test.data)
colnames(test.data.with.prediction) <- colnames(train.data)
all.data <- rbind( train.data, test.data.with.prediction)
for ( col in 1:ncol(all.data))
	all.data[,col] <- type.convert ( as.character( all.data[,col] ) )
	
get.last.names <- function ( complete.names, pclass )
{
	last.names <- substr ( complete.names, 0, regexpr(",",complete.names	)-1 )
	return ( paste(pclass,last.names,sep="_") )
}	
	
get.num.survivors.per.name <- function ( input.matrix )
{
	# assume that family look after each other - see how well the family did
	# + scores for how many survived, - scores for how many died. 0 is we can't find any
	last.names <- get.last.names ( input.matrix[,"name"], input.matrix[,"pclass"] )
	last.names.plus.survived <- cbind(last.names, input.matrix[,c("survived")])
	colnames(last.names.plus.survived) <- c("last name","survived")
	num.survivors.per.name <- lapply(split(as.data.frame(last.names.plus.survived)[,,drop=F], 
		last.names.plus.survived[,"last name"]),
		function(x){return(sum(as.numeric(as.character(x$survived)),na.rm=T))})
	return ( num.survivors.per.name )
}	
	
train.data <- all.data[1:nrow(train.data),,drop=F]
test.data <- all.data[-(1:nrow(train.data)),,drop=F]
train.survivors.per.name <- get.num.survivors.per.name ( train.data )
other.family.survived <- 
	unlist(train.survivors.per.name[match(get.last.names(train.data[,"name"], train.data[,"pclass"]),names(train.survivors.per.name))]) - 		
	train.data[,"survived"]
train.data <- cbind( train.data, other.family.survived)
colnames(train.data)[ncol(train.data)] <- "nfs"

test.data.names <- get.last.names(test.data[,"name"], test.data[,"pclass"])
test.other.family.survivors <- train.survivors.per.name[match(test.data.names,names(train.survivors.per.name))]
nfs <- rep(0,length(test.other.family.survivors))
nfs[!is.na(names(test.other.family.survivors))] <- unlist(test.other.family.survivors[!is.na(names(test.other.family.survivors))])
test.data <- cbind( test.data, nfs)
colnames(test.data)[ncol(test.data)] <- "nfs"

in.sample <- train.data[1:800,,drop=F]
out.sample <- train.data[-(1:800),,drop=F]

require(randomForest)

in.sample.rf <- randomForest(survived ~ ., data=in.sample[,chosen.columns], ntree=1000,
                              keep.forest=T, importance=T, na.action=na.exclude)
print ( in.sample.rf )
out.sample.pr<-predict(in.sample.rf,out.sample[,chosen.columns[-1]],predict.all=T)
out.sample.pr$aggregate[ is.na ( out.sample.pr$aggregate )] <- 0
test.results <- cbind(ifelse(out.sample.pr$aggregate>0.5,1,0),out.sample[,"survived"])
score <- as.numeric ( lm(test.results[,1] ~ test.results[,2] +- 1)$coefficients[1] )
print ( score )

train.rf <- randomForest(survived ~ ., data=train.data[,chosen.columns], ntree=1000,
                              keep.forest=T, importance=T, na.action=na.exclude)

test.pr<-predict(train.rf,	
	test.data[,chosen.columns[-1]],
	predict.all=T)
test.output <- test.pr$aggregate
test.output [ is.na ( test.output )] <- 0
test.output <- ifelse(test.output > 0.5 , 1, 0)

write.table(test.output, row.names=F, col.names = F, file="output.csv" )