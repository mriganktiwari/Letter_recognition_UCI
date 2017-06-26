## Jacob Maibach, 2017
## Clustering analysis for letter recognition

library(nnet)

letter_rec <- read.csv("~/Documents/School/Spring 2017/Machine Learning I/Project/Data/letter_recognition_raw.txt", header = FALSE)
X = letter_rec[,2:length(letter_rec)]
cutoff_full = read.csv("~/Documents/School/Spring 2017/Machine Learning I/Project/Data/cutoff_output56.csv")

km = kmeans(X,centers = 26)
km2 = kmeans(X,centers = 260)
# km3 = kmeans(X,centers = 520)

kmeans_ratio <- function(km){
  between_SS = km$betweenss
  total_SS = km$tot.withinss + km$betweenss
  return(between_SS / total_SS)
}

argmin_vector <- function(v){
  min_val = v[1]
  min_ind = 1
  for(i in 2:length(v)){
    if(v[i] < min_val){
      min_val = v[i]
      min_ind = i
    }
  }
  return(min_ind)
}

kmeans_prediction <- function(km,x){
  len_x = length(x)
  d_cluster = c()
  for(i in 1:nrow(km$centers)){
    centroid = km$centers[i,]
    d = sum((centroid - x)^2) # not dividing for efficiency
    d_cluster = c(d_cluster,d)
  }
  min_ind = argmin_vector(d_cluster)
  return(min_ind)
}

kmeans_prediction_vector <- function(km,X){
  km.pred = c()
  for(i in 1:nrow(X)){
    x = X[i,]
    km.pred[i] = kmeans_prediction(km,x)
  }
  return(km.pred)
}

letter_comp <- function(letter){
  data = subset(letter_rec, V1 == letter)[,2:length(letter_rec)]
  pred = c()
  for(i in 1:length(data)){
    x = data[i,]
    pred = c(pred,kmeans_prediction(km,x))
  }
  return(pred)
}

### cutoff clustering

square_difference_dist <- function(x,y){
  return(sum((x - y)^2))
}

cutoff_clustering <- function(data,cutoff_val,dist_func = square_difference_dist){
  length_data = nrow(data)
  cluster_label = 0*vector(length = length_data)
  cluster_label[1] = 1
  max_label = 1
  for(i in 1:(length_data-1)){
    if(cluster_label[i] == 0){
      max_label = max_label + 1
      cluster_label[i] = max_label
    }
    for(j in (i+1):length_data){
      if(cluster_label[j] == 0){
        dist_val = dist_func(data[i,],data[j,])
        if(dist_val <= cutoff_val){
          cluster_label[j] = cluster_label[i]
        }
      }
    }
  }
  if(cluster_label[length_data] == 0){
    cluster_label[length_data] = max_label + 1
  }
  return(cluster_label)
}

## kmeans labeling
km.pred = kmeans_prediction_vector(km,X)
cluster.df = data.frame(km.pred = km.pred,cutoff.pred = cutoff_full$c,target = letter_rec$V1)
write.csv(cluster.df,"~/Documents/School/Spring 2017/Machine Learning I/Project/Data/cluster1.csv")

## example for presentation
pos = c(1,2,3,4,5,6,25,26,30,31,35,36,40,41)
plot(data.frame(x = pos, y = vector(length = length(pos))),ylab = "",xlab = "Position",main = "Cutoff Clustering Example Data")

## logistic
log_model_full = multinom(V1~., data = letter_rec)
full_pred = data.frame(pred = predict(log_model_full,letter_rec), target = letter_rec$V1)

cluster.df$km.pred = as.factor(cluster.df$km.pred)
cluster.df$cutoff.pred = as.factor(cluster.df$cutoff.pred)
log_model_ens = multinom(target ~ cutoff.pred + km.pred, data = cluster.df, MaxNWts = 2000)

ens_pred = data.frame(pred = predict(log_model_ens,cluster.df), target = cluster.df$target)

## confusion calculations

unique_letters = unique(letter_rec$V1) # be careful for other data

get_confusion_table <- function(pred.df){
  values = unique_letters
  table = matrix(vector(length = length(values)^2), ncol = length(values))
  for(i in 1:length(values)){
    for(j in 1:length(values)){
      table[i,j] = nrow(subset(pred.df, pred == values[i] & target == values[j]))
    }
  }
  colnames(table) = values
  rownames(table) = values
  return(table)
}

get_confusion_rate <- function(conf_table){
  values = unique_letters
  rate_table = matrix(vector(length = length(values)^2), ncol = length(values))
  colnames(rate_table) = values
  rownames(rate_table) = values
  for(i in 1:length(values)){
    for(j in 1:length(values)){
      tot_i = conf_table[i,i] + conf_table[i,j]
      tot_j = conf_table[j,i] + conf_table[j,j]
      a = conf_table[i,i]/tot_i
      b = conf_table[i,j]/tot_i
      c = conf_table[j,i]/tot_j
      d = conf_table[j,j]/tot_j
      rate_table[i,j] = a*d - b*c
    }
  }
  return(rate_table)
}

table_min <- function(table){
  names = unique_letters
  max_i = 0
  max_j = 0
  min_val = Inf
  for(i in 1:(length(names)-1)){
    for(j in (i+1):length(names)){
      val = table[i,j]
      if(val < min_val){
        max_i = i
        max_j = j
        min_val = val
      }
    }
  }
  return(c(max_i,max_j,min_val))
}

table_avg <- function(table){
  names = unique_letters
  tot = 0
  count = 0
  for(i in 1:(length(names)-1)){
    for(j in (i+1):length(names)){
      tot = tot + table[i,j]
      count = count + 1
    }
  }
  return(tot/count)
}

# models
full_conf_table = get_confusion_table(full_pred)
full_rate_table = get_confusion_rates(full_conf_table)
full_min_conf = table_min(full_rate_table)
full_avg_conf = table_avg(full_rate_table)

ens_conf_table = get_confusion_table(ens_pred)
ens_rate_table = get_confusion_rate(ens_conf_table)
ens_min_conf = table_min(ens_rate_table)
ens_avg_conf = table_avg(ens_rate_table)
