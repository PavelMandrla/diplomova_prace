library(vioplot)
library(moments)

counts5_1 <- as.data.frame(read.csv(file = './statistics/data/counts5_1.csv'))
colnames(counts5_1) = c("truth", "pred")

counts5_3 <- as.data.frame(read.csv(file = './statistics/data/counts5_3.csv'))
colnames(counts5_1) = c("truth", "pred")

data = as.data.frame(cbind(counts5_1$truth, counts5_1$pred, counts5_3$pred))
colnames(data) = c("truth", "pred_5_1", "pred_5_3")

data$err_5_1 = data$truth - data$pred_5_1
data$err_5_3 = data$truth - data$pred_5_3

vioplot(cbind(data$err_5_1, data$err_5_3))
#boxplot(cbind(x1, x2))

qqnorm(data$err_5_1)
qqline(data$err_5_1)
hist(data$err_5_1)

print("... length 5, stride 1 ...")
sprintf("   skewness: %f", skewness(data$err_5_1))
sprintf("   kurtosis: %f", moments::kurtosis(data$err_5_1)-3)
sprintf("   mean: %f", mean(data$err_5_1))

shapiro.test(data$err_5_1)

