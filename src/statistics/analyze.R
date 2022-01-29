library(vioplot)
library(moments)
library(dplyr)

counts5_1 <- as.data.frame(read.csv(file = './statistics/data/counts5_1.csv'))
colnames(counts5_1) = c("truth", "pred")
counts5_1$err = counts5_1$truth - counts5_1$pred
tmp = boxplot(counts5_1$err)
counts5_1$err[counts5_1$err %in% tmp$out] = NA


counts5_3 <- as.data.frame(read.csv(file = './statistics/data/counts5_3.csv'))
colnames(counts5_3) = c("truth", "pred")
counts5_3$err = counts5_3$truth - counts5_3$pred
tmp = boxplot(counts5_3$err)
counts5_3$err[counts5_3$err %in% tmp$out] = NA

counts1_1 <- as.data.frame(read.csv(file = './statistics/data/counts1_1.csv'))
colnames(counts1_1) = c("truth", "pred")
counts1_1$err = counts1_1$truth - counts1_1$pred
tmp = boxplot(counts1_1$err)
counts1_1$err[counts1_1$err %in% tmp$out] = NA

data = as.data.frame(cbind(counts5_1$err, counts5_3$err, counts1_1$err))
data_err.s = stack(data)
data_err.s = na.omit(data_err.s)
vioplot(data_err.s$values~data_err.s$ind)



#
# data = as.data.frame(cbind(counts5_1$truth, counts5_1$pred, counts5_3$pred, counts1_1$pred))
# colnames(data) = c("truth", "pred_5_1", "pred_5_3", "pred_1_1")
#
# data$err_5_1 = data$truth - data$pred_5_1
# data$err_5_3 = data$truth - data$pred_5_3
# data$err_1_1 = data$truth - data$pred_1_1
#
# data_err = as.data.frame(cbind(data$err_5_1, data$err_5_3, data$err_1_1))
# data_err.s = stack(data_err)
#
# vioplot(data_err.s$values~data_err.s$ind)
# # boxplot(data_err.s$values~data_err.s$ind)
#
# pro výběry ověříme normalitu
data_err.s %>% group_by(ind) %>% summarise(norm.pval = shapiro.test(values)$p.value)
#
# # homeoskedasticita - test pro ověření shody rozptylů
# # výběry jsou z normálního rozdělení -> Bartlettův test

bartlett.test(data_err.s$values ~ data_err.s$ind)


# ANOVA - porovnávám střední hodnoty výběrů
results = aov(data_err.s$values ~ data_err.s$ind)
summary(results)

#POST HOC analýza - pro ANOVA Tukey HSD
TukeyHSD(results)
