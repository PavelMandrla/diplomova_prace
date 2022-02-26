library(vioplot)
library(moments)
library(dplyr)
library(lawstat)
library(Metrics)

load_errs <- function (files, omit_outliers) {
  data = data.frame()
  names = c()

  print("NUMBER OF OUTLIERS", quote=FALSE)
  for (file in files) {
    file_data <- as.data.frame(read.csv(file=file[1]))
    colnames(file_data) = c("truth", "pred")
    file_data$err = file_data$truth - file_data$pred


    tmp = boxplot(file_data$err)
    print(sprintf('   %s: %d', file[2], length(tmp$out)), quote=FALSE)
    if (omit_outliers) {
      file_data$err[file_data$err %in% tmp$out] = NA
    }


    if (nrow(data) == 0) {
      data = cbind(file_data$err)
    } else {
      data = cbind(data, file_data$err)
    }
    names = c(names, file[2])
  }
  data = as.data.frame(data)
  colnames(data) = names
  data.s = stack(data)
  if (omit_outliers){
    data.s = na.omit(data.s)
  }
  result = data.s
}

load_data <- function (files) {
  data = data.frame()
  names = c("truth")

  for (file in files) {
    file_data <- as.data.frame(read.csv(file=file[1]))
    colnames(file_data) = c("truth", "pred")

    if (nrow(data) == 0) {
      data = cbind(file_data$truth, file_data$pred)
    } else {
      data = cbind(data, file_data$pred)
    }
    names = c(names, file[2])
  }
  data = as.data.frame(data)
  print(names)
  colnames(data) = names
  #data.s = stack(data)
  #result = data.s
  result = data
}

files = list(
  c('./statistics/data_new/counts1_1.csv', '1_1'),
  # c('./statistics/data_new/counts2_1.csv', '2_1'),
  c('./statistics/data_new/counts5_1.csv', '5_1'),
  c('./statistics/data_new/counts5_3.csv', '5_3')
)


data = load_data(files)
print("MAE")
Metrics::mae(actual = data$truth, predicted = data$`5_3`)
Metrics::mae(actual = data$truth, predicted = data$`5_1`)
Metrics::mae(actual = data$truth, predicted = data$`1_1`)
# Metrics::mae(actual = data$truth, predicted = data$`2_1`)

print("RMSE")
Metrics::rmse(actual = data$truth, predicted = data$`5_3`)
Metrics::rmse(actual = data$truth, predicted = data$`5_1`)
Metrics::rmse(actual = data$truth, predicted = data$`1_1`)
# Metrics::rmse(actual = data$truth, predicted = data$`1_1`)

print("MAPE")
Metrics::mape(actual = data$truth, predicted = data$`5_3`)
Metrics::mape(actual = data$truth, predicted = data$`5_1`)
Metrics::mape(actual = data$truth, predicted = data$`1_1`)
# Metrics::mape(actual = data$truth, predicted = data$`2_1`)





data = load_errs(files, TRUE)


vioplot(data$values~data$ind)

# OVĚŘENÍ NORMALITY
print('NORMALITA:', quote=FALSE)
data %>% group_by(ind) %>% summarise(norm.pval = shapiro.test(values)$p.value)

# POKUD ZAMÍTÁM NORMALITU
#   TEST SYMETRIE
print('SYMETRIE:', quote=FALSE)
data %>% group_by(ind) %>% summarise(test.pval = lawstat::symmetry.test(values, boot=FALSE)$p.value)


a = data %>% filter(ind == "5_1") %>% mutate(rounded = round(values, 0))
print(shapiro.test(a$values))
hist(a$rounded, breaks = 64)
print(shapiro.test(a$rounded))

qqnorm(a$rounded)
qqline(a$rounded, col="red")

t.test(a$values)
#qqnorm(a$values)
#qqline(a$values)









# POKUD NEZAMÍTÁM NORMALITU
# # OVĚŘENÍ SHODY ROZPTYLŮ
# # výběry jsou z normálního rozdělení -> Bartlettův test
# bartlett.test(data$values~data$ind)
#
# # ANOVA - porovnávám střední hodnoty výběrů
# results = aov(data$values~data$ind)
# summary(results)
#
# #POST HOC analýza - pro ANOVA Tukey HSD
# TukeyHSD(results)