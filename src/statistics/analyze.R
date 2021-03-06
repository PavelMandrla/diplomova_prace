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
    colnames(file_data) = c("truth", "pred", "time")

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

# files = list(
#   c('./statistics/data_new/counts1_1.csv', '1_1'),
#   # c('./statistics/data_new/counts2_1.csv', '2_1'),
#   c('./statistics/data_new/counts5_1.csv', '5_1'),
#   c('./statistics/data_new/counts5_3.csv', '5_3')
# )

files = list(
  # FDST
  # c('./statistics/data_new/len1_stride1_fdst.csv', '1_1'),
  # c('./statistics/data_new/len2_stride1_fdst.csv', '2_1'),
  # c('./statistics/data_new/len2_stride3_fdst.csv', '2_3'),
  # c('./statistics/data_new/len2_stride5_fdst.csv', '2_5'),
  # c('./statistics/data_new/len3_stride1_fdst.csv', '3_1'),
  # c('./statistics/data_new/len3_stride3_fdst.csv', '3_3'),
  # c('./statistics/data_new/len3_stride5_fdst.csv', '3_5'),
  # c('./statistics/data_new/len5_stride1_fdst.csv', '5_1'),
  # c('./statistics/data_new/len5_stride3_fdst.csv', '5_3'),
  # c('./statistics/data_new/len5_stride5_fdst.csv', '5_5')

  #PETS
  c('./statistics/data_new/pets/len1_stride1_pets.csv', '1_1'),
  c('./statistics/data_new/pets/len2_stride1_pets.csv', '2_1'),
  c('./statistics/data_new/pets/len2_stride3_pets.csv', '2_3'),
  c('./statistics/data_new/pets/len2_stride5_pets.csv', '2_5'),
  c('./statistics/data_new/pets/len3_stride1_pets.csv', '3_1'),
  c('./statistics/data_new/pets/len3_stride3_pets.csv', '3_3'),
  c('./statistics/data_new/pets/len3_stride5_pets.csv', '3_5'),
  c('./statistics/data_new/pets/len5_stride1_pets.csv', '5_1'),
  c('./statistics/data_new/pets/len5_stride3_pets.csv', '5_3'),
  c('./statistics/data_new/pets/len5_stride5_pets.csv', '5_5')

  # c('./statistics/data_new/new_len1_stride1_fdst.csv', 'new_1_1'),
  # c('./statistics/data_new/new_len3_stride3_fdst.csv', 'new_3_3'),
  # c('./statistics/data_new/new_len5_stride3_fdst.csv', 'new_5_3'),
  # c('./statistics/data_new/old_len1_stride3_fdst.csv', 'old_1_1'),
  # c('./statistics/data_new/old_len3_stride3_fdst.csv', 'old_3_3'),
  # c('./statistics/data_new/old_len5_stride3_fdst.csv', 'old_5_3')
)



data = load_data(files)
print("MAE")
Metrics::mae(actual = data$truth, predicted = data$`1_1`)
Metrics::mae(actual = data$truth, predicted = data$`2_1`)
Metrics::mae(actual = data$truth, predicted = data$`2_3`)
Metrics::mae(actual = data$truth, predicted = data$`2_5`)
Metrics::mae(actual = data$truth, predicted = data$`3_1`)
Metrics::mae(actual = data$truth, predicted = data$`3_3`)
Metrics::mae(actual = data$truth, predicted = data$`3_5`)
Metrics::mae(actual = data$truth, predicted = data$`5_1`)
Metrics::mae(actual = data$truth, predicted = data$`5_3`)
Metrics::mae(actual = data$truth, predicted = data$`5_5`)

# Metrics::mae(actual = data$truth, predicted = data$`new_1_1`)
# Metrics::mae(actual = data$truth, predicted = data$`new_3_3`)
# Metrics::mae(actual = data$truth, predicted = data$`new_5_3`)
# Metrics::mae(actual = data$truth, predicted = data$`old_1_1`)
# Metrics::mae(actual = data$truth, predicted = data$`old_3_3`)
# Metrics::mae(actual = data$truth, predicted = data$`old_5_3`)

print("RMSE")
Metrics::rmse(actual = data$truth, predicted = data$`1_1`)
Metrics::rmse(actual = data$truth, predicted = data$`2_1`)
Metrics::rmse(actual = data$truth, predicted = data$`2_3`)
Metrics::rmse(actual = data$truth, predicted = data$`2_5`)
Metrics::rmse(actual = data$truth, predicted = data$`3_1`)
Metrics::rmse(actual = data$truth, predicted = data$`3_3`)
Metrics::rmse(actual = data$truth, predicted = data$`3_5`)
Metrics::rmse(actual = data$truth, predicted = data$`5_1`)
Metrics::rmse(actual = data$truth, predicted = data$`5_3`)
Metrics::rmse(actual = data$truth, predicted = data$`5_5`)

# Metrics::rmse(actual = data$truth, predicted = data$`new_1_1`)
# Metrics::rmse(actual = data$truth, predicted = data$`new_3_3`)
# Metrics::rmse(actual = data$truth, predicted = data$`new_5_3`)
# Metrics::rmse(actual = data$truth, predicted = data$`old_1_1`)
# Metrics::rmse(actual = data$truth, predicted = data$`old_3_3`)
# Metrics::rmse(actual = data$truth, predicted = data$`old_5_3`)

print("MAPE")
Metrics::mape(actual = data$truth, predicted = data$`1_1`)
Metrics::mape(actual = data$truth, predicted = data$`2_1`)
Metrics::mape(actual = data$truth, predicted = data$`2_3`)
Metrics::mape(actual = data$truth, predicted = data$`2_5`)
Metrics::mape(actual = data$truth, predicted = data$`3_1`)
Metrics::mape(actual = data$truth, predicted = data$`3_3`)
Metrics::mape(actual = data$truth, predicted = data$`3_5`)
Metrics::mape(actual = data$truth, predicted = data$`5_1`)
Metrics::mape(actual = data$truth, predicted = data$`5_3`)
Metrics::mape(actual = data$truth, predicted = data$`5_5`)

# Metrics::mape(actual = data$truth, predicted = data$`new_1_1`)
# Metrics::mape(actual = data$truth, predicted = data$`new_3_3`)
# Metrics::mape(actual = data$truth, predicted = data$`new_5_3`)
# Metrics::mape(actual = data$truth, predicted = data$`old_1_1`)
# Metrics::mape(actual = data$truth, predicted = data$`old_3_3`)
# Metrics::mape(actual = data$truth, predicted = data$`old_5_3`)





# data = load_errs(files, TRUE)
# #
# #
# vioplot(data$values~data$ind)
#
# # OVĚŘENÍ NORMALITY
# print('NORMALITA:', quote=FALSE)
# data %>% group_by(ind) %>% summarise(norm.pval = shapiro.test(values)$p.value)
#
# # POKUD ZAMÍTÁM NORMALITU
# #   TEST SYMETRIE
# print('SYMETRIE:', quote=FALSE)
# data %>% group_by(ind) %>% summarise(test.pval = lawstat::symmetry.test(values, boot=FALSE)$p.value)
#
#
# a = data %>% filter(ind == "5_1") %>% mutate(rounded = round(values, 0))
# print(shapiro.test(a$values))
# hist(a$rounded, breaks = 64)
# print(shapiro.test(a$rounded))
#
# qqnorm(a$rounded)
# qqline(a$rounded, col="red")
#
# t.test(a$values)
# #qqnorm(a$values)
# #qqline(a$values)









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