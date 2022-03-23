
load_data <- function() {
  file_data <- as.data.frame(read.csv(file="results_old_pets_txt.csv"))
  colnames(file_data) = c("len", "stride", "MAE", "RMSE", "MAPE")
  result = file_data
}

# Library
library(ggplot2)
library(hrbrthemes)
library(dplyr)
library(viridis)

my_data = load_data()

my_data$stride = as.character(factor(my_data$stride))
my_data$len = as.character(factor(my_data$len))


ggplot(my_data, aes(len, stride, fill=RMSE)) +
  geom_tile() +
  scale_fill_viridis(discrete=FALSE) +
  theme_ipsum() +
  theme(legend.key.size = unit(3, 'cm')) +
  theme(legend.title = element_text(size=45)) +
  theme(legend.text = element_text(hjust=1, size=35)) +
  theme(axis.title.x = element_text(size=45, vjust=1, hjust=0.5)) +
  theme(axis.title.y = element_text(size=45, vjust=1, hjust=0.5)) +
  theme(axis.text.x = element_text(hjust=1, size=35)) +
  theme(axis.text.y = element_text(hjust=1, size=35)) +
  labs(x = "délka sekvence", y = "krok") +
  scale_x_discrete(position = "bottom")


