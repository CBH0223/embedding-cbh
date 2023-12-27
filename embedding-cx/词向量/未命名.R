setwd('~/近期工作/Embedding算法/20230325DrXgxjIk/词向量')
library(corrplot)
library(readxl)
data <- read_excel("matrix_data.xlsx", sheet = "Sheet1")
data <- as.matrix(data)
# 最小-最大缩放
scaled_df <- apply(data, 2, function(x) (x - min(x)) / (max(x) - min(x)))
# 线性转换到[-1, 1]
scaled_df <- 2 * scaled_df - 1
#corrplot(data,method = "number",type="full",col="black")

corrplot(scaled_df,method = "color",type="full")


