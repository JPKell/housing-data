df <- read.csv('data/correlation.csv')
X  <- subset(df, select = -c(SalePrice))
y  <- df$SalePrice

cor_matrix <- cor(df)

cor_target <- cor_matrix[, "SalePrice"]
sort(cor_target, decreasing = TRUE)

write.csv(cor_target, file='data/corr_result.csv')
write.csv(cor_matrix, file="data/corr_matrix.csv", )