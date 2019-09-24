# library(foreign)
rm(list = ls())
library(plm)
library(xlsx)
lag = c('1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-12', '1-13', '1-14', '1-15', '1-16', '1-17', '1-18', '1-19', '1-20', '1-21', '1-22', '1-23', '1-24', '1-25', '1-26', '1-27', '1-28', '1-29', '1-30', '1-31', '1-32', '1-33', '1-34', '1-35', '1-36', '1-37', '1-38', '1-39', '1-40', '1-41', '1-42', '1-43', '1-44', '1-45', '1-46', '1-47', '1-48', '1-49', '1-50', '1-51', '1-52', '1-53', '1-54', '1-55', '1-56', '1-57', '1-58', '1-59', '1-60')
lag_1 = c('1-1')

coefficients_estimate <- vector(mode = "numeric",60)
coefficients_t_statistic <- vector(mode = "numeric",60)
coefficients_p_value <- vector(mode = "numeric",60)

for (i in 1:60) {
  panel = read.xlsx(paste('./Smart_Beta/data/',lag[i],'.xlsx',sep=""), 'Sheet1', header=TRUE, encoding = "UTF-8")
  # print(panel)
  fixed <- plm(y ~ x, data=panel, index=c("index", "month"), model="within")
  random <- plm(y ~ x, data=panel, index=c("index", "month"), model="random")
  hausman_test <- phtest(fixed, random)
  # print(hausman_test$p.value)
  if (hausman_test$p.value<0.05){
    temp_result <- fixed
  } else{
    temp_result <- random
  }
  print(i)
  summary_temp_result <- summary(temp_result)
  print(summary_temp_result$coefficients)
  print(summary_temp_result$coefficients['x',1])
  coefficients_estimate[i] <- summary_temp_result$coefficients['x',1]
  print(summary_temp_result$coefficients['x',3])
  coefficients_t_statistic[i] <- summary_temp_result$coefficients['x',3]
  print(summary_temp_result$coefficients['x',4])
  coefficients_p_value[i] <- summary_temp_result$coefficients['x',4]
  }

write.xlsx(coefficients_estimate, file='./Smart_Beta/data/coefficients_estimate.xlsx')
write.xlsx(coefficients_t_statistic, file='./Smart_Beta/data/coefficients_t_statistic.xlsx')
write.xlsx(coefficients_p_value, file="./Smart_Beta/data/coefficients_p_value.xlsx")
# coplot(y ~ month|index, type="l", data=panel) # Lines
# coplot(y ~ month|index, type="b", data=panel) # Points and lines

# library(car)
# scatterplot(y~month|index, boxplots=FALSE, smooth=TRUE, reg.line=FALSE, data=panel)

# library(gplots)
# plotmeans(y ~ index, main="Heterogeineity across indexes", data=panel)
# plotmeans(y ~ month, main="Heterogeineity across years", data=panel)

# # OLS
# ols = lm(y ~ x, data=panel)
# # summary(ols)
# 
# # ¹Ì¶¨Ð§Ó¦
# fixed.dum <-lm(y ~ x + factor(index) - 1, data=panel)
# # summary(fixed.dum)
# 
# library(plm)
# fixed <- plm(y ~ x, data=panel, index=c("index", "month"), model="within")
# # summary(fixed)
# random <- plm(y ~ x, data=panel, index=c("index", "month"), model="random")
# print(summary(random))
# 
# print(phtest(fixed, random))
