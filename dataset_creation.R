library(arfima)
library(dplyr)
library(tseries)
library(forecast)
library(purrr)
library(tidyverse)
library(readxl)

stock_name <- readline()

setwd("/Users/uladzimircharniauski/Documents/AR_Bandits/Long-Memory-Adapted-Autoregressive-Bandits/LMA_ARB/data/")

stock <- read.csv(paste(stock_name,".csv", sep=""))

stock_data <- data.frame(
  "Close" = stock$Close,
  "Volume"= stock$Volume
)

stock_data["x"] <- c(0,diff(log(stock_data$Close))) 

stock_data <- stock_data %>% mutate(index =case_when(x > 0 ~ 1, TRUE ~ -1))

# stock_data["x"] <- abs(stock_data["x"]) 

data_log_returns <- stock_data["x"]

data_log_returns <- data_log_returns[-1,]

names(data_log_returns) <- gsub('"', '', names(data_log_returns))

setwd("/Users/uladzimircharniauski/Documents/MATH-STATS-ECON-CSE/CSE/FALL 2025/CSE 5825/MVRNN/data")

write.csv(data_log_returns, paste(toupper(stock_name),".csv",sep=""), row.names = FALSE, quote = FALSE)
