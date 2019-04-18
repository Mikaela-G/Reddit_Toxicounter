#Make histograms relative, log proportion. Combining all three graphs

# import libraries
library(tidyverse)
library(RSQLite)
library(DBI)
#####library(quanteda)
#####library(tidytext)

# import sqlite database as dataframe
con = dbConnect(SQLite(), dbname="/Users/Mikki/Documents/GitHub/Reddit_Toxicounter/database/AskReddit_Complete.db")
myQuery <- dbSendQuery(con, "SELECT * FROM AskReddit WHERE toxic_label NOT NULL AND toxic_label not like '%None%'")
df <- dbFetch(myQuery, n=-1) # n=-1 for all records

# removing all "newlinechar" and "http" from the df
df$comment <- gsub("newlinechar", " ", df$comment)
df$comment <- gsub("http", " ", df$comment)

# adding comment_length (by number of characters) to df
df$comment_length <- nchar(as.character(df$comment))

# reordering toxic label factor levels (so it goes: not toxic, moderately toxic, very toxic)
df$toxic_label <- as.factor(df$toxic_label)
df$toxic_label = factor(df$toxic_label,levels(df$toxic_label)[c(2,1,3)])
levels(df$toxic_label)

#rm(list=ls())
#dbClearResult(myQuery)

###################### TOXIC_LABEL HISTOGRAMS ######################
########################################

# FINAL VERSION OF combined score histogram
ggplot(df, aes(x=score, fill=toxic_label)) + geom_histogram(binwidth=1, alpha=0.2, color="black") + scale_fill_manual(name="group",values=c("blue","red","green"),labels=c("not toxic","moderately toxic","very toxic")) + coord_cartesian(xlim=c(-5, 15)) + theme_bw(base_size=30) #+ theme(legend.text=element_text(size=20))

########################################
# toxic_label vs...
#   comment length
#   score
# (Ordinal categorical variable vs. Continuous numerical variable)

### SUBSETTING TESTS
##df <- df[!(df$score>3 & df$score<8),]
##df2 <- df[!(df$score>3 & df$score<8),]
df_not_toxic <- df[(df$toxic_label=="not toxic"),]
df_moderately_toxic <- df[(df$toxic_label=="moderately toxic"),]
df_very_toxic <- df[(df$toxic_label=="very toxic"),]
##df_very_toxic <- drop_na(df_very_toxic)

# combined comment length histogram
ggplot(df, aes(x=comment_length)) + 
  geom_histogram(binwidth=1, data = df_not_toxic, fill = "red", alpha = 0.2) + 
  geom_histogram(binwidth=1, data = df_moderately_toxic, fill = "blue", alpha = 0.2) +
  geom_histogram(binwidth=1, data = df_very_toxic, fill = "green", alpha = 0.5) + coord_cartesian(xlim=c(0, 1000)) #+ scale_x_log10()

# combined and zoomed in comment length histogram
ggplot(df, aes(x=comment_length)) + 
  geom_histogram(binwidth=1, data = df_not_toxic, fill = "red", alpha = 0.2) + 
  geom_histogram(binwidth=1, data = df_moderately_toxic, fill = "blue", alpha = 0.2) +
  geom_histogram(binwidth=1, data = df_very_toxic, fill = "green", alpha = 0.5) + coord_cartesian(xlim=c(0, 1000), ylim=c(0,200)) #+ scale_x_log10()

# combined score histogram
ggplot(df, aes(x=score)) + 
  geom_histogram(binwidth=1, data = df_not_toxic, color = "black", fill = "red", alpha = 0.2) + 
  geom_histogram(binwidth=1, data = df_moderately_toxic, color = "black", fill = "blue", alpha = 0.2) +
  geom_histogram(binwidth=1, data = df_very_toxic, color = "black", fill = "green", alpha = 0.5) + coord_cartesian(xlim=c(-5, 15)) #+ scale_x_log10()

# combined and zoomed in score histogram
ggplot(df, aes(x=score)) + 
  geom_histogram(binwidth=1, data = df_not_toxic, color = "black", fill = "red", alpha = 0.2) + 
  geom_histogram(binwidth=1, data = df_moderately_toxic, color = "black", fill = "blue", alpha = 0.2) +
  geom_histogram(binwidth=1, data = df_very_toxic, color = "black", fill = "green", alpha = 0.5) + coord_cartesian(xlim=c(-5, 15), ylim=c(0,5000)) #+ scale_x_log10()