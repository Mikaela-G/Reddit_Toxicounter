# import libraries
library(tidyverse)
library(RSQLite)
library(DBI)
#####library(quanteda)
#####library(tidytext)

# import sqlite database as dataframe
con = dbConnect(SQLite(), dbname="/Users/Mikki/Documents/GitHub/Reddit_Toxicounter/database/AskReddit_Complete.db")
myQuery <- dbSendQuery(con, "SELECT * FROM AskReddit")
df <- dbFetch(myQuery, n=1000000) # n=-1 for all records

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

###################### TOXIC_LABEL BOXPLOTS ######################

# toxic_label vs...
#   comment length
#   score
# (Ordinal categorical variable vs. Continuous numerical variable)

# represent toxic_label as 0, 1, 2?

# toxic_label vs. comment length
ggplot(df, aes(x=df$toxic_label, y=df$comment_length, fill=df$toxic_label)) + geom_boxplot() + xlab("Toxic Label") + ylab("Comment Length (in characters)") + guides(fill=guide_legend(title="Toxic Label"))

# toxic_label vs. score
ggplot(df, aes(x=df$toxic_label, y=df$score, fill=df$toxic_label)) + geom_boxplot() + xlab("Toxic Label") + ylab("Score") + guides(fill=guide_legend(title="Toxic Label"))

###################### TOXIC_LABEL HISTOGRAMS ######################

### SUBSETTING TESTS
##df <- df[!(df$score>3 & df$score<8),]
##df2 <- df[!(df$score>3 & df$score<8),]
df_not_toxic <- df[(df$toxic_label=="not toxic"),]
df_moderately_toxic <- df[(df$toxic_label=="moderately toxic"),]
df_very_toxic <- df[(df$toxic_label=="very toxic"),]
##df_very_toxic <- drop_na(df_very_toxic)

# histogram 1: score vs. frequency for not toxic
ggplot(df_not_toxic, aes(x=score)) + geom_histogram(binwidth=1, col="orange", fill="orange", alpha = .4) + labs(title="Score Histogram for Not Toxic Comments", x="Score", y="Frequency") + coord_cartesian(xlim=c(-5, 20)) #+ scale_x_log10()

# histogram 2: score vs. frequency for moderately toxic
ggplot(df_moderately_toxic, aes(x=score)) + geom_histogram(binwidth=1, col="green", fill="green", alpha = .2) + labs(title="Score Histogram for Moderately Toxic Comments", x="Score", y="Frequency") + coord_cartesian(xlim=c(-5, 25))

# histogram 3: score vs. frequency for very toxic
ggplot(df_very_toxic, aes(x=score)) + geom_histogram(binwidth=1, col="blue", fill="blue", alpha = .2) + labs(title="Score Histogram for Very Toxic Comments", x="Score", y="Frequency") + coord_cartesian(xlim=c(-10, 30))