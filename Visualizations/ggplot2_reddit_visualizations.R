# import libraries
library(tidyverse)
library(RSQLite)
library(DBI)
library(quanteda)
library(tidytext)

# import sqlite database as dataframe
con = dbConnect(SQLite(), dbname="/Users/Mikki/Documents/GitHub/Reddit_Toxicounter/AskRedditScrape 2016-Recent/database/AskReddit_2016 - Recent.db")
myQuery <- dbSendQuery(con, "SELECT * FROM AskReddit")
df <- dbFetch(myQuery, n=10000) # n=-1 for all records

# removing all "newlinechar" from the df
df$comment <- gsub("newlinechar", " ", df$comment)

###################### WORD CLOUDS ######################

# creating word cloud for all the comments in the df
df.corpus <- corpus(df, text_field = "comment")
#summary(df.corpus, n=3)
df.dfm <- dfm(df.corpus, tolower=TRUE, remove_punct=TRUE, remove_numbers=TRUE, remove=stopwords(source="smart"), groups = "toxic_label")
df.cloud <- textplot_wordcloud(df.dfm)

# creating subsets of corpus based on toxicity label
not.toxic <- corpus_subset(df.corpus, toxic_label=="not toxic")
mod.toxic <- corpus_subset(df.corpus, toxic_label=="moderately toxic")
very.toxic <- corpus_subset(df.corpus, toxic_label=="very toxic")

# creating word cloud for all "not toxic" comments
not.toxic.dfm <- dfm(not.toxic, tolower=TRUE, remove_punct=TRUE, remove_numbers=TRUE, remove=stopwords(source="smart"))
not.toxic.cloud <- textplot_wordcloud(not.toxic.dfm)

# creating word cloud for all "moderately toxic" comments
mod.toxic.dfm <- dfm(mod.toxic, tolower=TRUE, remove_punct=TRUE, remove_numbers=TRUE, remove=stopwords(source="smart"))
mod.toxic.cloud <- textplot_wordcloud(mod.toxic.dfm)

# creating word cloud for all "very toxic" comments
very.toxic.dfm <- dfm(very.toxic, tolower=TRUE, remove_punct=TRUE, remove_numbers=TRUE, remove=stopwords(source="smart"))
very.toxic.cloud <- textplot_wordcloud(very.toxic.dfm)

#rm(list=ls())
#dbClearResult(myQuery)

###################### TOXICITY BAR PLOT ######################

# bar plot for toxic_label frequency
ggplot(df, aes(x=df$toxic_label)) + geom_bar(fill="red", color="darkred", width=0.9, alpha = .2) + xlab("Toxic Label") + ylab("Frequency") + theme(axis.text.x=element_text(angle=20,hjust=1))

###################### TOXICITY OVER TIME ######################

###################### TOXICITY SCATTERPLOTS ######################

# possible variables to explore: comment length, score, toxicity, severe toxicity

# toxicity vs. score
ggplot(df) + geom_point(aes(x=df$toxicity,y=df$score), col="blue", alpha=.4) + xlab("Toxicity") + ylab("Score")

# severe toxicity vs. score
ggplot(df) + geom_point(aes(x=df$severe_toxic,y=df$score), col="orange", alpha=.4) + xlab("Severe Toxicity") + ylab("Score")

# toxicity vs. severe toxicity vs. score
ggplot(df) + geom_point(aes(x=df$toxicity,y=df$severe_toxic,color=df$score), alpha=.4) + xlab("Toxicity") + ylab("Severe Toxicity")