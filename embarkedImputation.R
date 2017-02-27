df <- read.csv('titanic_age.csv')
library(ggplot2)
ggplot(df, aes(x = embarked, y = fare, fill = factor(pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous()

df$embarked[c(169, 285)] <- 'C'
show(df)
write.csv(file='titanic_embarked.csv', x=df)