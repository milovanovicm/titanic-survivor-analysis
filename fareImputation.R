df <- read.csv('titanic_embarked.csv')
ggplot(df[df$pclass == '3' & df$embarked == 'S', ], 
       aes(x = fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous()
median(df[df$pclass=='3' & df$embarked=='S',]$fare, na.rm=T)
df$fare[1226] <- median(df[df$pclass == '3' & df$embarked == 'S', ]$fare, na.rm = TRUE)
show(df)
write.csv(file = 'titanic_final.csv', x = df)