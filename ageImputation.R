df <- read.csv('titanic.csv')
library(mice)
imputed_df <- mice(df[, !names(df) %in% c('name','ticket','cabin','home_dest')], method='rf')
final_df <- complete(imputed_df)
par(mfrow=c(1,2))
hist(df$age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(final_df$age, freq=F, main='Age: Data after impute',
     col='lightgreen', ylim=c(0,0.04))
df$age <- final_df$age
show(df)
write.csv(file='titanic_age.csv', x=df)