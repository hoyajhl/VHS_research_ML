## plot ROC curve/ comparison for Figure
### Data security issures, all values were omitted
# rf_all_ans3<-C(0,1,1,1,0,~)
# rf_all_pet3<-C(0.11548556, 0.80839895, 0.97375328, 0.91076115, 0.21259843,~)

roc.pet_all_rf<-roc(rf_all_ans3,rf_all_pet3,smooth=FALSE)
roc.pet_all_rf
plot.roc(roc.pet_all_rf,col="blue",add=TRUE) ##Plot AUC curve

##color version
plot(roc.pet30_rf,col="blue",add=TRUE) ## add=TRUE option indicating overlapping graphs
plot(roc.pet20_rf,col="black",add=TRUE)
plot(roc.pet10_rf,col="green",add=TRUE)

##Balck-white version
plot.new()
plot(roc.pet30_rf,lty=2,add=TRUE)
plot(roc.pet20_rf,lty=3,add=TRUE)
plot(roc.pet10_rf,lty=4,add=TRUE)

## Delong test with p-value
library(pROC)
roc.test(roc.pet30_rf, roc.pet20_rf, reuse.auc=TRUE, method="delong", na.rm=TRUE)
roc.test(roc.pet30_rf, roc.pet10_rf, reuse.auc=TRUE, method="delong", na.rm=TRUE)
roc.test(roc.pet20_rf, roc.pet10_rf, reuse.auc=TRUE, method="delong", na.rm=TRUE)
