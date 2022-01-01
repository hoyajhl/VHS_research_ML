# bohun_research_ml
## Analysis Flow in this project

### * Data preparation and Preprocessing 
#### -EDA / Missing values 
#### -Sacling process 
#### -We adopted SMOTE (Synthetic Minority Oversampling Technique) in order to solve Class Imbalanced problems in our data. 
#### After constructing new data with this technique, 
#### we compared the performeace of prediction according to different types of Machine Learning models. 

### * ML Model comparison (Optimization process Included)  
#### -Traditional ML model 
##### -Linear regression, SVM, Naive Bayes 
#### Ensemble ML model
##### -Bagging: RF model
##### -Boosting: XGB, LGBM, CatBoost

### * Feature Selection 
##### After comparing our ML models, 
##### we used feature selection among radiomics variables(pet1, pet2) to evaluate importance of our predcition

### * DeLong test: Auc Comparison and p-vlaue were considered - Using R software
### * Research Thesis Link: currently working on it 
