from __future__ import division
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn import preprocessing 

dataset = pd.read_csv("1.csv", low_memory=False)

#replace null values, convert object data to categorical
dataset.fillna(-9999, inplace=True)
for f in dataset.columns: 
    if dataset[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(dataset[f].values)) 
        dataset[f] = lbl.transform(list(dataset[f].values))

#split data in a random way - equal to train_test_split but data is stored in arrays
dataset['split'] = np.random.randn(dataset.shape[0], 1)
msk = np.random.rand(len(dataset)) <= 0.8
data = dataset[msk]
test = dataset[~msk]
#print data.head(5)
X_validation = test.drop('overdueCurrent', axis=1)
y_validation = test.overdueCurrent


#split dataset for train and validation
X_train = dataset.drop('overdueCurrent', axis=1)
y_train = dataset.overdueCurrent
arr=np.array(y_validation, dtype=pd.Series) #for future 



model = CatBoostRegressor(iterations=500,#100,500,1000 
                        learning_rate=0.3, #0.1,1,3,0.33
                        depth=6, #6 worse but faster
                        l2_leaf_reg=15, 
                        rsm=1, #all features in every iteration
                        loss_function='RMSE', 
                        border_count=255,#1, 255
                        feature_border_type='MinEntropy',#Median MinEntropy  
                        fold_permutation_block_size=1,
                        od_wait=20,
                        od_type='Iter',
                        #gradient_iterations=1,
                        leaf_estimation_method='Newton',
                        priors=None,feature_priors=None,#play with feature priors???
                        #random_strength=1,
                        #custom_loss=None,#loss function
                        #eval_metric='RMSE',   Logloss    MAE    CrossEntropy    Quantile    LogLinQuantile    MultiClass    MultiClassOneVsAll    MAPE    Poisson  Recall    Precision    F1    TotalF1    AUC    Accuracy    R2    MCC,
                        save_snapshot=True,
                        allow_writing_files=True)

model.fit(X_train, y_train)

#print model.score (X_validation, y_validation)
sub = model.predict(X_validation)


s = 0
t = 0
m = 0
criticalerror = 0
criticalerrorrate = 10
more = 0
less = 0
big = 0
a = 0
b = 0
fp = 0
zero = 0
ex=0
k=0
av=0
fl=0
for i in sub:
	sub[t]=round(sub[t])
	if sub[t]<0:	#minimum overdue can be 0
		#print sub[t]
		sub[t]=0
	if arr[t]==0 and sub[t]==0:
		zero+=1
		ex+1
	if arr[t]==180 and sub[t]==180:
		ex+=1
	if arr[t]>180 and sub[t]>180:
		big+=1
	if (arr[t]<180 or sub[t]<180) and (arr[t]>0 or sub[t]>0):
			k+=1
			err= abs(arr[t]-sub[t]) #calculate the error	
			if arr[t]>180 and sub[t]<180:
				a+=1
				fl+=err
				#print arr[t], " ", sub[t]
			if arr[t]<180 and sub[t]>180:
				b+=1
				fp+=err
				#print arr[t], " ", sub[t]
			if arr[t]<180 and sub[t]<180:
				av+=1			 
				#print arr[t], "  ", sub[t]
				s+=err
				if arr[t]>sub[t]:
					less+=1
				if arr[t]<sub[t]:
					more+=1
				if arr[t]==sub[t]:
					ex+=1
				if criticalerrorrate<err:
					criticalerror+=1
					if err>m:  #find the maximum error in predictions
						m=err
						#print arr[t], "   ", sub[t]	
	t+=1
print "Number of predictions = ", t
print "-----------------------------------"
print "Predicted & current overdue > 180 = ", big, "    ", 100*big/t,"%"
print "Predicted & current overdue is 0 = ", zero, "    ", 100*zero/t,"%"
print "Predicted & current overdue is in (0;180) = ", av, "    ", 100*av/t,"%"
print "Right classification in classes = ", big+zero+av, "    ", 100*(big+zero+av)/t,"%"
print "------------------------------------------------"
print "Wrong predicted <180 = ", a, " av. error  ",fl/a,"     ", 100*a/t,"%"
print "Wrong predicted >180 = ", b, " av. error  ",fp/b, "    ", 100*b/t,"%"
print "Wrong classification ", a+b, "    ", 100*(a+b)/t,"%"
print "------------------------------------------------"
print "Predicted value is lower = ", less, "    ", 100*less/k,"%"		
print "Predicted value is higher = ", more, "     ", 100*more/k,"%"
print "Predicted value is equal to data = ", ex, "    ", 100*ex/t,"%"
print "------------------------------------------------"
print "Predictions with error > ", criticalerrorrate, " = ",criticalerror, "    ", 100*criticalerror/k,"%"
print "Predictions with error < ", criticalerrorrate, " = ", k - criticalerror, "    ", 100*(k-criticalerror)/k,"%"
print "Max error in [0; 180]= ", m
print "Mean error in [0; 180]= ", s/k, " days for a loan"

result = pd.DataFrame({
        "Actual": y_validation,
	"Predicted": sub,
    })
result.to_csv('result.csv',header=True, index=False)
