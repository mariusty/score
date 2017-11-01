from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn import preprocessing 

dataset = pd.read_csv("test_task.csv", low_memory=False, usecols=['gender', 'children', 'age', 'maritalSt', 'region', 'city', 'education', 'pens', 'car', 'occupation', 'work_experience', 'loanPeriod', 'loanType', 'loanAmount', 'rate', 'loanNumberTotal', 'loanNumberOverdue', 'overdueTotal', 'overdueMax', 'prolongation', 'collector', 'overdueCurrent', 'monthlyBudget', 'student','indpred', 'realEstate', 'limitSimple', 'limitAnn', 'marketChannel', 'loanPurpose', 'amountAsked','overdue_30', 'target_prediction', 'location_type', 'mean_overdue_days', 'workpos_correct', 'main_income', 'total_income', 'compound_int', 'loans_last30', 'loans_last90', 'loans_last90', 'loans_last180', 'loans_last365', 'deviation_from_region_median', 'deviation_from_city_median', 'new_client', 'collector_ratio', 'prolong_ratio', 'amount_ratio', 'sum_income_ratio', 'overdue_ratio', 'good_loans']) 

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
for i in sub:
	sub[t]=round(sub[t])
	if sub[t]<0:	#minimum overdue can be 0
		sub[t]=0
	if arr[t]>=180 and sub[t]>=180:
		big+=1
	if arr[t]==0 and sub[t]==0:
			zero+=1
	if arr[t]<180 or sub[t]<180:	
			if arr[t]>=180 and sub[t]<=180:
				a+=1
			if arr[t]<=180 and sub[t]>=180:
				b+=1
				fp+=sub[t]-arr[t]
			if arr[t]<180 and sub[t]<180:			
				err= abs(arr[t]-sub[t]) #calculate the error 
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
print "Wrong predicted <180 = ", a
print "Wrong predicted >180 = ", b, " av. error  ",fp/b
print "Predicted & current overdue > 180 = ", big
print "Predicted & current overdue is 0 = ", zero
print "Predicted value is lower = ", less		
print "Predicted value is higher = ", more
print "Predicted value is equal to test info = ", ex
print "Number of errors with difference > ", criticalerrorrate, " = ",criticalerror
print "Max error = ", m
print "Mean error in [0,180] = ", s/t, " days for a loan"

result = pd.DataFrame({
        "Actual": y_validation,
	"Predicted": sub,
    })
result.to_csv('result.csv',header=True, index=False)
