import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture


df=pd.read_csv("ITData_train.csv")

header = df.columns 

def encode_data(data):		
	data.replace({
		"Requester_Seniority":{
			"1 - Junior":0,
			"2 - Regular":1,
			"3 - Senior":2,
			"4 - Management":3
		},
		'Agent':{
			'Systems':0,
			'Software':1,
			'Access/Login':2,
			'Hardware':3
		},
		'Ticket_Type':{
			'Issue':0,
			'Request':1
		},
		'Severity':{
			'0 - Unclassified': 0,
			'1 - Minor':1,
			'2 - Normal':2,
			'3 - Major':3,
			'4 - Critical':4
		},
		'Priority':{
			'0 - Unassigned':0,
			'1 - Low':1,
			'2 - Medium':2,
			'3 - High':3
		},
		'Satisfaction':{
			'0 - Unknown':0,
			'1 - Unsatisfied':1,
			'2 - Satisfied':2,
			'3 - Highly satisfied':3
		}
		},inplace = True)
	return data



def delete_columns(data):
	data.drop('Ticket', axis=1, inplace=True)
	return data

def seperate_label(data):
	label=data.Satisfaction
	data.drop('Satisfaction', axis=1, inplace=True)
	return data,label


def scale_columns(data):	
	scaler = MinMaxScaler()
	data = data.astype('float64')
	scaler.fit(data)
	data=scaler.transform(data)
	return data,scaler


df=encode_data(df)
df=delete_columns(df)
df,label=seperate_label(df)
df,scaler=scale_columns(df)

pickle.dump(scaler,open('./scaler.model','wb'))

x_train,x_test,y_train,y_test=train_test_split(df,label,test_size=.5)

# classifier=tree.DecisionTreeClassifier()
# classifier.fit(x_train,y_train)
# predictions=classifier.predict(x_test)

classifier = MLPClassifier()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

print("Accuracy:",accuracy_score(y_test,predictions))

pickle.dump(classifier, open("model.model", 'wb'))



print("Training completed. \nModel dumped succesfully..\n  -----------------------")


###############Evaluating#################


data=pd.read_csv("ITData_eval-unlabeled.csv")
data.columns = header
df2  = data.drop(['Satisfaction'],axis=1)
df2=encode_data(df2)
df2=delete_columns(df2)

scaler_model = pickle.load(open("scaler.model", 'rb'))
df2 = scaler_model.transform(df2)

loaded_model = pickle.load(open("model.model", 'rb'))
result = loaded_model.predict(df2)

label = data['Satisfaction']
result = pd.DataFrame(result)
out = pd.concat([result,label],axis=1)
out.columns= ['predicted','actual']
print(out)

