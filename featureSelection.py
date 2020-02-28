import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import preprocessing
import sys

# sys.path.append('/home/marielb/anaconda3/lib/python3.7/site-packages/boruta')

from boruta import BorutaPy

seed = 1603
np.random.seed(seed)


class Data:
    def __init__(self, filename):
        self.data = pd.read_csv(filename, index_col=0)

    def getY(self):
        group = self.data['Group'].values
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(group)
        return y

    def getX(self):
        x = self.data.drop('Group', axis=1).values
        return x

    def getSplit(self, X, y):
        X_Pretrain, X_validation, y_Pretrain, y_validation = train_test_split(X, y, test_size=.20, random_state=seed,
                                                                              stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X_Pretrain, y_Pretrain, test_size=.20, random_state=seed,
                                                            stratify=y_Pretrain)
        return (X_train, X_test, y_train, y_test, X_validation, y_validation)


rf_default = RandomForestClassifier(random_state=seed)
svm_default = SVC(probability=True, random_state=seed)
lg_default = LogisticRegression(random_state=seed)
knn_default = KNeighborsClassifier()
lda_default = LinearDiscriminantAnalysis()
vote = VotingClassifier(
    estimators=[('SVM', svm_default), ('Random Forests', rf_default), ('LogReg', lg_default), ('KNN', knn_default),
                ('LDA', lda_default)], voting='soft')

##LDA
rf_ADA = RandomForestClassifier(n_estimators=11, random_state=seed)
svm_ADA = SVC(kernel='poly', gamma='auto', C=4.12481631472, probability=True, random_state=seed)
lg_ADA = LogisticRegression(solver='newton-cg', max_iter=235, C=135.6688, random_state=seed)
knn_ADA = KNeighborsClassifier(n_neighbors=5)
lda_ADA = LinearDiscriminantAnalysis(tol=0.000584259)
adasyn = VotingClassifier(
    estimators=[('SVM', svm_ADA), ('Random Forests', rf_ADA), ('LogReg', lg_ADA), ('KNN', knn_ADA), ('LDA', lda_ADA)],
    voting='soft')

rf_SMOTE = RandomForestClassifier(n_estimators=84, random_state=seed)
svm_SMOTE = SVC(kernel='poly', gamma='auto', C=12.6360380346, probability=True, random_state=seed)
lg_SMOTE = LogisticRegression(solver='newton-cg', max_iter=432, C=50.99227570850435, random_state=seed)
knn_SMOTE = KNeighborsClassifier(n_neighbors=5)
lda_SMOTE = LinearDiscriminantAnalysis(tol=9.25895394348e-06)
smote = VotingClassifier(
    estimators=[('SVM', svm_SMOTE), ('Random Forests', rf_SMOTE), ('LogReg', lg_SMOTE), ('KNN', knn_SMOTE),
                ('LDA', lda_SMOTE)], voting='soft')

rf_Amazon = RandomForestClassifier(n_estimators=389, random_state=seed)
svm_Amazon = SVC(kernel='poly', gamma='auto', C=2.48906112826, probability=True, random_state=seed)
lg_Amazon = LogisticRegression(solver='newton-cg', max_iter=1022, C=0.0224618581186563, random_state=seed)
knn_Amazon = KNeighborsClassifier(n_neighbors=5)
lda_Amazon = LinearDiscriminantAnalysis(tol=0.000785350859773)
pso = VotingClassifier(
    estimators=[('SVM', svm_Amazon), ('Random Forests', rf_Amazon), ('LogReg', lg_Amazon), ('KNN', knn_Amazon),
                ('LDA', lda_Amazon)], voting='soft')
##LDA

print("reading data...")


# MLData = Data("../../ADB_code/expData_PSO.csv")

def whichData(dataset):
    if dataset == 'unbalanced':
        xydata = Data("../../ADB_code/expData_PSO.csv")
        y = xydata.getY()
        X = xydata.getX()
        ensemble = [rf_Amazon, svm_Amazon, lg_Amazon, knn_Amazon, lda_Amazon]
    if dataset == 'ada':
        xydata = Data("../../ADB_code/ADASYN_PSOdata.csv")
        y = xydata.getY()
        X = xydata.getX()
        ensemble = [rf_ADA, svm_ADA, lg_ADA, knn_ADA, lda_ADA]
    if dataset == 'smote':
        xydata = Data("../../ADB_code/randomOverSamplingPSOdata.csv")
        y = xydata.getY()
        X = xydata.getX()
        ensemble = [rf_SMOTE, svm_SMOTE, lg_SMOTE, knn_SMOTE, lda_SMOTE]
    return ({'X': X, 'y': y, 'dataset': dataset, 'ensemble': ensemble})
class getSelectedFeatures:
	def __init__(self, filename):
		self.data = pd.read_csv(filename)

	def getTrue(self):
		df = self.data
		sf  = df.index[df['0']].tolist()
		return sf

data = whichData('unbalanced')

## PSO Unbalanced
names = ['RF', 'SVM', 'LogReg', 'KNN', 'LDA']
# ensemble = [rf_Amazon,svm_Amazon,lg_Amazon,knn_Amazon,lda_Amazon]
sfdict = {}
for clf, label in zip(data['ensemble'], names):
    print(label)
    print("Boruta feature selection method")
    # define Boruta feature selection method
    feat_selector = BorutaPy(clf, n_estimators='auto', verbose=2, random_state=seed)

    print("fit")
    feat_selector.fit(data['X'], data['y'])
    # check selected features
    print("selected features")
    feat_selector.support_
    sf = feat_selector.support_
    pd.DataFrame(sf).to_csv(data['dataset'] + "_BorutaSF.csv", header=True)
    # check ranking of features
    print("ranking of features")
    rank = feat_selector.ranking_
    pd.DataFrame(rank).to_csv(data['dataset'] + "_BorutaRank.csv", header=True)

    # call transform() on X to filter it down to selected features
    print("X_filtered")
    X_filtered = feat_selector.transform(data['X'])

    pd.DataFrame(X_filtered).to_csv(data['dataset'] + "_BorutaX_filtered.csv", header=True)

    sf = getSelectedFeatures(data['dataset'] + '_BorutaSF.csv')

    SF = sf.getTrue()

    Xfiltered = data['X'].iloc[:, SF]

    Xfiltered.to_csv(label + "_" + data['dataset'] + "_X.csv")

    sfdict[label] = np.array([list(Xfiltered)])

for k, v in sfdict.iteritems():
    print(k + '\t' + v)

df = pd.DataFrame.from_dict
df.to_csv(data['dataset'] + "_SelectedFeatures.csv")
# print("RFECV...")
# cv = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=seed)
# selector = RFECV(vote, step=10, cv=cv)
# selector = selector.fit(X, y)
#
# print(selector.support_)
