import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.tree import DecisionTreeClassifier

from core.Feature_Selection.Abstract import Evaluation_Abstract, Categorical_Evaluation_Abstract, \
    Classification_Abstract
from core.Feature_Selection.Discretization import bins_discretizer


def mutual_information(x, y):
    mi = mutual_info_classif(x.reshape((-1,1)), y)
    return mi

#Linear correlation
def correlation(x,y):
    k = np.corrcoef(x, y)
    return np.abs(k[1,0])

#Correlation as defined in signal processing
def correlation_sp(x,y):
    return np.correlate(x,y)[0]



class Correlation_Function(Evaluation_Abstract):
    def __init__(self):
        super().__init__(correlation)

class CorrelationSP_Function(Evaluation_Abstract):
    def __init__(self):
        super().__init__(correlation_sp)

#class MI_CAT_Function(Evaluation_Abstract):
 #   def __init__(self):
 #       super().__init__(mutual_information)

class MI_Function(Categorical_Evaluation_Abstract):
    def __init__(self):
        super().__init__(mutual_information, bins_discretizer)

class CHI2_Function(Categorical_Evaluation_Abstract):
    def __init__(self):
        neg_chi2 = lambda x,y: -chi2(x.reshape((-1,1)),y)[1][0]
        super().__init__(neg_chi2)



class DTClassification_Function(Classification_Abstract):
    def __init__(self):
        super().__init__(DecisionTreeClassifier)


class LogisticRegression_Function(Classification_Abstract):
    def __init__(self):
        super().__init__(LogisticRegression)
#class RFClassification_Function(Classification_Abstract):
#    def __init__(self):
 #       super().__init__(RandomForestClassifier)


if(__name__ == "__main__"):
    X,y = make_classification(n_samples=100, n_features=200, n_informative=100, n_redundant=100, n_repeated=0, n_classes=2)
    R = CHI2_Function()
    R.evaluate(X[:, 1],y)
