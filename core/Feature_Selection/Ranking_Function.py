from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from core.Feature_Selection.Abstract import Evaluation_Abstract


class Ranking_Selection:

    def __init__(self, evaluator : Evaluation_Abstract):
        self.evaluator = evaluator
        self.importances = None
        self.ranked_x = None

    def fit(self, X, y):
        self.importances = np.zeros(X.shape[1])
        self.X = X
        self.y = y
        for i in range(X.shape[1]):
            x = X[:, i]
            self.importances[i] = self.evaluator(x,y)
        self.importances = self.importances / self.importances.sum()
        self.ranking = self.importances.argsort()[::-1]
        return self

    def select(self, n_chosen = None):
        self.ranked_x = []
        for i in self.ranking:
            self.ranked_x.append(self.X[:, i])
        self.ranked_x = np.array(self.ranked_x).T
        k  = self.X.shape[1]
        if(n_chosen):
            k = n_chosen
        out = self.ranked_x[: , :k]
        return out


if(__name__ == "__main__"):
    X,y = make_classification(n_samples=1000, n_features=100, n_informative=50, n_classes=2)
    R = Ranking_Selection(MI_CAT_Function())
    c = R.fit(X,y)
    print(R.ranking)

