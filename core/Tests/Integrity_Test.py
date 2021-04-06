from inspect import getmembers, isclass
from sklearn.datasets import make_classification
from core.Feature_Selection import Evaluation_Function
from core.Feature_Selection.Ranking_Function import Ranking_Selection
X,y = make_classification(n_samples=1000, n_features=10, n_classes=2)
for i, j in (getmembers(Evaluation_Function, isclass)):
    if("Function" in i and not "CAT" in i):
        print("Testing: ",i, end=" ")
        EF = j()
        R = Ranking_Selection(j())
        c = R.fit(X, y)
        print("✓")