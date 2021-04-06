from inspect import getmembers, isclass
from sklearn.datasets import make_classification, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from core.Feature_Selection import Evaluation_Function
from core.Feature_Selection.Ranking_Function import Ranking_Selection
import numpy as np
from tqdm import tqdm
import time


eff_dict = {}
time_dict = {}
n_rep = 10


def generate_full_random(n_features=200, n_informative=5):
    X, y = make_classification(n_samples=2000, n_features=n_features, n_informative=n_informative, n_redundant=0,
                               n_repeated=0, n_classes=2)
    return X,y

def generate_iris():
    X, y = load_iris(return_X_y=True)
    X = np.hstack((X, 2 * np.random.random((X.shape[0], 36))))
    return X,y


clf = SVC(gamma="auto")

for i in tqdm(range(n_rep)):
    #X,y = generate_iris()
    X,y = generate_full_random()
    t = time.time()
    t = time.time() - t
    base =  cross_val_score(clf, X, y)
    eff_dict["base"] = eff_dict.get("base", []) + [base]
    time_dict["base"] = time_dict.get("base", []) + [0]

    for i, j in (getmembers(Evaluation_Function, isclass)):
        if ("Function" in i):
            EF = j()
            R = Ranking_Selection(j())
            t = time.time()
            c = R.fit(X, y)
            t = time.time() - t
            x_new = R.select(50)
            clf = SVC(gamma="auto")
            clf.fit(x_new, y)
            eff_dict[i] = eff_dict.get(i, []) + [cross_val_score(clf, x_new, y)]
            time_dict[i] = time_dict.get(i, []) + [t]



for i in eff_dict:
    arr = np.array(eff_dict[i])
    time_arr = np.array(time_dict[i])
    print(i , ": ", arr.mean(), "  Time: ", time_arr.mean())