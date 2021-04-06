from core.Feature_Selection.Discretization import bins_discretizer


class Evaluation_Abstract():

    def __init__(self, function = None):
        self.function = None
        if(function):
            self.function = function

    def evaluate(self,x,y):
        e = self.function(x,y)
        return e

    def __call__(self, *args, **kwargs):
        return self.evaluate(args[0], args[1])


class Categorical_Evaluation_Abstract(Evaluation_Abstract):

    def __init__(self, function=None, discretizer=bins_discretizer):
        super().__init__(function)
        if(discretizer):
            self.discretizer = discretizer

    def evaluate(self,x,y):
        x = self.discretizer(x)
        return super().evaluate(x,y)

    def __call__(self, *args, **kwargs):
        return self.evaluate(args[0], args[1])


class Classification_Abstract(Evaluation_Abstract):

    def __init__(self, clf_class):
        super().__init__()
        self.fited = False
        self.clf_class = clf_class


    def evaluate(self,x,y, **kwargs):
        self.clf = self.clf_class(**kwargs)
        self.clf.fit(x.reshape((-1,1)), y)
        self.fited = True
        return self.clf.score(x.reshape((-1,1)),y)