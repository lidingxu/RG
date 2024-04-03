import pandas as pd
import unittest

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from features import FeatureBinarizer
from boolean_rule_cg_nonconvex import BooleanRuleCGNonconvex
from boolean_rule_cg_convex import BooleanRuleCGConvex
from BRCG import  BRCGExplainer

class TestBooleanmRuleCG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bc = load_breast_cancer()

    def test_classification(self):
        bc_df = pd.DataFrame(self.bc.data, columns=self.bc.feature_names)
        X_train, X_test, Y_train, Y_test = train_test_split(bc_df, self.bc.target, test_size = 0.2, random_state = 31)
        fb = FeatureBinarizer(negations=True)
        X_train_fb = fb.fit_transform(X_train)
        X_test_fb = fb.transform(X_test)

        self.assertEqual(len(X_train_fb.columns), 540)
        self.assertEqual(len(X_test_fb.columns), 540)

        '''
        boolean_model = BooleanRuleCGConvex(silent = False)
        explainer = BRCGExplainer(boolean_model)
        z, A, w = explainer._model.fit(X_train_fb, Y_train)
        Y_pred = explainer.predict(X_test_fb)

        self.assertGreater(accuracy_score(Y_test, Y_pred), 0.9)
        self.assertGreater(precision_score(Y_test, Y_pred), 0.9)
        self.assertGreater(recall_score(Y_test, Y_pred), 0.9)
        self.assertGreater(f1_score(Y_test, Y_pred), 0.9)
        

        print(accuracy_score(Y_test, Y_pred), precision_score(Y_test, Y_pred), recall_score(Y_test, Y_pred), f1_score(Y_test, Y_pred))
        '''

        boolean_model2 = BooleanRuleCGNonconvex(silent = False)
        explainer2 = BRCGExplainer(boolean_model2)
        z = A = w = None
        explainer2._model.fit(X_train_fb, Y_train)
        Y_pred2 = explainer2.predict(X_test_fb)

        self.assertGreater(accuracy_score(Y_test, Y_pred2), 0.9)
        self.assertGreater(precision_score(Y_test, Y_pred2), 0.9)
        self.assertGreater(recall_score(Y_test, Y_pred2), 0.9)
        self.assertGreater(f1_score(Y_test, Y_pred2), 0.9)


if __name__ == '__main__':
    unittest.main()