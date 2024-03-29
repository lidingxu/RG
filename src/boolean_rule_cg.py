import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from linesearch import line_search

import time

from beam_search import beam_search, beam_search_K1


class BooleanRuleCG(BaseEstimator, ClassifierMixin):
    """BooleanRuleCG is a directly interpretable supervised learning method
    for binary classification that learns a Boolean rule in disjunctive
    normal form (DNF) or conjunctive normal form (CNF) using column generation (CG).
    AIX360 implements a heuristic beam search version of BRCG that is less 
    computationally intensive than the published integer programming version [#NeurIPS2018]_.

    References:
        .. [#NeurIPS2018] `S. Dash, O. Günlük, D. Wei, "Boolean decision rules via
           column generation." Neural Information Processing Systems (NeurIPS), 2018.
           <https://papers.nips.cc/paper/7716-boolean-decision-rules-via-column-generation.pdf>`_
    """
    def __init__(self,
        lambda0=0.001,
        lambda1=0.001,
        CNF=False,
        iterMax=2000,
        timeMax=200,
        K=10,
        D=10,
        B=5,
        eps=1e-6,
        silent=False):
        """
        Args:
            lambda0 (float, optional): Complexity - fixed cost of each clause
            lambda1 (float, optional): Complexity - additional cost for each literal
            CNF (bool, optional): CNF instead of DNF
            iterMax (int, optional): Column generation - maximum number of iterations
            timeMax (int, optional): Column generation - maximum runtime in seconds
            K (int, optional): Column generation - maximum number of columns generated per iteration
            D (int, optional): Column generation - maximum degree
            B (int, optional): Column generation - beam search width
            eps (float, optional): Numerical tolerance on comparisons
            silent (bool, optional): Silence overall algorithm messages
        """
        # Complexity parameters
        self.lambda0 = lambda0      # fixed cost of each clause
        self.lambda1 = lambda1      # additional cost per literal
        # CNF instead of DNF
        self.CNF = CNF
        # Column generation parameters
        self.iterMax = iterMax      # maximum number of iterations
        self.timeMax = timeMax      # maximum runtime in seconds
        self.K = K                  # maximum number of columns generated per iteration
        self.D = D                  # maximum degree
        self.B = B                  # beam search width
        # Numerical tolerance on comparisons
        self.eps = eps
        # Silence output
        self.silent = silent
    
    # return the value of the continuous loss function: 
    def _loss(self, w, A, Pindicate, Zindicate, cs):
        Aw = np.dot(A, w)
        n =  Aw.shape[0]
        inds_neg = np.where(Zindicate)[0]
        inds_pos = np.where(Pindicate)[0]
        Ploss = np.sum(np.maximum(1 - Aw[inds_pos], 0))
        Zloss = np.sum(np.minimum(Aw[inds_neg], 1))
        loss =  (Ploss + Zloss) / n  +  np.dot(cs , w)
        return loss

    # return the gradient 
    def _gradient(self, w, A, Pindicate, Zindicate, cs):
        Aw = np.dot(A, w)
        AwL1 = Aw <= 1
        n =  Aw.shape[0]
        at_pos = np.where( AwL1 & Pindicate)[0]
        at_neg = np.where( AwL1 & Zindicate)[0]
        g = (-np.sum(A[at_pos], 0) + np.sum(A[at_neg], 0)) / n  + cs
        return g
    
    # return the reduced cost
    def _reduced_cost(self, w, A, Pindicate, Zindicate):
        Aw = np.dot(A, w)
        AwL1 = Aw <= 1
        n =  Aw.shape[0]
        at_pos = np.where( AwL1 & Pindicate)[0]
        at_neg = np.where( AwL1 & Zindicate)[0]
        r_pos = np.zeros(n)
        r_pos[at_pos] = -1
        r_neg = np.zeros(n)
        r_neg[at_neg] = 1
        r = (r_pos + r_neg) / n
        return  r

    # gradient descent with line search
    def _line_search(self, w, A, Pindicate, Zindicate, cs, old_fval=None, old_old_fval=None, c1=1e-4, c2=0.9, amax=50, amin=1e-8, xtol=1e-14):
        g = self._gradient(w, A, Pindicate, Zindicate, cs)
        d = -g
        # clipping direction
        fmax = np.finfo(np.float64).max
        amax = fmax
        for i in range(w.shape[0]):
            if abs(d[i]) > self.eps:
                maxi = (1- w[i]) / d[i] if d[i] > 0 else w[i] / -d[i]
                if maxi < self.eps:
                    d[i] = 0
                else:
                    amax = min(amax, maxi)
        alpha, _ = line_search(self._loss, self._gradient, w, d, g, args=(A, Pindicate, Zindicate, cs), amax = amax, maxiter=40)
        if alpha is None:
            return None
        else:
            return w + alpha * d

   
    def fit(self, X, y):
        """Fit model to training data.
        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
            y (array): Binary-valued target variable
        Returns:
            BooleanRuleCG: Self
        """
        if not self.silent:
            print('Learning {} rule with complexity parameters lambda0={}, lambda1={}'\
                  .format('CNF' if self.CNF else 'DNF', self.lambda0, self.lambda1))
        if self.CNF:
            # Flip labels for CNF
            y = 1 - y
        # Positive (y = 1) and negative (y = 0) samples
        Pindicate = y > 0.5
        P = np.where(Pindicate)[0]
        Zindicate = y < 0.5
        Z = np.where(Zindicate)[0]
        nP = len(P)
        n = len(y)

        # Initialize with empty and singleton conjunctions, i.e. X plus all-ones feature
        # Feature indicator and conjunction matrices
        # z: num_features * num_rules
        z = pd.DataFrame(np.eye(X.shape[1], X.shape[1]+1, 1, dtype=int), index=X.columns)
        # A: num_samples * num_rules
        A = np.hstack((np.ones((X.shape[0],1), dtype=int), X))

        cs = self.lambda0 + self.lambda1 * z.sum().values
        cs[0] = 0

        # Iteration counter
        self.it = 0
        # Start time
        self.starttime = time.time()

        # Formulate master LP
        w = np.random.uniform(size = A.shape[1])

        sufficient_descent = True
        find_neg_cost = True

        while ( find_neg_cost or sufficient_descent) and (self.it < self.iterMax) and (time.time()-self.starttime < self.timeMax):
            # Negative reduced costs found
            self.it += 1
            obj = self._loss(w, A, Pindicate, Zindicate, cs)
            if not self.silent:
                print('Iteration: {}, Objective: {:.4f}, Find negative cost: {}, Sufficient descent: {} '.format(self.it, obj, find_neg_cost, sufficient_descent))

            # Line search based gradient descent
            new_w = self._line_search(w, A, Pindicate, Zindicate, cs)

            if new_w is None:
                sufficient_descent = False
            else:
                w = new_w
                sufficient_descent = True

            # Compute reduced cost
            r = self._reduced_cost(w, A, Pindicate, Zindicate)

            if not self.silent:
                print('Reduced cost norm: {}, number of rules: {}'.format(np.linalg.norm(r), w.shape[0]))

            # Beam search for conjunctions with negative reduced cost
            # Most negative reduced cost among current variables
            UB = np.dot(r, A) + cs
            UB = min(UB.min(), 0)
            v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1, K=self.K, UB=UB, D=self.D, B=self.B, eps=self.eps)
            print(v.shape, zNew.shape)

            # Add to existing conjunctions
            find_neg_cost =  (v < -self.eps).any()
            if find_neg_cost:
                z = pd.concat([z, zNew], axis=1, ignore_index=True)
                A = np.concatenate((A, Anew), axis=1)
                w = np.concatenate((w, np.zeros(Anew.shape[1])))
                cs = np.concatenate((cs, self.lambda0 + self.lambda1 * zNew.sum().values))


        # Save generated conjunctions and LP solution
        self.z = z

        r = np.full(nP, 1./n)
        self.w = beam_search_K1(r, pd.DataFrame(1-A[P,:]), 0, A[Z,:].sum(axis=0) / n + cs,
                                UB=r.sum(), D=100, B=2*self.B, eps=self.eps, stopEarly=False)[1].values.ravel()
        if len(self.w) == 0:
            self.w = np.zeros_like(self.wLP, dtype=int)

    def compute_conjunctions(self, X):
        """Compute conjunctions of features as specified in self.z.
        
        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
        Returns:
            array: A -- Conjunction values
        """
        try:
            A = 1 - (np.dot(1 - X, self.z) > 0) # Changed matmul to dot, because failed on some machines
        except AttributeError:
            print("Attribute 'z' does not exist, please fit model first.")
        return A

    def predict(self, X):
        """Predict class labels.

        Args:
            X (DataFrame): Binarized features with MultiIndex column labels
        Returns:
            array: y -- Predicted labels
        """
        # Compute conjunctions of features
        A = self.compute_conjunctions(X)
        # Predict labels
        if self.CNF:
            # Flip labels since model is actually a DNF for Y=0
            return 1 - (np.dot(A, self.w) > 0)
        else:
            return (np.dot(A, self.w) > 0).astype(int)

    def explain(self, maxConj=None, prec=2):
        """Return rules comprising the model.

        Args:
            maxConj (int, optional): Maximum number of conjunctions to show
            prec (int, optional): Number of decimal places to show for floating-value thresholds
        Returns:
            Dictionary containing
            
            * isCNF (bool): flag signaling whether model is CNF or DNF
            * rules (list): selected conjunctions formatted as strings
        """
        # Selected conjunctions
        z = self.z.loc[:, self.w > 0.5]
        truncate = (maxConj is not None) and (z.shape[1] > maxConj)
        nConj = maxConj if truncate else z.shape[1]

        """
        if self.CNF:
            print('Predict Y=0 if ANY of the following rules are satisfied, otherwise Y=1:')
        else:
            print('Predict Y=1 if ANY of the following rules are satisfied, otherwise Y=0:')
        """

        # Sort conjunctions by increasing order
        idxSort = z.sum().sort_values().index[:nConj]
        # Iterate over sorted conjunctions
        conj = []
        for i in idxSort:
            # MultiIndex of features participating in rule i
            idxFeat = z.index[z[i] > 0]
            # String representations of features
            strFeat = idxFeat.get_level_values(0) + ' ' + idxFeat.get_level_values(1)\
                + ' ' + idxFeat.get_level_values(2).to_series()\
                .apply(lambda x: ('{:.' + str(prec) + 'f}').format(x) if type(x) is float else str(x))
            # String representation of rule
            strFeat = strFeat.str.cat(sep=' AND ')
            conj.append(strFeat)

        return {
            'isCNF': self.CNF,
            'rules': conj
        }

