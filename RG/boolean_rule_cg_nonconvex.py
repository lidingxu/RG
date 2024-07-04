import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from .linesearch import line_search
from scipy import optimize 
import cvxpy as cvx

import time

from .beam_search import beam_search, beam_search_K1


class BooleanRuleCGNonconvex(BaseEstimator, ClassifierMixin):
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
        lambda2=0.00,
        CNF=False,
        iterMax=200,
        iterGDMax=15000,
        timeMax=200,
        K=10,
        D=10,
        B=5,
        eps=1e-6,
        maxRound=20,
        filter_eps= 0.0,
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
        self.lambda2 = lambda2
        # CNF instead of DNF
        self.CNF = CNF
        # Column generation parameters
        self.iterMax = iterMax      # maximum number of iterations
        self.iterGDMax = iterGDMax  # maximum number of gradient descent iterations
        self.timeMax = timeMax      # maximum runtime in seconds
        self.K = K                  # maximum number of columns generated per iteration
        self.D = D                  # maximum degree
        self.B = B                  # beam search width
        self.maxRound = maxRound
        # Numerical tolerance on comparisons
        self.eps = eps
        # Silence output
        self.silent = silent
        self.filter_eps = filter_eps
    
    # return the value of the continuous loss function: 
    def _loss(self, w, A, Pindicate, Zindicate, cs):
        Aw = np.dot(A, w)
        n =  Aw.shape[0]
        Ploss = np.sum(np.maximum(1 - Aw[Pindicate], 0))
        Zloss = np.sum(np.minimum(Aw[Zindicate], 1))
        loss =  (Ploss + Zloss) / n  +  np.dot(cs , w)
        if self.lambda2 > 0:
            loss =  (Ploss + Zloss) / n  +  np.dot(cs , w) + self.lambda2 * np.sum(np.minimum(1 -w, w ))
        else:
            loss =  (Ploss + Zloss) / n  +  np.dot(cs , w) 
        return loss
    
    # return the value of the continuous validation loss function: 
    def _loss_val(self, w, A_val, Pindicate_val, Zindicate_val):
        Aw = np.dot(A_val, w)
        n =  Aw.shape[0]
        Ploss = np.sum(np.maximum(1 - Aw[Pindicate_val], 0))
        Zloss = np.sum(np.minimum(Aw[Zindicate_val], 1))
        loss =  (Ploss + Zloss) / n 
        return loss

    # return the gradient 
    def _gradient(self, w, A, Pindicate, Zindicate, cs):
        Aw = np.dot(A, w)
        AwL1 = Aw <= 1
        n =  Aw.shape[0]
        at_pos = np.where( AwL1 & Pindicate)[0]
        at_neg = np.where( AwL1 & Zindicate)[0]
        g = (-np.sum(A[at_pos], 0) + np.sum(A[at_neg], 0)) / n  + cs + np.where(w >= 0.5, 1, -1) * self.lambda2
        return g
    
    # return the reduced cost
    def _reduced_cost(self, w, r, A, Pindicate, Zindicate):
        Aw = np.dot(A, w)
        AwL1 = Aw <= 1
        n =  Aw.shape[0]
        at_pos = np.where( AwL1 & Pindicate)[0]
        at_neg = np.where( AwL1 & Zindicate)[0]
        r.fill(0)
        r[at_pos] += -1
        r[at_neg] += 1
        r /= n
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
        alpha, falpha = line_search(self._loss, self._gradient, w, d, g, args=(A, Pindicate, Zindicate, cs), amax = amax, maxiter=20)
        if alpha is None:
            return None, None
        else:
            return w + alpha * d, falpha

    # gradient descent w
    def _gradient_descent(self, w, A, Pindicate, Zindicate, cs, old_fval=None, old_old_fval=None, c1=1e-4, c2=0.9, amax=50, amin=1e-8, xtol=1e-14):
        g = self._gradient(w, A, Pindicate, Zindicate, cs)
        mu = 0.9
        tau = 0.1
        self.b = g if self.it == 0 else (mu * self.b + (1 - tau) * g)
        g = g + mu * self.b
        # clipping direction
        alpha_max = 1e-1
        alpha_min = 1e-4
        alpha = alpha_min + (alpha_max - alpha_min) * (1 + np.cos( (self.it / self.iterMax) * np.pi) ) /2
        return np.clip(w - alpha * g, 0, 1), g
   
    def fit(self, X, y, X_val, y_val):
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
        r = np.zeros(n)

        # Initialize with empty and singleton conjunctions, i.e. X plus all-ones feature
        # Feature indicator and conjunction matrices
        # z: num_features * num_rules
        z = pd.DataFrame(np.eye(X.shape[1], X.shape[1]+1, 1, dtype=int), index=X.columns) 
        # A: num_samples * num_rules
        A = np.hstack((np.ones((X.shape[0],1), dtype=int), X))

        if X_val is None and y_val is None:
            self.use_val = False
        else:
            self.use_val = True

        if self.use_val:
            Pindicate_val = y_val > 0.5
            P_val = np.where(Pindicate_val)[0]
            Zindicate_val = y_val < 0.5
            Z_val = np.where(Zindicate_val)[0]
            nP_val = len(P_val)
            n_val = len(y_val) 
            A_val = np.hstack((np.ones((X_val.shape[0],1), dtype=int), X_val))
        self.logs = {}
        self.logs["regobjs_train"] = []
        self.logs["conv_points"] = []

        cs = self.lambda0 + self.lambda1 * z.sum().values
        cs[0] = 0

        # Iteration counter
        self.it = 0
        self.itGD = 0
        # Start time
        self.starttime = time.time()

        # Formulate master LP
        w = np.random.uniform(size = A.shape[1])

        obj = self._loss(w, A, Pindicate, Zindicate, cs)
        if self.use_val:
            obj_val = self._loss_val(w, A_val, Pindicate_val, Zindicate_val)

        prev_obj = np.finfo(np.float64).max
        generate_rule = True
        self.real_obj = np.finfo(np.float64).max
        stop_obj = np.finfo(np.float64).max
        #stop_objs.fill(np.finfo(np.float64).max)
        round_k = 0
        best_obj = np.finfo(np.float64).max
        best_w = np.rint(w)
        best_z = z
        while (self.it < self.iterMax) and (self.itGD < self.iterGDMax) and (time.time()-self.starttime < self.timeMax):
            w, _ = self._gradient_descent(w, A, Pindicate, Zindicate, cs)
            obj = self._loss(w, A, Pindicate, Zindicate, cs)
            func_converge = abs(obj - prev_obj) < self.eps * (1 + prev_obj)
            restrict_converge = func_converge # and optimal_converge # and step_converge # (prev_obj - obj) < self.eps
            self.itGD += 1

            self.real_obj = obj   

            # records logs
            self.logs["regobjs_train"].append(obj)
            self.logs["conv_points"].append(1 if restrict_converge else 0)  
            
            # relative function tolerance
            generate_rule = restrict_converge
            prev_obj = obj

            find_rule = False
            if generate_rule:
                # Compute reduced cost
                r = self._reduced_cost(w, r, A, Pindicate, Zindicate)
                # Beam search for conjunctions with negative reduced cost
                # Most negative reduced cost among current variables
                UB = np.dot(r, A) + cs
                UB = min(UB.min(), 0)
                v, zNew, Anew = beam_search(r, X, self.lambda0, self.lambda1, K=self.K, UB=UB, D=self.D, B=self.B, eps=self.eps)

                find_rule = (v < -self.eps).any()
                # Add to existing conjunctions
                if find_rule:
                    z = pd.concat([z, zNew], axis=1, ignore_index=True)
                    A = np.concatenate((A, Anew), axis=1)
                    if self.use_val:
                        # Conjunctions corresponding to solutions
                        Anew  = 1 - (np.dot(1 - X_val, zNew) > 0)
                        A_val = np.concatenate((A_val, Anew), axis = 1)
                    w = np.concatenate((w, np.zeros(Anew.shape[1])))
                    self.b = np.concatenate((self.b, np.zeros(Anew.shape[1])))
                    cs = np.concatenate((cs, self.lambda0 + self.lambda1 * zNew.sum().values))
                self.it += 1

                if not self.silent:
                    print('Iteration: {}, Objective: {:.6f}, Convergence (abs.): {}, Generate rule: {},  Number of rules: {} '.format(self.it, obj, restrict_converge, generate_rule, w.shape[0]))            

            stop = False
            if restrict_converge:
                stop = abs(obj - stop_obj) < self.eps * (1 + obj)
                stop_obj = obj  

            if restrict_converge and not find_rule and stop:
                if  self.maxRound > 0:
                    rd_w = np.rint(w)
                    rd_obj = self._loss(rd_w, A, Pindicate, Zindicate, cs)
                    if  rd_obj < best_obj:
                        best_w = rd_w 
                        best_z = z
                        best_obj = rd_obj
                    w = rd_w
                    self.b = 0
                    round_k += 1
                    print('Round Iteration: {}, Round Obj: {}'.format(round_k, rd_obj))
                    if round_k > self.maxRound:
                        break
                else:
                    break

        # Save generated conjunctions and LP solution
        self.z = z
        self.wLP = w

        if self.maxRound > 0:
            self.w = best_w
            self.z = best_z
        else:
            if self.use_val:
                r = np.full(nP_val, 1./n_val)
                self.w = beam_search_K1(r, pd.DataFrame(1-A_val[P_val,:]), 0, A[Z_val,:].sum(axis=0) / n_val,
                                        UB=r.sum(), D=100, B=2*self.B, eps=self.eps, stopEarly=False)[1].values.ravel()
            else:
                if self.filter_eps > 0:
                    filter_inds = np.where(self.wLP > self.filter_eps)
                    r = np.full(nP, 1./n)
                    A=A[:, filter_inds]
                    cs=cs[filter_inds]   
                    z=z[filter_inds]
                    self.w = beam_search_K1(r, pd.DataFrame(1-A[P,:]), 0, A[Z,:].sum(axis=0) / n + cs,
                                            UB=r.sum(), D=100, B=2*self.B, eps=self.eps, stopEarly=False)[1].values.ravel()
                    obj = self._loss(self.w, A, Pindicate, Zindicate, cs)          
                else:
                    r = np.full(nP, 1./n)
                    self.w = beam_search_K1(r, pd.DataFrame(1-A[P,:]), 0, A[Z,:].sum(axis=0) / n + cs,
                                            UB=r.sum(), D=100, B=2*self.B, eps=self.eps, stopEarly=False)[1].values.ravel()
                    obj = self._loss(self.w, A, Pindicate, Zindicate, cs)

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

    def statistics(self, **kwargs):
        """Return statistics.

        """
        return self.real_obj, self.logs
