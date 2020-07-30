'''
[1] Yong Liu, Min Wu, Chunyan Miao, Peilin Zhao, Xiao-Li Li, "Neighborhood Regularized Logistic Matrix Factorization for Drug-target Interaction Prediction", under review.
'''
import numpy as np
import os

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class NRLMF:

    def __init__(self, cfix=5, K1=5, K2=5, num_factors=10, theta=1.0, lambda_d=0.625,
                 lambda_t=0.625, alpha=0.1, beta=0.1, max_iter=100):
        self.cfix = int(cfix)  # importance level for positive observations
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.num_factors = int(num_factors)
        self.theta = float(theta)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iter = int(max_iter)

    def AGD_optimization(self, seed=None):
        if seed is None:
            self.U = np.sqrt(1 / float(self.num_factors)) * \
                np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1 / float(self.num_factors)) * \
                np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1 / float(self.num_factors)) * \
                prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1 / float(self.num_factors)) * \
                prng.normal(size=(self.num_targets, self.num_factors))
        dg_sum = np.zeros((self.num_drugs, self.U.shape[1]))
        tg_sum = np.zeros((self.num_targets, self.V.shape[1]))
        # import pdb; pdb.Pdb().set_trace()
        last_log = self.log_likelihood()
        for t in range(self.max_iter):
            dg = self.deriv(True)
            dg_sum += np.square(dg)
            vec_step_size = self.theta / np.sqrt(dg_sum)
            self.U += vec_step_size * dg
            tg = self.deriv(False)
            tg_sum += np.square(tg)
            vec_step_size = self.theta / np.sqrt(tg_sum)
            self.V += vec_step_size * tg
            curr_log = self.log_likelihood()
            delta_log = (curr_log - last_log) / abs(last_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log

    def deriv(self, drug):
        if drug:
            vec_deriv = np.dot(self.intMat, self.V)
        else:
            vec_deriv = np.dot(self.intMat.T, self.U)
        A = np.dot(self.U, self.V.T)
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.intMat1 * A
        if drug:
            vec_deriv -= np.dot(A, self.V)
            vec_deriv -= self.lambda_d * self.U + self.alpha * np.dot(self.DL, self.U)
        else:
            vec_deriv -= np.dot(A.T, self.U)
            vec_deriv -= self.lambda_t * self.V + self.beta * np.dot(self.TL, self.V)
        return vec_deriv

    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        B = A * self.intMat
        loglik += np.sum(B)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.intMat1 * A
        loglik -= np.sum(A)
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self.U)) + \
            0.5 * self.lambda_t * np.sum(np.square(self.V))
        # import pdb; pdb.Pdb().set_trace()
        loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loglik

    def construct_neighborhood(self, drugMat, targetMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)

    def laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5 * (np.diag(x + y) - (S + S.T))  # neighborhood regularization matrix
        return L

    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    def fix_model(self, W, intMat, drugMat, targetMat, seed=None):
        self.num_drugs, self.num_targets = intMat.shape
        self.ones = np.ones((self.num_drugs, self.num_targets))
        # self.intMat is like the R defined in the main function
        self.intMat = self.cfix * intMat * W
        self.intMat1 = (self.cfix - 1) * intMat * W + self.ones
        x, y = np.where(self.intMat > 0)
        # self.train_drugs and self.train_targets are defined here
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        self.construct_neighborhood(drugMat, targetMat)
        self.AGD_optimization(seed)

    def predict_scores(self, test_data, N):
        # trouver la signification de DS
        # trouver la signification de TS

        dinx = np.array(list(self.train_drugs))
        # DS est la matrice des similarités pour les molécules du traindataset
        DS = self.dsMat[:, dinx]
        
        tinx = np.array(list(self.train_targets))
        # TS est la matrice des similarités
        TS = self.tsMat[:, tinx]
        
        scores = []
        for d, t in test_data:
            if d in self.train_drugs:
                if t in self.train_targets:
                    val = np.sum(self.U[d, :] * self.V[t, :])
                else:
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    val = np.sum(self.U[d, :] * np.dot(TS[t, jj], self.V[tinx[jj], :])) / \
                        np.sum(TS[t, jj])
            else:
                if t in self.train_targets:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :]) * self.V[t, :]) / \
                        np.sum(DS[d, ii])
                else:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    v1 = DS[d, ii].dot(self.U[dinx[ii], :]) / np.sum(DS[d, ii])
                    v2 = TS[t, jj].dot(self.V[tinx[jj], :]) / np.sum(TS[t, jj])
                    val = np.sum(v1 * v2)
            scores.append(np.exp(val) / (1 + np.exp(val)))
        return np.array(scores)

    def evaluation(self, test_data, test_label, intMat, R):
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        if self.K2 > 0:
            for d, t in test_data:
                if d in self.train_drugs:
                    if t in self.train_targets:
                        val = np.sum(self.U[d, :] * self.V[t, :])
                    else:
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        val = np.sum(self.U[d, :] * np.dot(TS[t, jj], self.V[tinx[jj], :])) / \
                            np.sum(TS[t, jj])
                else:
                    if t in self.train_targets:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :]) * self.V[t, :]) / \
                            np.sum(DS[d, ii])
                    else:
                        ii = np.argsort(DS[d, :])[::-1][:self.K2]
                        jj = np.argsort(TS[t, :])[::-1][:self.K2]
                        v1 = DS[d, ii].dot(self.U[dinx[ii], :]) / np.sum(DS[d, ii])
                        v2 = TS[t, jj].dot(self.V[tinx[jj], :]) / np.sum(TS[t, jj])
                        val = np.sum(v1 * v2)
                scores.append(np.exp(val) / (1 + np.exp(val)))
        elif self.K2 == 0:
            for d, t in test_data:
                val = np.sum(self.U[d, :] * self.V[t, :])
                scores.append(np.exp(val) / (1 + np.exp(val)))
        prec, rec, thr = precision_recall_curve(test_label, np.array(scores))
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, np.array(scores))
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val, np.array(scores)

    def predict(self, test_data, intMat_for_verbose, true_test_data=None):
        
        self.predictions = np.full(intMat_for_verbose.shape, np.inf)
        
        iii, jjj = test_data[:, 0], test_data[:, 1]

        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        if self.K2 > 0:
            for dd, tt in test_data:
                casse = False
                # print('d', dd, 'label', intMat_for_verbose[dd, tt])
                # if dd == 33:
                #     import pdb; pdb.Pdb().set_trace()
                if dd in self.train_drugs:
                    # print('in selftrain drugs')
                    if tt in self.train_targets:
                        # print('in selftrain targets')
                        val = np.sum(self.U[dd, :] * self.V[tt, :])
                    else:
                        jj = np.argsort(TS[tt, :])[::-1][:self.K2]
                        val = np.sum(self.U[dd, :] * np.dot(TS[tt, jj], self.V[tinx[jj], :])) / \
                            np.sum(TS[tt, jj])
                else:
                    if tt in self.train_targets:
                        # print('in selftrain targets')
                        ii = np.argsort(DS[dd, :])[::-1][:self.K2]
                        val = np.sum(np.dot(DS[dd, ii], self.U[dinx[ii], :]) * self.V[tt, :]) / \
                            np.sum(DS[dd, ii])
                    else:
                        ii = np.argsort(DS[dd, :])[::-1][:self.K2]
                        jj = np.argsort(TS[tt, :])[::-1][:self.K2]
                        # if d == 413:
                        #     import pdb; pdb.Pdb().set_trace()
                        if np.sum(DS[dd, ii]) == 0 or np.sum(TS[tt, jj]) == 0:
                            val = -100
                            casse = True
                        else:
                            v1 = DS[dd, ii].dot(self.U[dinx[ii], :]) / np.sum(DS[dd, ii])
                            v2 = TS[tt, jj].dot(self.V[tinx[jj], :]) / np.sum(TS[tt, jj])
                            # print(v1, self.U[dinx[ii], :], DS[dd, ii],
                            #       DS[dd, ii].dot(self.U[dinx[ii], :]), np.sum(DS[dd, ii]))
                            # print(v2, self.V[tinx[jj], :], TS[tt, jj],
                            #       TS[tt, jj].dot(self.V[tinx[jj], :]), np.sum(TS[tt, jj]))
                            val = np.sum(v1 * v2)
                ss = np.exp(val) / (1 + np.exp(val))
                if not (np.isinf(ss) or np.isnan(ss)) and not casse:
                    scores.append(ss)
                else:
                    print('casse')
                    scores.append(np.random.random())
                # print(casse, val, scores[-1], round(scores[-1]))
                # print('')
        elif self.K2 == 0:
            for d, t in test_data:
                val = np.sum(self.U[d, :] * self.V[t, :])
                scores.append(np.exp(val) / (1 + np.exp(val)))
        if true_test_data is not None:
            iii, jjj = true_test_data[:, 0], true_test_data[:, 1]
            self.predictions[iii, jjj] = scores
        else:
            self.predictions[iii, jjj] = scores

        return self.predictions

    def get_perf(self, intMat):
        pred_ind = np.where(self.predictions!=np.inf)
        pred_local = self.predictions[pred_ind[0], pred_ind[1]]
        test_local = intMat[pred_ind[0], pred_ind[1]]
        prec, rec, thr = precision_recall_curve(test_local, pred_local)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_local, pred_local)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val

    def __str__(self):
        return "Model: NRLMF, c:%s, K1:%s, K2:%s, r:%s, lambda_d:%s, lambda_t:%s, alpha:%s," + \
            "beta:%s, theta:%s, max_iter:%s" % \
            (self.cfix, self.K1, self.K2, self.num_factors, self.lambda_d, self.lambda_t,
             self.alpha, self.beta, self.theta, self.max_iter)