"""
Authors: Ramin Zarei-Sabzevar, Sayed Kamaledin Ghiasi-Shirazi
Year: 2021

"""

import numpy as np
import numpy.matlib

class CompetitiveCrossEntropy_pmEDWTA:
    def __init__ (self, trainingData, learning_rate, lr_decay_mult, beta_learning_rate,
                  max_epochs, initialization_negative, init_beta=1):
        l2_reg = 0
        l1_reg = 0
        l1_reg_always=0
        self.trainingData = trainingData
        self.learning_rate = learning_rate
        self.beta_learning_rate = beta_learning_rate
        self.max_epochs = max_epochs
        self.lr_decay_mult = lr_decay_mult
        self.init_beta = init_beta
        self.initialization_negative = initialization_negative
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.l1_reg_always = l1_reg_always
        self.td = trainingData
        self.W_pos = self.td.subclassMeans
        if initialization_negative == 'mean':
            self.W_neg = np.tile(np.mean (self.td.X,axis=0),(self.td.L,1))
        elif initialization_negative == 'mean_samples_and_features':
            self.W_neg = 0 * self.td.subclassMeans + np.mean (self.td.X)
        elif self.initialization_negative == 'random':
            self.W_pos = 0.01 * (2*np.random.rand(self.td.L, 784)-1)
            self.W_neg = 0.01 * (2*np.random.rand(self.td.L, 784)-1)
        else:
            self.W_neg = self.td.subclassMeans * (1 - self.initialization_negative)
        self.beta = self.init_beta

    def chooseBeta (self, noSamples):
        X = self.td.X
        N, dim = X.shape
    
        W_pos = self.W_pos
        W_neg = self.W_neg
        sum = 0
        shuffle = np.random.permutation(N)
        for j in range (noSamples):
            n = shuffle[j]
            x = X[n,:]
    
            d_X_WPos = W_pos - np.matlib.repmat(x, self.td.L, 1)
            d_pos_2 = 0.5 * np.sum(d_X_WPos ** 2, axis=1)
    
            d_X_WNeg = W_neg - np.matlib.repmat(x, self.td.L, 1)
            d_neg_2 = 0.5 * np.sum(d_X_WNeg ** 2, axis=1)
    
            d_neg_2_minus_d_pos_2 = d_neg_2 - d_pos_2
            sum += np.max(d_neg_2_minus_d_pos_2)
        self.beta = 1/ (sum / noSamples)

    def fit (self, reset=False):
        X = self.td.X
        N, dim = X.shape

        W_pos = self.W_pos
        W_neg = self.W_neg
        beta = self.beta
        theta = np.log(beta)

        if reset:
            W_pos = self.td.subclassMeans
            beta  = self.init_beta            
            if self.initialization_negative == 'mean':
                self.W_neg = np.tile(np.mean (self.td.X,axis=0),(self.td.L,1))
            elif self.initialization_negative == 'mean_samples_and_features':
                self.W_neg = 0 * self.td.subclassMeans + np.mean (self.td.X)
            else:
                self.W_neg = self.td.subclassMeans * (1 - self.initialization_negative)

        alpha = self.learning_rate
        iter = 0
        # target_iter = 10 * self.td.K * self.td.C / alpha
        target_iter = N

        for i in range (self.max_epochs):
            shuffle = np.random.permutation(N)
            for j in range (N):
                n = shuffle[j]
                x = X[n,:]

                iter = iter + 1
                if (iter >= target_iter):
                    # target_iter = 10 * self.td.K * self.td.C / alpha
                    alpha *= self.lr_decay_mult
                    self.beta_learning_rate *= self.lr_decay_mult
                    iter = 0

                d_X_WPos = W_pos - np.matlib.repmat(x, self.td.L, 1)
                #WPos_minus_WNeg = W_pos - W_neg
                #WPos_minus_WNeg_sign = np.sign(W_pos - W_neg)
                d_pos_2 = 0.5 * np.sum(d_X_WPos ** 2, axis=1)

                d_X_WNeg = W_neg - np.matlib.repmat(x, self.td.L, 1)
                #WNeg_minus_WPos = W_neg - W_pos
                #WNeg_minus_WPos_sign = np.sign(W_neg - W_pos)
                d_neg_2 = 0.5 * np.sum(d_X_WNeg ** 2, axis=1)

                d_neg_2_minus_d_pos_2 = d_neg_2 - d_pos_2
                out = beta * d_neg_2_minus_d_pos_2

                t = self.td.y[n]
                out = out - np.max(out)
                z = np.exp(out)
                denum = np.sum(z)
                if denum <= 10e-10:
                    denum = 10e-10
                z = z / denum

                tau = np.zeros(len(z))
                split_grad_pos = np.zeros(len(z))
                split_grad_neg = np.ones(len(z))

                idx1 = t * self.td.K
                idx2 = (t+1) * self.td.K
                denum = np.sum (z[idx1:idx2])
                if (denum <= 10e-10):
                    denum = 10e-10
                tau[idx1:idx2] = z[idx1:idx2] / denum
                split_grad_pos[idx1:idx2] = 1
                split_grad_neg[idx1:idx2] = 0

                dE_dwk = z - tau

                grad_pos = alpha * split_grad_pos * beta * -dE_dwk
                grad_neg = alpha * split_grad_neg * beta * dE_dwk

                W_pos = W_pos - grad_pos[:,np.newaxis] * d_X_WPos 
                W_neg = W_neg - grad_neg[:,np.newaxis] * d_X_WNeg 
                '''
                W_pos = W_pos - grad_pos[:,np.newaxis] * (
                        d_X_WPos +
                        self.l2_reg * WPos_minus_WNeg +
                        self.l1_reg * WPos_minus_WNeg_sign) \
                        - self.l1_reg_always * WPos_minus_WNeg_sign
                W_neg = W_neg - grad_neg[:,np.newaxis] * (
                        d_X_WNeg +
                        self.l2_reg * WNeg_minus_WPos +
                        self.l1_reg * WNeg_minus_WPos_sign) \
                        - self.l1_reg_always * WNeg_minus_WPos_sign
                '''

                dE_d_beta = dE_dwk * d_neg_2_minus_d_pos_2

                beta = beta - self.beta_learning_rate * sum(dE_d_beta)
                # theta = theta - self.beta_learning_rate * sum(dE_d_beta) * beta
                # beta = np.exp(theta)

        self.W_pos = W_pos
        self.W_neg = W_neg 
        self.beta = beta

        # As learning rate decays, we also decay the l1_reg_always
        self.l1_reg_always *= alpha / self.learning_rate
        self.learning_rate = alpha 


    def classifyByMaxClassifier(self, X_test):
        
        X_test = X_test
        N, dim = X_test.shape
        W_pos = self.W_pos
        W_neg = self.W_neg

        W_pos_2 = np.sum(W_pos**2, axis=1)
        W_neg_2 = np.sum(W_neg**2, axis=1)

        beta = self.beta
        A = beta * (W_pos - W_neg)
        b = - 0.5 * beta * (W_pos_2 - W_neg_2)
        out = X_test @ A.T + np.matlib.repmat(b,N,1)

        y_pred = self.td.label[out.argmax(axis=1)]
        return y_pred
                    

    def GenerateImagesOfWeights(self, width, height, color='color',
                        n_images=1, rows=None, cols=None, eps=0, weight='diff'):
        if weight == 'diff':
            A = self.W_pos - self.W_neg
        elif weight == 'pos':
            A = self.W_pos
        elif weight == 'neg':
            A = self.W_neg
        else:
            assert(0)

        n_features_per_image = rows * cols  # (A.shape[0] + 1) // n_images
        if rows == None or cols == None:
            cols = int(np.sqrt(A.shape[0] - 1)) + 1
            rows = (A.shape[0] + cols - 1) // cols
        images = []
        for picture in range(n_images):
            img = np.ones([rows * (height + 1), cols * (width + 1), 3])
            for nn in range(n_features_per_image):
                n = picture * n_features_per_image + nn
                if (n >= A.shape[0]):
                    continue
                j = nn // cols
                i = nn % cols
                idx2 = i * (height + 1)
                idx1 = j * (width + 1)
                T = max(-np.min(A[n, :]), np.max(A[n, :])) + eps
                if color == 'color':
                    arr_pos = np.maximum(A[n,:] / T, 0)
                    arr_neg = np.maximum(-A[n,:] / T, 0)
                    mcimg_pos = np.reshape(arr_pos, [height, width])  
                    mcimg_neg = np.reshape(arr_neg, [height, width])  
                    mcimg_oth = 0
                elif color == 'gray':
                    if weight == 'diff':
                        arr = A[n, :] / (2 * T) + 0.5
                    else:
                        arr = A[n, :] / T
                    arr = np.maximum(0,arr)
                    mcimg_pos = np.reshape(arr, [height, width])
                    mcimg_neg = mcimg_pos
                    mcimg_oth = mcimg_pos
                else:
                    assert(0)
                    
                img[idx1:idx1 + height, idx2:idx2 + width, 1] = mcimg_pos
                img[idx1:idx1 + height, idx2:idx2 + width, 0] = mcimg_neg
                img[idx1:idx1 + height, idx2:idx2 + width, 2] = mcimg_oth
            images.append(img)
        return images
