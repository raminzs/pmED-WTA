import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.io as sio
import time
import matplotlib as mpl
import os.path
import pickle

from CCETrainingDataPreparation import TrainingData
from CompetitiveCrossEntropy_pmEDWTA import CompetitiveCrossEntropy_pmEDWTA

mpl.rcParams['figure.dpi']= 600

C  = 10
K = 6
d = 1024
maxVqIteration = 100
L = C * K
width = 28
height = 28
N = None
NTest = None

initialization_epsilon = 'random'
init_beta = 1 #Would be chosen automatically
learning_rate = 0.01 / init_beta
beta_learning_rate = 0.0001
# initialization_epsilon = 'mean'
# init_beta = 10 #Would be chosen automatically
# learning_rate = 0.01 / init_beta
# beta_learning_rate = 0.000001

max_epochs = 50
max_epochs_one = 1
lr_decay_mult  = 0.95
l1_reg = 0 #0.01
l1_reg_always = 0 #0.00001
exp_no = 1

XTrain = sio.loadmat ('MNIST/mnistTrainX')['MnistTrainX']
yTrain = sio.loadmat ('MNIST/mnistTrainY')['MnistTrainY']
yTrain = np.array(yTrain, dtype = int)
if N:
    XTrain = XTrain[:N,:]
    yTrain = yTrain[:N]

XTest = sio.loadmat ('MNIST/mnistTestX')['MnistTestX']
yTest = sio.loadmat ('MNIST/mnistTestY')['MnistTestY']
yTest = np.array (yTest, dtype = int)
if NTest:
    XTest = XTest[:NTest,:]
    yTest = yTest[:NTest]

#std_train = np.std(XTrain, axis = 0) + 10E-12
#std_train_mean = np.mean (std_train)
#std_train_max = np.max (std_train)

Z = np.max(XTrain)
XTrain = XTrain / Z
XTest = XTest / Z

for exp_no in range(5):

    np.random.seed(exp_no*10 + 1)

    td = TrainingData(XTrain, yTrain)
    filename = f'mnist_cce_pmEDWTA_clusters_k_{K}.pickle'
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            subclassMeans = pickle.load(file)
        td.setSubclasses(subclassMeans)
    else:
        clusAlg = sklearn.cluster.KMeans()
        clusAlg.max_iter = maxVqIteration
        start = time.time()
        td.findSubclasses(K, clusAlg)
        end = time.time()
        print ('Time for clustering: ', end - start)
        with open(filename, 'wb') as file:
            pickle.dump(td.subclassMeans, file)


    if not os.path.exists('./pickle'):
        os.mkdir('./pickle')
    if not os.path.exists('./png'):
        os.mkdir('./png')

    cce = CompetitiveCrossEntropy_pmEDWTA(td,
                                        learning_rate,
                                        lr_decay_mult,
                                        beta_learning_rate,
                                        max_epochs_one,
                                        initialization_epsilon,
                                        init_beta=init_beta)

    continue_train = False
    if continue_train:
        filename = f'./pickle/learned_pmEDWTA_centers_exp-{exp_no}.pickle'
        with open(filename, 'rb') as file:
            [W_pos, W_neg] = pickle.load(file)

        cce.W_pos = W_pos
        cce.W_neg = W_neg

    start = time.time()

    if cce.beta == -1:
        cce.chooseBeta(100)

    for epoch in range (max_epochs):
        if epoch > 0:
            cce.fit()
        print (cce.beta)
        yHat = cce.classifyByMaxClassifier(XTest)
        yHat = np.array(yHat, dtype='int')
        outVal = sklearn.metrics.accuracy_score(yTest, yHat)
        print('Test classification accuracy: ' + str(outVal))
    

    img = cce.GenerateImagesOfWeights(width, height, color='color', rows=C, cols=K, weight='diff')
    plt.axis('off')
    plt.imshow (img[0])
    fn = f'./png/diff_MNIST_cce_pmEDWTA_epoch-{epoch}_exp-{exp_no}_acc-{outVal}.png'
    plt.imsave(fn, img[0])
    #plt.show()

    img = cce.GenerateImagesOfWeights(width, height, color='color', rows=C, cols=K, weight='pos')
    plt.axis('off')
    plt.imshow (img[0])
    fn = f'./png/pos_MNIST_cce_pmEDWTA_epoch-{epoch}_exp-{exp_no}_acc-{outVal}.png'
    plt.imsave(fn, img[0])
    #plt.show()

    img = cce.GenerateImagesOfWeights(width, height, color='color', rows=C, cols=K, weight='neg')
    plt.axis('off')
    plt.imshow (img[0])
    fn = f'./png/neg_MNIST_cce_pmEDWTA_epoch-{epoch}_exp-{exp_no}_acc-{outVal}.png'
    plt.imsave(fn, img[0])
    #plt.show()

    end = time.time()
    print ('cca.fit took time: ', end - start)

    filename = f'./pickle/learned_pmEDWTA_centers_exp-{exp_no}.pickle'
    with open(filename, 'wb') as file:
        pickle.dump([cce.W_pos, cce.W_neg], file)
