
import numpy as np
from scipy import sparse
from itertools import product
from matplotlib import pyplot
import sys
import time

# Include path for tqdm package, not necessary for the content of the script.  It just displays a progress bar.
sys.path.append('/anaconda3/pkgs/conda-4.5.11-py37_0/lib/python3.7/site-packages/conda/_vendor')
from tqdm import tqdm

# Include path for graphnn.py
sys.path.append('/Users/jarodhart/Documents/Python/Neural Networks')
from graphnn import GraphNeuralNet

# Make a sample adjascency matrix
n = 2
m = 1
p = 15
N = n + m + p

row_col = [(i,j) for i,j in product(range(0,n),range(n,n+p))] \
            + [(i,j) for i,j in product(range(n,n+p),range(n+p,N))]
            
row = np.array([r for r,_ in row_col],dtype=np.int8)
col = np.array([c for _,c in row_col],dtype=np.int8)

# A is the adjascency matrix.  This is the only thing needed from here on.
A = sparse.csr_matrix((np.ones(len(row)),(row,col)),shape=(N,N))

# A visualization of the adjascency matrix
pyplot.figure(figsize=(8,8))
pyplot.imshow(A.toarray())
pyplot.title('Adjascency matrix for network',fontsize=18)
pyplot.ylabel('Origin neuron',fontsize=14)
pyplot.xlabel('Destination neuron',fontsize=14)
pyplot.show()



# Make a sample training set, just an XOR problem
X_train = np.array([[1,1],[-1,-1],[1,-1],[-1,1]])
y_train = np.array([[1],[1],[0],[0]])




# Make instance of GraphNeuralNet object
NN = GraphNeuralNet(A=A,init_weights=True,init_activation=True)


# Locations of nonzero elements in adjascency matrix
row,col = NN.A.nonzero()
# Set the learning rate
gamma = 1
# Set the LASSO regularization parameter
lam = .1
# Initialize error vector
errors = np.array([])
# Set batch size for stochastic gradient descent
#
weight_array = [NN.W]
batch_size = 50
for i in tqdm(range(1200)):
    # Sample from training set
    samp = np.random.choice(a=X_train.shape[0],size=batch_size)
    X_tr = X_train[samp,:]
    y_tr = y_train[samp,:]
    
    # Feedforward
    y,mu,dmu = NN.feedforward(X_tr)
    # Backprop
    delta = NN.backpropagation(y_tr,y,dmu)
    
    ## Update weights
    # Non-threshold weight updates classic, sans weight regulariztion
 #   dW_data = [gamma * sum(delta[:,c] * y[:,r]) / X_tr.shape[0] for r,c in zip(row,col)]
    dW_data = [gamma * (sum(delta[:,c] * y[:,r]) - lam * np.sign(NN.W[r,c])) / X_tr.shape[0] for r,c in zip(row,col)]
    # Non-threshold weight updates with LASSO weight regularization to encourage weight sparsity
    dW = sparse.csr_matrix((dW_data,(row,col)),shape=(N+1,N))
    
    dT_data = gamma * sum(delta,0) / X_tr.shape[0]
    dT = sparse.csr_matrix((dT_data,([NN.N for i in range(NN.N)],list(range(NN.N)))),shape=(N+1,N))
    
    NN.W = NN.W + dW + dT
    
    weight_array += [NN.W]
    # Feedforward training set and compute errors.  (Not necessary in most full implementations, just for this demo).
    y,mu,dmu = NN.feedforward(X_train)
    errors = np.append(errors,NN.batch_error(y_train,y) / X_train.shape[0])


# Plot errors
pyplot.figure(figsize=(6,4))
pyplot.plot(errors)
pyplot.ylim([0,1])
pyplot.show()

pyplot.figure(figsize=(8,6))
for r,c in zip(row,col):
    w = np.array([W[r,c] for W in weight_array])
    pyplot.plot(np.abs(w))
pyplot.show()

W = weight_array[-1]

row,col = W.nonzero()

[(r,c) for r,c in zip(row,col) if abs(W[r,c]) < .01]

[(r,c) for r,c in zip(row,col) if abs(W[r,c]) < .01 and r < NN.N]

A_ = sparse.csr_matrix(NN.A.toarray())
for r,c in zip(row,col):
    if abs(W[r,c]) < .01 and r < NN.N:
        A_[r,c] = 0
        W[r,c] = 0

pyplot.figure(figsize=(8,8))        
pyplot.title('Original Adjacency Matrix')
pyplot.imshow(NN.A.toarray())
pyplot.show()

pyplot.figure(figsize=(8,8))        
pyplot.title('LASSO Trimmed Adjacency Matrix')
pyplot.imshow(A_.toarray())
pyplot.show()

retained_weights = []
for i in range(NN.N):
    if np.sum(A_[i,:]) + np.sum(A_[:,i]) > 0:
        retained_weights += [i]





# LASSO-trimmed GraphNeuralNet object
NN_ = GraphNeuralNet(A=A_[retained_weights,:][:,retained_weights],init_weights=False,init_activation=True)
NN_.W = W[retained_weights + [NN.N],:][:,retained_weights]

NN.get_num_weights()
NN_.get_num_weights()

NN.N
NN_.N

pyplot.figure(figsize=(8,8))        
pyplot.title('Original Adjacency Matrix')
pyplot.imshow(NN.A.toarray())
pyplot.show()

pyplot.figure(figsize=(8,8))
pyplot.title('LASSO Trimmed Adjacency Matrix')
pyplot.imshow(NN_.A.toarray())
pyplot.show()




# Locations of nonzero elements in adjascency matrix
row,col = NN_.A.nonzero()
# Set the learning rate
gamma = 1
# Set the LASSO regularization parameter
#lam = .1
# Initialize error vector
errors = np.array([])
# Set batch size for stochastic gradient descent
#
batch_size = 50
for i in tqdm(range(1200)):
    # Sample from training set
    samp = np.random.choice(a=X_train.shape[0],size=batch_size)
    X_tr = X_train[samp,:]
    y_tr = y_train[samp,:]
    
    # Feedforward
    y,mu,dmu = NN_.feedforward(X_tr)
    # Backprop
    delta = NN_.backpropagation(y_tr,y,dmu)
    
    ## Update weights
    # Non-threshold weight updates classic, sans weight regulariztion
    dW_data = [gamma * sum(delta[:,c] * y[:,r]) / X_tr.shape[0] for r,c in zip(row,col)]
#    dW_data = [gamma * (sum(delta[:,c] * y[:,r]) - lam * np.sign(NN.W[r,c])) / X_tr.shape[0] for r,c in zip(row,col)]
    # Non-threshold weight updates with LASSO weight regularization to encourage weight sparsity
    dW = sparse.csr_matrix((dW_data,(row,col)),shape=(NN_.N+1,NN_.N))
    
    dT_data = gamma * sum(delta,0) / X_tr.shape[0]
    dT = sparse.csr_matrix((dT_data,([NN_.N for i in range(NN_.N)],list(range(NN_.N)))),shape=(NN_.N+1,NN_.N))
    
    NN_.W = NN_.W + dW + dT
    
    weight_array += [NN_.W]
    # Feedforward training set and compute errors.  (Not necessary in most full implementations, just for this demo).
    y,mu,dmu = NN_.feedforward(X_train)
    errors = np.append(errors,NN_.batch_error(y_train,y) / X_train.shape[0])

# Plot errors
pyplot.figure(figsize=(6,4))
pyplot.plot(errors)
pyplot.ylim([0,1])
pyplot.show()



class GraphNeuralNetTrainer():
    def __init__(self,**attributes):
        if 'SparseTrainer' in list(attributes.keys()):
            self.sparse_train = True
        if 'TrainingInputs' in list(attributes.keys()):
            self.X_train = attributes['TrainingInputs']
        if 'TrainingOutputs' in list(attributes.keys()):
            self.y_train = attributes['TrainingOutputs']
        if 'TestInputs' in list(attributes.keys()):
            self.X_test = attributes['TestInputs']
        if 'TestOutputs' in list(attributes.keys()):
            self.y_test = attributes['TestOutputs']
        if 'ValidationInputs' in list(attributes.keys()):
            self.X_validation = attributes['ValidationInputs']
        if 'ValidationOutputs' in list(attributes.keys()):
            self.y_validation = attributes['ValidationOutputs']
        
    def StochasticDescent(self,graphnet,LearningRate=.1,BatchSize=50,NumEpochs=10,TrackStats=False):
        
        row,col = graphnet.A.nonzero()
        
        if TrackStats:
            errors = np.array([])
            y,mu,dmu = graphnet.feedforward(self.X_train)
            errors = np.append(errors,graphnet.batch_error(self.y_train,y) / self.X_train.shape[0])
            weight_array = [graphnet.W]
        
        for epoch in tqdm(range(NumEpochs)):
            sample_order = list(np.random.choice(a=self.X_train.shape[0],size=self.X_train.shape[0],replace=False))
            samples = []
            while len(sample_order) > 0:
                samples += [sample_order[:BatchSize]]
                del(sample_order[:BatchSize])
            if samples[-1] == []:
                del(samples[-1])
            
            for samp in samples:
                X_tr = self.X_train[samp,:]
                y_tr = self.y_train[samp,:]
                
                y,mu,dmu = graphnet.feedforward(X_tr)
                delta = graphnet.backpropagation(y_tr,y,dmu)
                
                dW_data = [LearningRate * sum(delta[:,c] * y[:,r]) / X_tr.shape[0] for r,c in zip(row,col)]
                dW = sparse.csr_matrix((dW_data,(row,col)),shape=(graphnet.N+1,graphnet.N))
                
                dT_data = LearningRate * sum(delta,0) / X_tr.shape[0]
                dT = sparse.csr_matrix((dT_data,([graphnet.N for i in range(graphnet.N)],list(range(graphnet.N)))),shape=(graphnet.N+1,graphnet.N))
                
                graphnet.W = graphnet.W + dW + dT
                
            if TrackStats:
                weight_array += [graphnet.W]
                y,mu,dmu = graphnet.feedforward(self.X_train)
                errors = np.append(errors,graphnet.batch_error(self.y_train,y) / self.X_train.shape[0])
            
        if TrackStats:
            return graphnet,weight_array,errors
        else:
            return graphnet


NN_trainer = GraphNeuralNetTrainer(TrainingInputs=X_train,TrainingOutputs=y_train)

start_time = time.perf_counter()
NN,weight_array,errors = NN_trainer.StochasticDescent(graphnet=NN,LearningRate=.5,BatchSize=4,NumEpochs=1000, TrackStats=True)
run_time=time.perf_counter()-start_time
time.sleep(.5)
print('\nTracking stats:\t%.5f\n'%run_time)

time.sleep(.5)

start_time = time.perf_counter()
NN= NN_trainer.StochasticDescent(graphnet=NN,LearningRate=.5,BatchSize=4,NumEpochs=1000, TrackStats=False)
run_time=time.perf_counter()-start_time
time.sleep(.5)
print('\nNot tracking stats:\t%.5f'%run_time)


pyplot.plot(errors)


