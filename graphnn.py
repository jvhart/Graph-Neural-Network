
import numpy as np
from scipy import sparse


def sigmoid(s):
    return np.exp(s) / (1 + np.exp(s))

def Dsigmoid(s):
    return np.exp(-s) / (np.exp(-s) + 1)**2

def relu(s):
    return (s > 0) * s

def Drelu(s):
    return (s > 0) * 1

def linear(s):
    return s

def Dlinear(s):
    return 1

def tanh(s):
    return np.tanh(s)

def Dtanh(s):
    return 1 / np.cosh(s)

def gaussian(s):
    return np.exp(-s**2)

def Dgaussian(s):
    return -2 * s * np.exp(-s**2)

def square_error(x):
    return .5 * x**2

def Dsquare_error(x):
    return x

class GraphNeuralNet:
    
    def __init__(self,**params):
        if 'A' in list(params.keys()):
            self.A = params['A']
        else:
            self.A = None
        
        if 'init_weights' in list(params.keys()):
            init_weights = params['init_weights']
        else:
            init_weights = True
            
        if self.A != None:
            self = self.initialize_network()
            
        if init_weights:
            self = self.initialize_weights()
            
        if 'labeled' in list(params.keys()):
            self.labeled = params['labeled']
        else:
            self.labeled = self.output_neurons
            
        if 'init_activation' in list(params.keys()):
            init_activation = params['init_activation']
        else:
            init_activation = True
        
        if init_activation:
            self = self.initialize_activation()
            
        if 'init_error' in list(params.keys()):
            init_error = params['init_error']
        else:
            init_error = True
            
        if init_error:
            self = self.initialize_error()
            
            
    def initialize_network(self,**params):
        if 'A' in list(params.keys()):
            self.A = params['A']
            
        if 'labeled' in list(params.keys()):
            self.labeled = params['labeled']
            
        if 'init_weights' in list(params.keys()):
            init_weights = params['init_weights']
        else:
            init_weights = False
            
        self.N = max(self.A.shape)
        
        # Construct feedforward evaluation order
        ff_order = np.array([])
        ff_list = []
        all_indices = list(range(self.N))
        while len(all_indices) > 0:
            _,c = np.where(np.sum(self.A[[j for j in all_indices if not j in ff_order],:],0)[0,:] == 0)
            ff_list += [np.array([c_ for c_ in c if not c_ in ff_order])]
            ff_order = np.append(ff_order,ff_list[-1])
            all_indices = [i for i in all_indices if not i in ff_order]
        self.ff_list = ff_list
        self.input_neurons = ff_list[0]
        self.n = len(ff_list[0])
        self.ff_uplist = []
        for fl in self.ff_list:
            for i in fl:
                self.ff_uplist += [self.A[:,i].nonzero()[0]]
        
        # Construct the back propagation order
        bp_order = np.array([])
        bp_list = []
        all_indices = list(range(self.N))
        while len(all_indices) > 0:
            r,_ = np.where(np.sum(self.A[:,[j for j in all_indices if not j in bp_order]],1)[:,0] == 0)
            bp_list += [np.array([r_ for r_ in r if not r_ in bp_order])]
            bp_order = np.append(bp_order,bp_list[-1])
            all_indices = [i for i in all_indices if not i in bp_order]
        self.bp_list = bp_list
        self.output_neurons = bp_list[0]
        self.m = len(bp_list[0])
        self.p = self.N - self.m - self.n

        self.bp_list = bp_list
        self.bp_downlist = []
        for bp in self.bp_list:
            for i in bp:
                self.bp_downlist += [self.A[i,:].nonzero()[1]]
        self.bp_downlist = self.bp_downlist[::-1]
        
        if init_weights:
            self = self.initialize_weights()
        return self
    
    def initialize_weights(self,mode='random_normal'):
        if mode == 'random_normal':
            row,col = self.A.nonzero()
            row = np.append(row,self.N*np.ones(self.p + self.m))
            col = np.append(col,np.array(list(range(self.n,self.N))))
 #           weights = np.random.normal(size=len(row))
            weights = 2 * (np.random.rand(len(row)) - .5)
            self.W = sparse.csr_matrix((weights,(row,col)),shape=(self.N+1,self.N))
        return self
    
    def initialize_activation(self,mode='sigmoid'):
        if mode == 'sigmoid':
            self.activation = {i : sigmoid for i in range(self.N) if not i in self.ff_list[0]}
            self.Dactivation = {i : Dsigmoid for i in range(self.N) if not i in self.ff_list[0]}
        return self
    
    def initialize_error(self,mode='square_error'):
        if mode == 'square_error':
            self.error = square_error #{i : square_error for i in self.labeled}
            self.Derror = Dsquare_error #{i : Dsquare_error for i in self.labeled}
        return self

    def predict(self,x,threshold=.5):
        mu = np.zeros((x.shape[0],self.N))
        mu[:,self.ff_list[0]] = x
        y = np.zeros((x.shape[0],self.N))
        y[:,self.ff_list[0]] = x
        for i in range(1,len(self.ff_list)):
            for ii in self.ff_list[i]:
                up_y = y[:,self.ff_uplist[ii]]
                up_w = self.W[self.ff_uplist[ii],ii]
                mu[:,ii] = (up_y * up_w + np.ones((x.shape[0],1)) * self.W[self.N,ii]).T
                y[:,ii] = self.activation[ii](mu[:,ii])
        return 1*(y[:,self.output_neurons] > threshold)
    
    def predict_proba(self,x):
        mu = np.zeros((x.shape[0],self.N))
        mu[:,self.ff_list[0]] = x
        y = np.zeros((x.shape[0],self.N))
        y[:,self.ff_list[0]] = x
        for i in range(1,len(self.ff_list)):
            for ii in self.ff_list[i]:
                up_y = y[:,self.ff_uplist[ii]]
                up_w = self.W[self.ff_uplist[ii],ii]
                mu[:,ii] = (up_y * up_w + np.ones((x.shape[0],1)) * self.W[self.N,ii]).T
                y[:,ii] = self.activation[ii](mu[:,ii])
        return y[:,self.output_neurons]

    def feedforward(self,x):
        mu = np.zeros((x.shape[0],self.N))
        mu[:,self.ff_list[0]] = x
        y = np.zeros((x.shape[0],self.N))
        y[:,self.ff_list[0]] = x
        dmu = np.zeros((x.shape[0],self.N))
        for i in range(1,len(self.ff_list)):
            for ii in self.ff_list[i]:
                up_y = y[:,self.ff_uplist[ii]]
                up_w = self.W[self.ff_uplist[ii],ii]
                mu[:,ii] = (up_y * up_w + np.ones((x.shape[0],1)) * self.W[self.N,ii]).T
                y[:,ii] = self.activation[ii](mu[:,ii])
                dmu[:,ii] = self.Dactivation[ii](mu[:,ii])
        return y,mu,dmu
    
    def backpropagation(self,label,y,dmu,**params):
        if 'labeled' in list(params.keys()):
            self.labeled = params['labeled']
        delta = np.zeros(y.shape)
        for i,ii in zip(self.output_neurons,range(len(self.output_neurons))):
            delta[:,i] = - np.multiply(self.Derror( y[:,i] - label[:,ii] ), dmu[:,i])
        for i in range(1,len(self.bp_list)):
            for ii in self.bp_list[i]:
                down_delta = delta[:,self.bp_downlist[ii]]
                down_w = self.W[ii,self.bp_downlist[ii]].T
                term = (down_delta * down_w)
                for j in range(y.shape[0]):
                    delta[j,ii] = dmu[j,ii] * term[j]
        return delta

    def batch_error(self,label,y,**params):
        if 'labeled' in list(params.keys()):
            self.labeled = params['labeled']
        all_errors = self.error(y[:,self.labeled]-label) 
        return np.sum(all_errors)
    
    def get_num_weights(self):
        return int(np.sum(self.A) + self.p + self.m)

    def get_network_depth(self):
        return len(self.ff_list) - 1
    
    def update_topology(self,threshold=.01):
        
        row,col = self.A.nonzero()
        for i,j in zip(row,col):
            if abs(self.W[i,j]) < threshold and i < self.N:
                self.W[i,j] = 0
                self.A[i,j] = 0
        
        retained_weights = [i for i in self.input_neurons] 
        for i in range(self.N):
            if (np.sum(self.A[:,i]) > 0 and np.sum(self.A[i,:]) > 0):
                retained_weights += [i]

        retained_weights += list(self.output_neurons)
        
        for i in [j for j in range(self.N) if not j in retained_weights]:
            for k in np.where((self.A[i,:] > 0).toarray())[1]:
                self.W[self.N,k] += self.activation[k](self.W[self.N,i])
        
        inverse_map = {r : i for i,r in zip(range(len(retained_weights)),retained_weights)}        
        
        self.A = self.A[retained_weights,:][:,retained_weights]
        self = self.initialize_network(A=self.A,init_weights=False)
        self.W = self.W[retained_weights + [self.N],:][:,retained_weights]
        self.activation = {inverse_map[i] : self.activation[i] for i in retained_weights if not i in self.input_neurons} 
        self.Dactivation = {inverse_map[i] : self.Dactivation[i] for i in retained_weights if not i in self.input_neurons}
        self.labeled = self.output_neurons

        return self
    
    def update_check(self):
        print('Does this version control work?')
        return self

    def another_update_check(self):
        print('Does this version control work now?')
        return self





        
        