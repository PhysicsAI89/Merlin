import numpy as np
class MerlinNetClassifier:
    """Tiny 1-hidden-layer NN (ReLU + sigmoid) with sklearn-like API."""
    def __init__(self, hidden=32, lr=0.01, epochs=200, batch_size=256, seed=42, l2=1e-4):
        self.hidden, self.lr, self.epochs, self.batch_size, self.seed, self.l2 = hidden, lr, epochs, batch_size, seed, l2
        self.W1=self.b1=self.W2=self.b2=None; self._fitted=False
    def _init_weights(self, d):
        rng = np.random.default_rng(self.seed)
        lim = (6/(d+self.hidden))**0.5
        self.W1 = rng.uniform(-lim, lim, size=(d,self.hidden)); self.b1 = np.zeros((1,self.hidden))
        lim2 = (6/(self.hidden+1))**0.5
        self.W2 = rng.uniform(-lim2, lim2, size=(self.hidden,1)); self.b2 = np.zeros((1,1))
    def _relu(self,x): return np.maximum(0,x)
    def _sigmoid(self,x): return 1/(1+np.exp(-x))
    def _forward(self,X):
        Z1 = X@self.W1 + self.b1; A1=self._relu(Z1); Z2=A1@self.W2 + self.b2; A2=self._sigmoid(Z2)
        return A1,A2
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y,float).reshape(-1,1)
        n,d = X.shape
        if self.W1 is None: self._init_weights(d)
        idx=np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0,n,self.batch_size):
                b = idx[s:s+self.batch_size]
                A1, A2 = self._forward(X[b])
                m=len(b)
                dZ2=(A2-y[b])/m; dW2=A1.T@dZ2 + self.l2*self.W2; db2=dZ2.sum(0,keepdims=True)
                dA1=dZ2@self.W2.T; dZ1=dA1*(A1>0); dW1=X[b].T@dZ1 + self.l2*self.W1; db1=dZ1.sum(0,keepdims=True)
                self.W1-=self.lr*dW1; self.b1-=self.lr*db1; self.W2-=self.lr*dW2; self.b2-=self.lr*db2
        self._fitted=True; return self
    def predict_proba(self,X):
        if not self._fitted: raise RuntimeError("MerlinNet not fitted.")
        X=np.asarray(X,float); _,A2=self._forward(X); return np.hstack([1-A2,A2])
    def predict(self,X): return (self.predict_proba(X)[:,1]>=0.5).astype(int)
