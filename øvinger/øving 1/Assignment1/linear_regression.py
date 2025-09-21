import numpy as np

class LinearRegression():
    
    def __init__(self, epochs=40, lr=1e-4):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.epochs = epochs
        self.lr = lr
        self.W = None
        self.b = None
        
        
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """

        n_features = X.shape[1] if X.ndim > 1 else 1
        self.W = np.zeros(n_features)
        self.b = 0.0
        losses = []
        
        for _ in range(self.epochs):
            y_pred_pre_sigmoid = X @ self.W + self.b
            y_pred = 1 / (1 + np.e ** (-y_pred_pre_sigmoid))
            loss = -y * np.log(y_pred) - (1-y) * np.log(1-y_pred)

            grads_W = (y_pred - y) @ X
            grads_b = (y_pred - y) 
            
            self.W -= grads_W.mean(axis=0) * self.lr
            self.b -= grads_b.mean(axis=0) * self.lr
            losses.append(loss.mean())

        return losses
            

    
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """

        y_pred_pre_sigmoid = X @ self.W + self.b
        y_pred = 1 / (1 + np.e ** (-y_pred_pre_sigmoid))
        return y_pred





