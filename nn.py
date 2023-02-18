import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from keras.datasets.mnist import load_data

class MLP():
    
    def __init__(self, din, dout):
        self.W = (2 * np.random.rand(dout, din) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        self.b = (2 * np.random.rand(dout) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        
    def forward(self, x): # x.shape = (batch_size, din)
        self.x = x # Storing x for latter (backward pass)
        return x @ self.W.T + self.b

    def backward(self, gradout):
        self.deltaW = gradout.T @ self.x
        self.deltab = gradout.sum(0)
        return gradout @ self.W
    
class SequentialNN():
    
    def __init__(self, blocks: list):
        self.blocks = blocks
        
    def forward(self, x):
        
        for block in self.blocks:
            x = block.forward(x)
  
        return x

    def backward(self, gradout):
        
        for block in self.blocks[::-1]:
            gradout = block.backward(gradout)
            
        return gradout

class ReLU():
    
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, gradout):
        new_grad = gradout.copy()
        new_grad[self.x < 0] = 0.
        return new_grad
    
class LogSoftmax():
    
    def forward(self, x):
        self.x = x
        return x - logsumexp(x, axis=1)[..., None]
    
    def backward(self, gradout):
        gradients = np.eye(self.x.shape[1])[None, ...]
        gradients = gradients - (np.exp(self.x) / np.sum(np.exp(self.x), axis=1)[..., None])[..., None]
        return (np.matmul(gradients, gradout[..., None]))[:, :, 0]
    
class NLLLoss():
    
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        
        loss = 0
        for b in range(pred.shape[0]):
            loss -= pred[b, true[b]]
        return loss

    def backward(self):
        din = self.pred.shape[1]
        jacobian = np.zeros((self.pred.shape[0], din))
        for b in range(self.pred.shape[0]):
            jacobian[b, self.true[b]] = -1

        return jacobian # batch_size x din
    
    def __call__(self, pred, true):
        return self.forward(pred, true)
    
class Optimizer():
    
    def __init__(self, lr, compound_nn: SequentialNN):
        self.lr = lr
        self.compound_nn = compound_nn
        
    def step(self):
        
        for block in self.compound_nn.blocks:
            if block.__class__ == MLP:
                block.W = block.W - self.lr * block.deltaW
                block.b = block.b - self.lr * block.deltab
                
def train(model, optimizer, trainX, trainy, loss_fct = NLLLoss(), nb_epochs=14000, batch_size=100):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):

        # Sample batch size
        batch_idx = [np.random.randint(0, trainX.shape[0]) for _ in range(batch_size)]
        x = trainX[batch_idx]
        target = trainy[batch_idx]

        prediction = model.forward(x) # Forward pass
        loss_value = loss_fct(prediction, target) # Compute the loss
        training_loss.append(loss_value) # Log loss
        gradout = loss_fct.backward()
        model.backward(gradout) # Backward pass

        # Update the weights
        optimizer.step()
    return training_loss
  
if __name__ == "__main__": 
    # Load and process data
    (trainX, trainy), (testX, testy) = load_data()
    trainX = (trainX - 127.5) / 127.5
    testX = (testX - 127.5) / 127.5
    trainX = trainX.reshape(trainX.shape[0], 28 * 28)

    mlp = SequentialNN([MLP(28*28, 128), ReLU(), 
                        MLP(128, 64), ReLU(), 
                        MLP(64, 10), LogSoftmax()])
    optimizer = Optimizer(1e-3, mlp)

    training_loss = train(mlp, optimizer, trainX, trainy)

    # Compute test accuracy
    accuracy = 0
    for i in range(testX.shape[0]):
        prediction = mlp.forward(testX[i].reshape(1, 784)).argmax()
        if prediction == testy[i]: accuracy += 1
    print('Test accuracy', accuracy / testX.shape[0] * 100, '%')
