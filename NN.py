import numpy as np

input_size = 3
output_size = 3
hidden = 4

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden):
        self.w_in = np.random.randn(input_size,hidden)
        self.b_hidden = np.random.randn(hidden)
        self.w_out = np.random.randn(hidden,output_size)
        self.b_out = np.random.randn(output_size)

    def forward(self, input_value):
        self.h = np.dot(input_value, self.w_in) + self.b_hidden
        self.h = np.tanh(self.h)
        y = np.dot(self.h,self.w_out) + self.b_out
        return y
    
    def backward(self, x, y_true, y_pred, learning_rate=0.01):
        error = y_pred - y_true
        dW2 = np.dot(self.h.T, error)
        db2 = np.sum(error, axis=0)
        d_hidden = np.dot(error, self.w_out.T) * (1 - self.h ** 2)
        dW1 = np.dot(x.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0)
        self.w_in -= learning_rate * dW1
        self.b_hidden -= learning_rate * db1
        self.w_out -= learning_rate * dW2
        self.b_out -= learning_rate * db2

nn = NeuralNetwork(input_size, output_size, hidden)

x = np.random.randn(5, input_size)
y_true = np.random.randn(5, output_size)

max_epoch = 50000
epoch = 0

for epoch in range(max_epoch):
    y_pred = nn.forward(x)
    nn.backward(x, y_true, y_pred)
    epoch = epoch + 1
    if epoch % 10000 == 0:
        print(y_pred)

print("true is \n",y_true)
print("y_pred is \n", y_pred)