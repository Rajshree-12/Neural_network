import numpy as np
#line 1:
def sigmoid(x):
    return 1/(1+np.exp(-x))
def derivative_sigmoid(x):
    return x*(1-x)
#line2:
x_in=np.array([[0,0,1,1],[1,0,0,1],[0,1,1,1]])
y_out=np.array([[0],[0],[1]])
print(x_in,'\n',y_out)
#line 3:
learning_rate=0.03
epochs=10000
hidden_layer=1
samples=np.size(x_in,0)
in_neurons=np.size(x_in,1)
out_neurons=np.size(y_out,1)
hd_neurons=4
print(in_neurons,out_neurons)
#line 4:
np.random.seed(101)
#weights and biases for 1st hidden layer:
w1=np.random.randn(hd_neurons,in_neurons)
b1=np.zeros((samples,hd_neurons))
#weights and biases for 2nd hidden layer:
w2=np.random.randn(out_neurons,hd_neurons)
b2=np.zeros((samples,out_neurons))
#line 5:
for i in range(epochs):
  #forward propagation:
  l0=x_in #1st layer
  z1=np.dot(l0,w1.T)+b1
  l1=sigmoid(z1) #2nd layer
  z2=np.dot(l1,w2.T)+b2
  l2=sigmoid(z2) #3rd layer
  y_predicted=l2
  #backprpagation:
  l2_error=(y_out-y_predicted)
  if (i%1000)==0:
     print("Epoch:" + str(np.mean(np.abs(l2_error))))
  l2_delta=l2_error*derivative_sigmoid(y_predicted)
  l1_error=l2_delta.dot(w2)
  l1_delta=l1_error*derivative_sigmoid(l1)
  #weights update:
  w2=w2+learning_rate*(l1.T.dot(l2_delta)).T
  w1=w1+learning_rate*(l0.T.dot(l1_delta))
#after training the data:
print('The predicted output:\n',y_predicted)
print('The original output:\n',y_out)
  
