import numpy as np
from sklearn.datasets import load_iris

#Activation function - Using a leaky relu to avoid issues I had with dead neurons in relu and vanishing derivatives with sigmoid
def relu(z, deriv):
    if(deriv):  
        return 1 *(z>0) #gradient = 1 if z>0, 0 otherwise
    else:
        return z * ((z > 0)+0.2)

#Mean Squared Error loss function
def costfunction(predictedvalues, truevalues):
    return np.mean(np.power(predictedvalues - truevalues,2))

class Neuron:
    #Intialise the neuron with the weights choosen in the layer intialisation
    def __init__(self, weights, bias,input_size):
        if (len(weights.shape) == 1):
            self.weights = np.reshape(weights,(1,input_size))
        else:
            self.weights = weights
        self.bias = bias
        print("Intialised a neuron", weights, bias)
    
    #Performs feedforward on each neuron by multipling the inputs by the respective weights
    def feedforward(self, inputs):
        sum = 0
        sum = np.dot(self.weights,inputs.T) + np.repeat(np.reshape(self.bias,(len(self.bias),1)),len(inputs),axis=1)
        return np.array([sum[0],relu(sum[0],False)])

class NeuralLayer:

    #Creates correct number of neurons in the layer
    def __init__(self,input_size,size):
        self.neurons = []
        #Initalise the weights (multiplying by 0.1 to make sure they are small as there is a risk Relu will explode)
        self.weights = np.random.normal(size=(size,input_size))*0.1
        self.bias = np.zeros((size,1))
        for i in range(0,size):
            self.neurons.append(Neuron(self.weights[i],self.bias[i],input_size))

    #Gets values of neurons before (xvalues) and after (outputs) applying activation function 
    def performFeedForward(self,inputs):
        #TODO: This can defintely be optimised
        
        self.xvalues = np.array(list(map(lambda x:x.feedforward(inputs)[0],self.neurons)))
        self.outputs = np.array(list(map(lambda x:x.feedforward(inputs)[1],self.neurons)))
        return self.outputs

    #Updates neurons with recalculated weights and biases
    def update(self):
        for i in range(0,len(self.neurons)):
            if (len(self.weights.shape) == 1):
                self.neurons[i].weights = np.reshape(self.weights,(input_size,1))
            else:
                self.neurons[i].weights = self.weights
            self.neurons[i].bias =self.bias[i]
            #print("Updated a neuron", self.neurons[i].weights, self.neurons[i].bias)

#Performs backpropagation
def backpropagation(layers, truevalues, learning_rate):
    N = len(layers)
    m = len(truevalues[0])#Need to check this works if I make the output layer larger

    #Initalise gradients
    dZ = []
    dW = []
    db = []
    for i in range(0,N):
        dZ.append(np.zeros_like(layers[i].xvalues))
        dW.append(np.zeros_like(layers[i].weights))
        db.append(np.zeros_like(layers[i].bias))
    dZ = np.array(dZ)
    dW = np.array(dW)
    db = np.array(db)
    
    N = N -1

    #Calculate gradients of final layer
    dZ[N] = layers[N].outputs - truevalues
    dW[N] = np.dot(dZ[N],layers[N-1].outputs.T)/m
    db[N] = np.sum(dZ[N], axis = 1, keepdims = True)/m

    #Updates the weights of the neurons with gradient descent
    layers[N].weights = layers[N].weights - learning_rate*dW[N]
    layers[N].bias = layers[N].bias - learning_rate*db[N]
    layers[N].update()

    #Calculate the gradients of all the other layers iteratively
    for i in range(N,0,-1):
        dA = np.dot(layers[i].weights.T,dZ[i])
        dZ[i-1]= dA*relu(layers[i-1].xvalues,True)
        dW[i-1] = np.dot(dZ[i-1],layers[i-2].outputs.T)/m
        db[i-1] = np.sum(dZ[i-1], axis =1, keepdims = True)/m
        layers[i-1].weights = layers[i-1].weights - 0.01*learning_rate*dW[i-1]
        layers[i-1].bias = layers[i-1].bias - learning_rate*db[i-1]
        layers[i-1].update()

def train_network(inputs,true_output,learning_rate,layers,iterations):
    
    for j in range(0,iterations):
        data = []
        data.append(inputs)
        #Forward propogation
        for k in range(0,len(layers)):
            data.append(layers[k].performFeedForward(data[k]).T)
        
        predicted_value = data[len(layers)].T

        #Backward propogation
        backpropagation(layers, true_output,learning_rate)

        #Print current cost to give an idea of how quickly it is progressing
        if j % 100 == 0:
            print(costfunction(predicted_value,true_output))
        
    return predicted_value

if __name__ == "__main__":
    
    # load dataset
    X,y = load_iris(return_X_y = True)
    
    #Reshape it to match my code
    y = np.reshape(y, (1,(y.T).shape[0]))

    #normalise the data
    X = (X - np.mean(X, axis = 1, keepdims=True))/np.std(X,axis=1,keepdims=True)

    #split data into train and test set
    X_train = X[0:int(X.shape[0]*0.8),:]
    Y_train = y[:,0:int(y.shape[1]*0.8)]

    X_test = X[int(X.shape[0]*0.8):,:]
    Y_test = y[:,int(y.shape[1]*0.8):]
    
    
    inputs = X_train
    true_output = Y_train
    
    #Set learning rate and number of iterations
    learning_rate = 0.001
    iterations = 10000

    input_size = len(inputs[0])

    #Create layers!
    hidden_layer = NeuralLayer(input_size,4)
    output_layer = NeuralLayer(4,1)
    layers = [hidden_layer,output_layer]

    #Train network
    print(train_network(inputs,true_output,learning_rate,layers,iterations))

    #Test network
    inputs = X_test
    true_output = Y_test
    valuesOnLayer = hidden_layer.performFeedForward(inputs)
    predicted_value = output_layer.performFeedForward(valuesOnLayer.T)
    print(costfunction(predicted_value,true_output)) 
