# Backpropagation
Python program to implement backpropagation :
Backpropagation is the backbone of artificial neural networks and entire premise of parametric learning is based upon
this central idea.In order to apply the gradient descent algorithm to optimize the parameters of 
our neural network,derivative of loss function with respect to the parameters of different layers are needed to be found 
and this is done by finding the derivative of the loss function with last output layer and then finding the derivative of
the loss with respect to hidden layers using the chain rule until we reach the paramerers of desired layers.
This process of finding gradient takes place from last layer to shallower layers hence called backpropagation.
Once we find out the derivative of loss wrt. to derised paramerter,we update our parameters using gradient descent.
This process continues untill we reaches a desire level of precision or we could run this for derised no of epochs.
Here i have provided a broad overview of backpropagation however program implement this algorithm in its entirety.Two working example
where this process is applied to XOR function and MNSIT dataset is also covered.I hope you will get to learn a lot from
asas a new comer. 
ne
##demo 
