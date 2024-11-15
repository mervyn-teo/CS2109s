# Imitation
the idea for neural networks comes from human brains, where it is built up of millions and millions neurons.

This simulated neuron, we call it perceptron's. Perceptrons take in on to many values, and produce one value.
![[Pasted image 20241115214816.png|500]]
This is basically a composite function consisting of 2 parts.
1. summation function takes in multiple values, calculate its weighted sum 
2. and an activation function, taking in the weighted sum calculated by the summation function.
In formula form it looks like this:
$$\hat y = h_w(x)= g\left(\sum_{i=0}^nw_ix_i\right)$$
where $g()$ is the activation function.

# Training
Training this is very similar to what was done during linear regression.

The loop is as follows:

Initialize all $w_i$
Loop (until convergence or max steps reached) • 
	For each instance $(x^{(i)},y^{(i)})$, calculate $\hat y^(i) = h_w(x^{(i)})$ 
	Select one misclassified instance $(x^{(i)},y^{(i)})$
	Update weights $w \leftarrow w+\gamma(y^{(i)}-\hat y ^{(i)})x^{(i)}$

# Single-layer Neural Network
Depending on the needs we can change the activation functions. Such function works for both linear separable data and non-linear separable data
For example:
- For linear separable data: If we want to do data labeling, we can use a step function, where positive data is labeled as +1, negative as -1.
- For non-linear separable data: if we want to to simulate logic gates, where outputs are only 1 or 0, we can use sigmoid functions as activation functions, as its range is \[0,1\]
# Multi-layer Neural Network
As mention at the start, the point of Neural Network is to simulate human brain. The neurons in the human brain are all interconnected, forming a net like structure, therefore granting the name. At this stage, we can combine multiple perceptron in our design to achieve a more complex effect.

This is very simple, we can just connect singular inputs to perceptrons, or connect multiple outputs of perceptrons to other perceptrons, or any combination above. 

For example, we can chain multiple logic gate perceptrons together to form a more complex logic circuit. In this case, we chain AND, NOR and OR gates together to form a XNOR gate
![[Pasted image 20241116020150.png|500]]

We can also simulate $|x-1|$ using this method, we can use two ReLU function in combination.
![[Pasted image 20241116020354.png|500]]
These are some simplistic examples, real life usage tend to be more complex:![[Pasted image 20241116020912.png|500]]

with all these layers, calculating $\hat y$ uses a series of matrix multiplication:
![[Pasted image 20241116021126.png]]
And depending on the purpose of the neural network, we apply different perceptrons at the end of the network:
![[Pasted image 20241116021428.png]]![[Pasted image 20241116021445.png]]
by altering the activation function, we can use neural network to simulate other machine learning methods:

Using a sigmoid function, we can simulate a logistic regression:
![[Pasted image 20241116021930.png]]
Adding feature mapping functions and sign functions, we can simulate SVM:
![[Pasted image 20241116022020.png]]
We can also let the network figure the feature mapping out itself:
![[Pasted image 20241116022101.png]]





