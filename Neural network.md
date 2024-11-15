# Imitation
the idea for neural networks comes from human brains, where it is built up of millions and millions neurons.

This simulated neuron, we call it perceptrons. Perceptrons take in on to many values, and produce one value.
![[Pasted image 20241115214816.png|500]]
This is basically a composite function consisting of 2 parts.
1. summaiton function takes in multiple values, calculate its weighted sum 
2. and an activation function, taking in the weighted sum calcualted by the summation function.
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

# 
