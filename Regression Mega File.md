# Linear regression
Idea: based on the data provided, fit a line to the data. we do this by calculating the **mean square error** to all the points, and we are trying to minimise the Loss. This "loss" is determined by the loss function $J_{MSE}(W)=\frac{1}{2m} \sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})$ . 
 
## Disecting the loss function
$$J(W)=\frac{1}{2m} \sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})^2$$
The $h_w$ here refers to the function that produces a **predicted value of y**. This represents the result of $y=mx+c$ , where the input $W$ of the function $J_{MSE}$ is a matrix of weights that is used to evaluate y, $w_j$ is one instance of weights. This can be represented in the form:
$$J_{MSE}(W)=\frac{1}{2m} \sum^{m}_{i=1} (w_j(x^{(i)})-y^{(i)})^2$$
## Gradient descent
To minimise the loss, we use gradient descent. The idea is to: 
1. find the gradient of $J_{MSE}$ against $w_1$ (one instance of the current weights)
2. take the negation of the gradient, divide by counts of data points ($m$ in this case)
3. $w_{1(new)}$ is $w_{1(old)} + (-ve\space Gradient)$  

It is described in this formula (comes from partial differntiation of the [Loss Function](#Disecting%20the%20loss%20function): $$-\frac{\delta J_{MSE}(W)}{\delta W_i}=-\frac{1}{m}\sum^{m}_{i=1}(w_1(x^{(i)}-y^{(i)})(x^{(i)})$$
And the gradient descent is:
$$w_i\rightarrow w_i - \frac{\delta J_{MSE}(W)}{\delta W_i}$$
The effect will look somthing like this:![[Pasted image 20241114165450.png]]

## Learning Rate
Using the gradient descent function in its raw form might result in many problems. The change per iteration (calling it step from here onwards) might be too small, meaning it will take a very long time to reach the minimum; it may be too big, the $w_1$ overshoots to the other side of the curve, unable to reach the true minimum. 
![[Pasted image 20241114170049.png | 300]]  ![[Pasted image 20241114170037.png | 300]]

To conteract this, we will use the learning rate: $\gamma$ to control how much each step takes. 
The descent formula now looks like this:
$$w_i\rightarrow w_i - \gamma \frac{\delta J_{MSE}(W)}{\delta W_i}$$
### Careful of common mistakes
![[Pasted image 20241114171120.png | 500]]

Take note: The gradient descent is performed using **the whole of previous weights**, do not use the **updated weights** to caculate the next one. Calculate the whole set of weights before updating it.
## Setting the right $\gamma$ value
As mentioned in the [learning rate part](#Learning%20Rate), setting a non-suitable gamma value will result in the same problem. Setting a suitable $\gamma$ value is very important. However, this value is often determined by pure chance and intuiation (basically by luck).

## Varients of gradient descent

We basically vary which data we consider in our loss functions. In practice, we dont really use batch gradient descent because it is not very good in performace and it might get stuck in local minimas.
![[Pasted image 20241114171620.png]]
![[Pasted image 20241114171656.png]]
## Dealing with datas with different scales
Sometimes, when we handle data with different scales, since we use the same learning rate, the gradient descent may favour the data with larger scale, resulting in exploding gradients (basically, gradient descent does not perform as promised, it goes to infinity instead of decreasing to minima). 
Imagine the the case where we are trying to predict the value $y$ of houses with 2 features: $x_1$ amount of bed rooms, $x_2$ area of the house. In this case, if we base the learing rate according to the amount of bed rooms, it will easily go out of hand for the area of the house.
![[Pasted image 20241114172918.png]]
To mitigate this, we can do:
1. Normalise the data
	1. Mean normalization:
	2. Min-max scaling
	3. robust scaling
2. dedicate different learning rates to different features

## Dealing with Non-Linear Relationship
In this situation, we can use a function to convert our predicted value to a non-linear fashion, something like this:
$$h_w(x)=w_0+w_1f_1+w_2f_2+w_3f_3...+w_nf_n$$
where:
$$f_1 = x, f_2 = x^2...$$
## Normal equation
Someone as smart as you might think, why cant we just solve the equation and just calculate the minima? Indeed, it is possible.
We can do this $$\frac{\delta J_{MSE}(W)}{\delta W_i}=\frac{1}{m}\sum^{m}_{i=1}(w_1(x^{(i)}-y^{(i)})(x^{(i)})=0 \newline $$
and after a bunch of math, get:
$$w = (X^TX)^{-1}X^TY$$
Looks simple? but if you look at how transpose and inverse is calculated, it will result in $O(n^3)$ runtime. And, the $X^TX$ will need to be invertable.

# Logistic Regression

## Problem
We want to come up with some method to classify some known data, and use this classification method to classify more data. (e.g. cancer classification based on size). 
## Classification (2 classes)
To classify 2 classes of data, we will normally make use of a step function:
$$h_w(x)=
    \begin{cases}
      1 & \text{if x > n}\\
      0 & \text{otherwise}
    \end{cases}
$$
However this is discontinuous, we cannot use gradient descent methods to optimise it. We need to simulate this somehow.  Heres where sigmoid functions comes in. 
### sigmoid function
sigmoid function looks like this:
$$\sigma (z) = \frac{1}{1+e^{-z}}$$
graphically like this:
![[Pasted image 20241114181124.png | 500 ]] 

Which is similar to the step function, with ranges $(0, 1)$, and smooth seperation in between.

Using this function, we can now rewrite our function to be (for 1D data):
$$h_w(x)=\sigma(w_0+w_1x)$$
and out output can be treated as a probability, where $P(x=malignant) > \alpha$ then will be taken as 1, else 0. This $\alpha$ here refers to the decision boundary, it is the line where we determine which class a datapoint will belong to.  
![[Pasted image 20241114182108.png]]
For 2D data, we can simply add in another feature:
$$h_w(x)=\sigma(w_0+w_1x_1+w_2x_2)$$
and now, instead of a line as a decision boundary, it will be a plane:![[Pasted image 20241114182133.png]]

## Loss Function
We *could* use the same loss function as before, but now we use sigmoid function to predict our y.
$$J_{MSE}(W)=\frac{1}{2m} \sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})$$
$$J_{MSE}(W)=\frac{1}{2m} \sum^{m}_{i=1} (\frac{1}{1+e^{-(w_0+w_1x_1+w_2x_2)}}-y^{(i)})^2$$
but this is not ideal as: 
1. MSE is simply not designed for probabilities, the results may not be so meaningful.
2. It produces a non-linear, non-convex gradient, which does not garuentee convergence

How? we use Binary Cross Enthrophy (BCE) instead.
$$BCE(y,\hat y) = -ylog(\hat y) - (1-y)log(1- \hat y)$$
and the loss function:
$$J_{BCE} (w)= \frac{1}{m}\sum^m_{i=1}BCE(y^{(i)}, h_w(x^{(i)}))$$
where $h_w(x)$ is the [sigmoid function](#sigmoid%20function)
I didnt have the proof that sigmoid function is convex but *trust me bro*

The weight update is similar as in the one in linear regression:
$$w_i\rightarrow w_i - \gamma \frac{\delta J_{BCE}(W)}{\delta W_i}$$
## Dealing with Non-Linear Decision Boundary
Sometimes things are not as black and white as we wish, we may want to classify some data that looks like this:
![[Pasted image 20241114184256.png | 500]]
if you squint your eyes hard enough, you can see that the data roughly resembles a circle. And in this case, we will need a non-linear way of classification.
![[Pasted image 20241114184746.png | 500]]
since we roughly know that this is most likely seperatable by a cone, we can apply feature transformation using a quadratic function, similar to what we can do in linear regression.
$$h_w(x)=\sigma (w_0+w_1f_1+w_2f_2+w_3f_3...+w_nf_n)$$
## Multi-class Classification
There are 2 methods of comparison:
1. One vs all
2. One vs one

### One VS All
in this method, we predict by one class or not that class. in the case of cat, dogs and rabbits, we will first compare the given data point against cats, subsequently dogs, then rabbits. we will compare the probability generated by these options and it will be classified under the one with highest probability.
![[Pasted image 20241114193306.png]]
### One VS One
In this method, we compare it against 2 classes only, and pick the most frequnt classification. 
![[Pasted image 20241114193443.png]]
## Evaluation
### True Positive Rate & False Positive Rate
True positive rates:
$$TPR = TP / (TP+FN)$$
False positive rates:
$$FPR = FP / (FP+TN)$$
Where 
- TP - Number of positive labeled correctly as positive
- FN - number positives wrongly labeld as negative
- TN - number of negative correctly labeled as negative
- FP - number of positive wrongly labeled as negative

we can use this information to plot a graph, of TPR against FPR, called the Receiver Operator Characteristic (ROC) curve:
![[Pasted image 20241114194646.png]]
The area under the curve gives how good the model is, where 1 is the best, 0.5 is how a random choice model performs and 0 is the least good.

we can use this error function to evaluate our model:
$$J_d(h) = \frac{1}{N}\sum^N_{i=1}error(h(x^{(i)}),y^{(i)})$$
where $h$ is the model, $x^{(i)} , y^{(i)}$ comes from the dataset

## Dataset seperation
we can seperate our dataset into 3 different sets:
1. training set
2. validation set
3. test set
![[Pasted image 20241114200300.png]]
we use the training set to train our models. With different parameters, we can produce different models.
with our validation set, we can choose the best performing set.
Lastly with the test set, we can use it to assess chosen model with our test set data, such that it is consistent with our findings.

The graph of error against degree of polynomial should look something like this
![[Pasted image 20241114200733.png]]

The curve of $J_{Dval}$ and $J_{Dtest}$ looks like this because of how the model fits the data. the error decreases at the start because it underfits the data (features not enough to represent the data properly, making it higly biased), increases later because it overfits the data (too many features, makes it too specific to the training data, high varience)

representation of underfit to adequetly fitted to overfit:
![[Pasted image 20241114201143.png]]
## Hyperparameter Tuning
To choose good parameters, we can 
1. Pick hyperparameters e.g. degree of polynomials, learning rate 
2. Train model with the hyperparameters 
3. Evaluate model

some of the methods are:
- Grid search (exhaustive search) 
	- Exhaustively try all possible hyperparameters 
- Random search 
	- Randomly select hyperparameters 
- Successive halving 
	- Use all possible hyperparameters but with reduced resources 
	- Successively increase the resources with smaller set of hyperparameters 
- Bayesian optimization 
	- Use Bayesian methods to estimate the optimization space of the hyperparameters 
- Evolutionary algorithms 
	- Use evolutionary algorithms (e.g., genetic algo) to select a population of hyperparameters
# Support Vector Machines (SVM)
## Problem
When using logistic regression, it is very difficult to find a "just right" amount of features, we often will lead to overfitting

## Addressing overfit
2 ways of doing this:
1. Reduce the number of features	$$w_0+w_1x+w_1x^2+w_1x^3+w_1x^4\Rightarrow w_0+w_1x+w_1x^2$$
2. Regularisation
	- Reducing the magnitude of weights for certain feature
### Regularisation
To reduce the magnitude of weights, we can do it in the [loss function](#Disecting%20the%20loss%20function).
$$\begin{align}J(W)&=\frac{1}{2m} \left[\sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})^2\right] +1000w_3+1000w_4 \\ &= \frac{1}{2m} \left[\sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})^2\right]+\lambda\sum^n_{i=1}w_n^2\end{align}$$

in this case $\lambda$ is the **regulisation perimeter**. This is a constant that indicate how much you want to regularise this equation.