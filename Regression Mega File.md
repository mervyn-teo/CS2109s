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

## Picking the right features
When using logistic regression, it is very difficult to find a "just right" amount of features, we often will lead to overfitting

## Addressing overfit
2 ways of doing this:
1. Reduce the number of features	$$w_0+w_1x+w_1x^2+w_1x^3+w_1x^4\Rightarrow w_0+w_1x+w_1x^2$$
2. Regularisation
	- Reducing the magnitude of weights for certain feature
### Regularisation
To reduce the magnitude of weights, we can do it in the [loss function](#Disecting%20the%20loss%20function).
$$\begin{align}J(W)&=\frac{1}{2m} \left[\sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})^2\right] +1000w_3^2+1000w_4^2 \\ &= \frac{1}{2m} \left[\sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})^2\right]+\lambda\sum^n_{i=1}w_n^2\end{align}$$

in this case $\lambda$ is the **regulisation perimeter**. This is a constant that indicate how much you want to regularise this equation. since the weights. There are multiple reasons to square the weights, but the main reason is for mathematical convenience while partial differentiation later.
### Modified Gradient Descent (linear regression)
in linear regression we subsitute $h_w$ as $h_w (x) := w^Tx$
we can futher modify this equation to use for gradient descent, we can divide $\lambda$ by $2m$ because both are constants.
$$\begin{align}J(w)&= \frac{1}{2m} \left[\sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})^2\right]+\frac{\lambda}{2m}\sum^n_{j=1}w_n^2 \\ &=\frac{1}{2m} \left[\sum^{m}_{i=1} (h_{w}(x^{(i)})-y^{(i)})^2+ \lambda\sum^n_{j=1}w^2_j\right]\end{align}$$
After partial differentiation and using it as gradient descent formula:
$$w_i\rightarrow w_i - \gamma\frac{1}{m}\sum^m_{i=1}(h_w(x^{(i)})-y^{(i)})x_j^{(i)}-\gamma\frac{\lambda}{m}w_j$$
### Normal Equation
Using this we can come up with a normal equation [similar to before](#Normal%20equation). Using intense big brain math we can rearrange the differentiated equation to this form:

$$w=(X^TX+\lambda\begin{bmatrix}  
0 & 0 & &0\\  
0 & 1 && 0\\
&&...\\
0 & 0 & &1

\end{bmatrix})^{-1}X^TY$$
Which when $\lambda > 0$, it works even when $X^TX$ is not invertable!
### Modified Gradient Descent (logistic regression)
similar to[ linear regression](#Modified%20Gradient%20Descent%20(linear%20regression)), we can do the same thing:
$$\begin{align}J(w)=\frac{1}{2m} \left[\sum^{m}_{i=1} y^{(i)}log\space h_{w}(x^{(i)})+(1-y^{(i)})log\left(1-h_w(x^{(i)})\right)+ \lambda\sum^n_{j=1}w^2_j\right]\end{align}$$
$$w_i\rightarrow w_i - \gamma\frac{1}{m}\sum^m_{i=1}(h_w(x^{(i)})-y^{(i)})x_j^{(i)}-\gamma\frac{\lambda}{m}w_j$$
where:
$$h_w (x) := \frac{1}{1+e^{-w^Tx}}$$
# Support Vector Machines (SVM)
SVM takes another approach to labeling data compared to logistic regression. The main idea behind SVM is we want to find a **margin**, that sperates different data point. We also want this margin to be **as large as possible**. 
![[Pasted image 20241115175745.png|500]]
From this margin, we will derive the equation of the hyperplane (aka our model) that helps us label our datapoints. The green dotted line indicates the linear equation $w \cdot x + b$ , it is a hyperplane seperating the elements. Those elements above the plane will be positive, below will be negative. (basically $w\cdot c+b\geq 0, \text{then +}$). in 2d, it is a line, while in 3d it is a plane, that can take any orientation.

This is a 3D graph with 3 features for better visualisation.
![[output.png|500]]

Our aim now is to find the equation of the hyperplane. We will first label the data. After labelling, the data will the values of +1 and -1, the graph will look something like this:
![[Pasted image 20241115182714.png|500]]
Form this, we can derive these equations:
$$\begin{align}
w\cdot x^+ +b&\geq 1\\
w\cdot x^- +b&\leq -1
\end{align}$$
combining them together using $$y^{(i)}=    \begin{cases}
      +1 & \text{for + samples}\\
      -1 & \text{for - samples}
    \end{cases}$$
We can derive:
$$\begin{align}
	y^{(i)}(w\cdot x^{(i)}+b)&\geq1\\
	y^{(i)}(w\cdot x^{(i)}+b)-1&\geq0\\
	y^{(i)}(w\cdot x^{(i)}+b)-1&=0, \text{for all x on margin}\\
	(w\cdot x^{(i)}+b)&=\frac{1}{y^{(i)}}
\end{align}$$
Also, we can rearrange $x^+,x^-$ to:
$$\begin{cases}
      w\cdot x^+ =1-b\\
      w\cdot x^- =-1-b
\end{cases}$$Since $w$ is the direction vector of the hyper plane, we can use some vector magic to calculate the margins.
![[Pasted image 20241115185651.png|500]]
then what we need to do is very clear now, we need to somehow find $max\frac{2}{||w||}$ while ensuring $y^{(i)}(w\cdot x^{(i)}+b)-1=0$
with some mathematical reduction, you will realise this problem is equivilent to:
$$\begin{align} 
	max\frac{2}{||w||} \\ 
	max\frac{}{||w||} \\
	min{||w||}\\
	min\frac{1}{2}{||w||^2}
\end{align}$$
and rearranging $y^{(i)}(w\cdot x^{(i)}+b)-1=0$:
$$b=y^{(i)}-w\cdot x^{(i)}$$
### Lagrange
Using Lagrange function (voodoo magic) we can get:
$$\mathop{max}_{a\geq 0}\sum _ia^{(i)}-\frac{1}{2}\sum_i\sum_ja^{(i)}a^{(j)}y^{(i)}y^{(j)}x^{(i)}\cdot x^{(j)}$$
## Soft-Margin
What happens when the data is not linearly seperable?
![[Pasted image 20241115192050.png|500]]
In this case we introduce some slack variables( $\xi$ ), to allow certain data to be misclassified.
$$\begin{align}
w\cdot x^+ +b&\geq 1-\xi^{(i)}\\
w\cdot x^- +b&\leq -1+\xi^{(i)}
\end{align}$$
our objective will become:
$$min\frac{1}{2}{||w||^2}+C\sum_i\xi^{(i)}$$
and classification condition will become
$$\xi^{(i)}\geq1-y^{(i)}(w\cdot x^{(i)}+b)$$
Combining them:
$$J(w,b)=\frac{1}{2}||w||^2+C\sum_imax\{0,1-y^{(i)}(w\cdot x^{(i)}+b)\}$$
### Disaster Strikes
What happens if the data is **truley non linear seperable**?
![[Pasted image 20241115193142.png|500]]
instead of considering $x$ on its own, we can use a transformation function ( $\phi$ ) to change the features to something we can work with. 
$$\phi(x_{1...i}) = [x_1,x_2,...,x_i,x_1^2,x_2^2,...,x_i^2,x_i^i]^T$$

replacing the features in our [magic function](#Lagrange):
$$\mathop{max}_{a\geq 0}\sum _ia^{(i)}-\frac{1}{2}\sum_i\sum_ja^{(i)}a^{(j)}y^{(i)}y^{(j)}\phi(x^{(i)})\cdot \phi(x^{(j)})$$
This, however, may break the nicely crafted magic spell, since $\phi$ changes the dimentions of x, it produces incredible amounts of terms. 
### Kernels
To reduce the number of terms we can use the kernel trick, transforming $\phi$ into something we can work with more comfortbly.
$$K(u,v)=\phi(u)\cdot\phi(v)=(u\cdot v)^d$$
this is not the only kernel we have, we also have Gaussian (RBF) Kernel:
$$K(u,v)=e^{-\frac{||u-v||^2}{2\sigma^2}}$$
now the [magic function](#Lagrange) looks like this:
$$\mathop{max}_{a\geq 0}\sum _ia^{(i)}-\frac{1}{2}\sum_i\sum_ja^{(i)}a^{(j)}y^{(i)}y^{(j)}K(x^{(i)}, x^{(j)})$$
and the decision rule becomes:
$$\sum_ia^{(i)}\hat y^{(i)}K(x^{(i)},x)+b\geq0, then \space+$$





