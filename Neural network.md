# Imitating human brain
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
	For each instance $(x^{(i)},y^{(i)})$, calculate $\hat y^{(i)} = h_w(x^{(i)})$ 
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

And depending on the purpose of the neural network, we apply different perceptrons at the end of the network:
![[Pasted image 20241116021428.png]]![[Pasted image 20241116021445.png]]
by altering the activation function, we can use neural network to simulate other machine learning methods:

Using a sigmoid function, we can simulate a logistic regression:
![[Pasted image 20241116021930.png]]
Adding feature mapping functions and sign functions, we can simulate SVM:
![[Pasted image 20241116022020.png]]
We can also let the network figure the feature mapping out itself:
![[Pasted image 20241116022101.png]]

## Training multi-layer neural networks
As [mentioned above](#Training), we will need to calculate our predicted output $\hat y$, compare it with the real y value and update the data.

### Forward Propagation
with all these layers, calculating $\hat y$ uses a series of matrix multiplication:
![[Pasted image 20241116021126.png]]
### Back Propagation
To calculate and update the weights, we need to differentiate the whole chain of functions. To do so, we will need to apply the chain rule.
We can simplify this problem by focusing on one section of the operation:
![[Pasted image 20241117003425.png|]]
From the chain rule, we can deduce that:
$$\frac{\delta L}{\delta w_i}=\frac{\delta L}{\delta u_i}v_i$$
where $L$ represents the loss function.

Using this in the network, we sill start from the last perceptron and slowly work forwards, until all weights are updated.
![[Pasted image 20241117003724.png|500]]
Using this forward and backward pass concept, we can find and update all the weight with 1 forward pass and 1 backward pass. This method even works with non-linear activation functions.

## Convolution Neural Network (CNN)
Now with neural network, we can try to do something different. One real world application of neural network is to perform computer vision.
We can feed all the pixels into the network, and use different perceptrons to identify different part of the image. 
![[Pasted image 20241117015857.png]]
### Convolution Layer 
As you can imagine, this will result in a very large input data, which takes a lot of computation power. However if you observe, not all the pixels are required. We can use a **convolution layer** to "compress" the image, retaining the important information about the original input.

What this layer does it will apply a Kernel/Filter to the original image and take a weighted sum of everything. This retain information in areas that we may want to focus in, combining it to a singular value.
![[Pasted image 20241117021208.png]]
To calculate the convoluted inputs, we will slide the filter base on the specific step numbers, producing a feature map.
![[Pasted image 20241117021632.png]]
Sometime we might want to have different feature maps, we can repeat this process using other kernels/filters, producing a new feature map.
![[Pasted image 20241117021744.png]]
Common filters include  (not in syllabus, included it because its cool and i like cool things) :
Edge detection filters:
- **Sobel Filter:**
    - Horizontal Sobel:$$\begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$​​
    - Vertical Sobel: $$\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}​$$​​
- **Prewitt Filter:**
    - Horizontal Prewitt: $$\begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}​$$​​
    - Vertical Prewitt: $$\begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}$$
Blurring (Smoothing) Filters:
- Reduce noise and smooth out details in the image.
    - **Box Blur:**
        - Averages the pixel values in a local region.
        - Example: $$\frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$​​
    - **Gaussian Blur:**
        - Weighted average of neighboring pixels, emphasizing the center pixel.
        - Example: A 3x3 Gaussian kernel:$$\frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$​​

We also apply paddings to our inputs before convolutions to preserve data at the edge of the input. This also let us control how big the output will be.
![[Pasted image 20241117022740.png]]

some common types of padding includes:
- **Zero Padding:** Pads the input with zeros (most common).
- **Reflective Padding:** Pads the input by reflecting the boundary pixels.
- **Replicative Padding:** Pads by replicating the boundary pixels.

Colored Photos often have multiple channels depending file type (e.g. RGB, CMYK etc.), we can convolute these layers individually to form desired output channels. 
![[Pasted image 20241117022929.png]]


### Pooling layers
Pooling layers are layers that down sampling layers that reduces a feature map to ideal dimensions. 
Common pooling methods include:
- **Max Pooling**:
    
    - Takes the maximum value in each pooling window.
    - Useful for capturing sharp, distinctive features like edges.
    - Example (2x2 window): $$\text{Input: } \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}, \text{ Output: } 4$$
- **Average Pooling**:
    
    - Takes the average value of each pooling window.
    - Smooths the features, useful for tasks where precise locations are less important.
    - Example (2x2 window): $$\text{Input: } \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}, \text{ Output: } 2.5$$
- **Sum Pooling**
	- Sums up all the values in each pooling window.  
	- Useful for emphasizing the total magnitude of activations in a region.
	- Example (2x2 window):  
$$\text{Input: }\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},\text{Output: }10$$
In real applications we drink some absinthe and see a combination of all these techniques to produce a Convolutional Neural Network (CNN). Depending on what is required and what features you want the model to highlight, you will choose different combination of these techniques. 
![[Pasted image 20241117024130.png]]
# Sequential data
Sequential data are data that have a relationship in between them. e.g. video, audio, text etc. Since neural networks are so good, can we use neural network for sequential data?

The answer is yes, but we need to modify out methods to work with sequential data. 
## Recurrent Neural Networks (RNN)
The idea here is to include a special perceptron in our neural network, that takes in contextual information and produce outputs based on our second input and previous contextual information.
![[Pasted image 20241117033303.png]]
in this case it will look like this:
$$h_t=g^{[h]}\left((W^{[xh]})^Tx_t+(W^{[hh]})^Th_{t-1}\right)$$
and the corresponding output will look like this:
$$\hat y_{t}=g^{[h]}\left((W^{[hy]})^Th_t\right)$$
Since there is a recurring pattern, as a leetcode hard enjoyer, your brain must be tingling. We can which make it loop back to it self, siplifying the process. It looks something like this diagram:
![[Pasted image 20241117033906.png|300]]
note that this also means that the same weights are applied at each time step. This recurring method also mean that we can handle sentences of varying lengths.

Using different layers similar to what is done in [CNN](#Convolution%20Neural%20Network%20(CNN)), we can have Deep RNNs:
![[Pasted image 20241117034652.png]]

Then you might wonder, sometimes what comes after the sequence might also be important. In this case, we can use a Bidirectional RNN, We can run another RNN, in the opposite direction, and we concat results from both sides together.![[Pasted image 20241117035319.png]]



