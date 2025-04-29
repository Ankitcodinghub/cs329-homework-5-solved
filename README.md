# cs329-homework-5-solved
**TO GET THIS SOLUTION VISIT:** [CS329 Homework#5 Solved](https://www.ankitcodinghub.com/product/cs329-solved-5/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;115926&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS329 Homework#5  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
&nbsp;

• Coding Homeworks. All coding assignments will be done in Jupyter Notebooks. We will provide a .ipynb template for each assignment. Your final submission will be a .ipynb file with your answers and explanations (you should know how to write in Markdown or LATEX). Make sure that all packages you need are imported at the beginning of the program, and your .ipynb file should work stepby-step without any error.

Question 1

Consider a regression problem involving multiple target variables in which it is assumed that the distribution of the targets, conditioned on the input vector x, is a Gaussian of the form p(t|x, w) = N(t|y(x, w), Σ)

where y(x, w) is the output of a neural network with input vector x and wight vector w, and Σ is the covariance of the assumed Gaussian noise on the targets.

(a) Given a set of independent observations of x and t, write down the error function that must be minimized in order to find the maximum likelihood solution for w, if we assume that Σ is fixed and known.

(b) Now assume that Σ is also to be determined from the data, and write down an expression for the maximum likelihood solution for Σ. (Note: The optimizations of w and Σ are now coupled.)

1

Question 2

The error function for binary classification problems was derived for a network having a logistic-sigmoid output activation function, so that 0 ≤ y(x, w) ≤ 1, and data having target values t ∈ {0, 1}. Derive the corresponding error function if we consider a network having an output −1 ≤ y(x, w) ≤ 1 and target values t = 1 for class C1 and t = −1 for class C2. What would be the appropriate choice of output unit activation function?

Hint. The error function is given by:

N

E yn)}.

Question 3

Can you represent the following boolean function with a single logistic threshold unit (i.e., a single unit from a neural network)? If yes, show the weights. If not, explain why not in 1-2 sentences.

A B f(A,B)

1 1 0

0 0 0

1 0 1

0 1 0

Question 5

Below is a diagram of a small convolutional neural network that converts a 13×13 image into 4 output values. The network has the following layers/operations from input to output: convolution with 3 filters, max pooling, ReLU, and finally a fullyconnected layer. For this network we will not be using any bias/offset parameters (b). Please answer the following questions about this network.

(a) How many weights in the convolutional layer do we need to learn?

(b) How many ReLU operations are performed on the forward pass?

(c) How many weights do we need to learn for the entire network?

(d) True or false: A fully-connected neural network with the same size layers as the above network (13 × 13 → 3 × 10 × 10 → 3 × 5 × 5 → 4 × 1) can represent any classifier?

(e) What is the disadvantage of a fully-connected neural network compared to a convolutional neural network with the same size layers?

Question 6

The neural networks shown in class used logistic units: that is, for a given unit U, if A is the vector of activations of units that send their output to U, and W is the weight vector corresponding to these outputs, then the activation of U will be (1 + exp(WTA))−1. However, activation functions could be anything. In this exercise we will explore some others. Consider the following neural network, consisting of two input units, a single hidden layer containing two units, and one output unit:

(a) Say that the network is using linear units: that is, defining W and and A as above, the output of a unit is C ∗ WTA for some fixed constant C. Let the weight values wi be fixed. Re-design the neural network to compute the same function without using any hidden units. Express the new weights in terms of the old weights and the constant C.

(b) Is it always possible to express a neural network made up of only linear units without a hidden layer? Give a one-sentence justification.

(c) Another common activation function is a theshold, where the activation is t(WTA) where t(x) is 1 if x &gt; 0 and 0 otherwise. Let the hidden units use sigmoid activation functions and let the output unit use a threshold activation function. Find weights which cause this network to compute the XOR of X1 and X2 for binary-valued X1 and X2. Keep in mind that there is no bias term for these units.

pre-Program Question

Answer the following questions in .ipynb file before beginning your program questions.

(a) You have an input volume that is 63 × 63 × 16, and convolve it with 32 filters that are each 7 × 7, using a stride of 2 and no padding. What is the output volume? (hint : the third dimension of each filter spans across the whole third dimension of the input)

(b) Suppose your input is a 300 × 300 color (RGB) image, and you are not using a convolutional network. If the first hidden layer has 100 neurons, each one fully connected to the input, how many parameters does this hidden layer have (including the bias parameters)?

(c) Using the following toy example, lets compute by hand exactly how convolutional layer works.

inputimage

filter1

filter2

Here we have a 5 × 5 × 1 input image, and we are going to use 2 different filters with size 3 × 3 × 1 and stride 1 with no padding as our first convolutional layer.

Compute the outputs and complete table. (hint: the output dimension is 3 × 3 ×

2)

Row Column Filter Value

1 1 1 –

1 1 2 –

1 2 1 –

2 1 1 –

(d) Let’s train a fully-connected neural network with 9 hidden layers, each with 20 hidden units. The input is 30-dimensional and the output is a scalar. What is the total number of trainable parameters in your network?

(e) State two advantages of convolutional neural networks over fully connected networks.

Program Question

For this question, refer to the Jupyter Notebook. You will be using PyTorch to implement a convolutional neural network the notebook will have detailed instructions. We will be using the fashion MNIST dataset for a classification task.

1. Convolutional Neural Network

2. Network Architecture and Implementation

This table describes the baseline architecture for the CNN. Please implement this architecture. You are, however, free to change the architecture as long as you beat the accuracy of this baseline.

Layers Hyperparameters

Convolution 1 Kernel = (5, 5, 16); Stride=1; Padding=2

ReLU 1 –

Maxpool 1 Kernel size=2

Convolution 2 Kernel = (5, 5, 32); Stride=1; Padding=2

ReLU 2 –

Max pool 2 Kernel size=2

Dropout Probability=0.5

Fully Connected Layer Output Channels=10; followed by Softmax

3. Accuracy

Reference. These questions are from websites, CMU and UPenn.
