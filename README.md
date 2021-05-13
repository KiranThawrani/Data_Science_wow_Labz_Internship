# Data_Science_wow_Labz_Internship
Repository for learning data science
 # Learning summary for Task 1 
 # The data science process:
 A quick outline When a non-technical supervisor asks you to solve a data problem, the description of your task can be quite ambiguous at first. It is up to you, as the data scientist, to translate the task into a concrete problem, figure out how to solve it and present the solution back to all of your stakeholders. We call the steps involved in this workflow the “Data Science Process. ” This process involves several important steps: Frame the problem: Who is your client?What exactly is the client asking you to solve?How can you translate their ambiguous request into a concrete, well-defined problem?Collect the raw data needed to solve the problem: Is this data already available?If so, what parts of the data are useful?If not, what more data do you need?What kind of resources (time, money, infrastructure) would it take to collect this data in a usable form?Process the data (data wrangling): Real, raw data is rarely usable out of the box.

There are errors in data collection, corrupt records, missing values and many other challenges you will have to manage. You will first need to clean the data to convert it to a form that you can further analyze. Explore the data: Once you have cleaned the data, you have to understand the information contained within at a high level. What kinds of obvious trends or correlations do you see in the data?What are the high-level characteristics and are any of them more significant than others?Perform in-depth analysis (machine learning, statistical models, algorithms): This step is usually the meat of your project,where you apply all the cutting-edge machinery of data analysis to unearth high-value insights and predictions.

# History of Machine Learning

Today, machine learning algorithms enable computers to communicate with humans, autonomously drive cars, write and publish sport match reports, and find terrorist suspects. 
Deep dive into the history and learned how data and computers were tested every year to prove that AI will be a boon or threat to society.

Scientists believe a computer will never “think” in the way that a human brain does, and that comparing the computational analysis and algorithms of a computer to the machinations of the human mind is like comparing apples and oranges. Regardless, computers’ abilities to see, understand, and interact with the world around them is growing at a remarkable rate. And as the quantities of data we produce continue to grow exponentially, so will our computers’ ability to process and analyze — and learn from — that data grow and expand.

## The Great AI Awakening:
They told Hughes that 2016 seemed like a good time to consider an overhaul of Google Translate — the code of hundreds of engineers over 10 years — with a neural network.
The first story, the story of Google Translate, takes place in Mountain View over nine months, and it explains the transformation of machine translation.
When word began to spread, over the following weeks, that Google had introduced neural translation for Chinese to English, some people speculated that it was because that was the only language pair for which the company had decent results.
Take the case of image recognition, which tends to rely on a contraption called a “convolutional neural net.” (These were elaborated in a seminal 1998 paper whose lead author, a Frenchman named Yann LeCun, did his postdoctoral research in Toronto under Hinton and now directs a huge A.I. endeavor at Facebook.) The first layer of the network learns to identify the very basic visual trope of an “edge,” meaning a nothing (an off-pixel) followed by a something (an on-pixel) or vice versa.
He knew that various people in various places at Google and elsewhere had been trying to make neural translation work — not in a lab but at production scale — for years, to little avail.
Members of the Google Brain team in 2012, after their famous “cat paper” demonstrated the ability of neural networks to analyze unlabeled data.
Imagine if you could tell Google Maps, “I’d like to go to the airport, but I need to stop off on the way to buy a present for my nephew.” A more generally intelligent version of that service — a ubiquitous assistant, of the sort that Scarlett Johansson memorably disembodied three years ago in the Spike Jonze film “Her”— would know all sorts of things that, say, a close friend or an earnest intern might know: your nephew’s age, and how much you ordinarily like to spend on gifts for children, and where to find an open store.
(The researchers discovered this with the neural-network equivalent of something like an M.R.I., which showed them that a ghostly cat face caused the artificial neurons to “vote” with the greatest collective enthusiasm.) Most machine learning to that point had been limited by the quantities of labeled data.
A rarefied department within the company, Google Brain, was founded five years ago on this very principle: that artificial “neural networks” that acquaint themselves with the world via trial and error, as toddlers do, might in turn develop something like human flexibility.
For the Google Brain team, though, or for nearly everyone else who works in machine learning in Silicon Valley, that view is entirely beside the point.
Google Brain was the first major commercial institution to invest in the possibilities embodied by this way of thinking about A.I. Dean, Corrado and Ng began their work as a part-time, collaborative experiment, but they made immediate progress.
How Google used artificial intelligence to transform Google Translate, one of its more popular services — and how machine learning is poised to reinvent computing itself.
Speech recognition didn’t work very well until Brain undertook an effort to revamp it; the application of machine learning made its performance on Google’s mobile platform, Android, almost as good as human transcription.
The simplest description of a neural network is that it’s a machine that makes classifications or predictions based on its ability to discover patterns in data.
What the cat paper demonstrated was that a neural network with more than a billion “synaptic” connections — a hundred times larger than any publicized neural network to that point, yet still many orders of magnitude smaller than our brains — could observe raw, unlabeled data and pick out for itself a high-order human concept.
We’re still far from the construction of a network of that size, but Google Brain’s investment allowed for the creation of artificial neural networks comparable to the brains of mice.


## Python :

learned basics of python through video

# DATA WRANGLING

###A comprehensive introduction to data wrangling

**Definition:**
It is often the case with data science projects that you’ll have to deal with messy or incomplete data. The raw data we obtain from different data sources is often unusable at the beginning. All the activity that you do on the raw data to make it “clean” enough to input to your analytical algorithm is called data wrangling or data munging. If you want to create an efficient ETL pipeline (extract, transform and load) or create beautiful data visualizations, you should be prepared to do a lot of data wrangling.



**Data Wrangling With Pandas**

Learned how to handle missing data, filtering data ,grouoping data and do time series analysis

Conclusion: Data wrangling is an important part of any data analysis. You’ll want to make sure your data is in tip-top shape and ready for convenient consumption before you apply any algorithms to it. Data preparation is a key part of a great data analysis. By dropping null values, filtering and selecting the right data, and working with timeseries, you can ensure that any machine learning or treatment you apply to your cleaned-up data is fully effective.

# SUPERVISED MACHINE LEARNING

##LINEAR PREDICTIONS

 ## Linear Regression:


      Linear regression is used to predict a value (like the sale price of a house).
      Given a set of data, first try to fit a line to it.
      The cost function tells you how good your line is.
      You can use gradient descent to find the best line.

   ## Logistic Regression:

   The logistic function spits out a percentage
   The sigmoid function is used to constrain the output to between 0 and 1
   The cost is calculated using a log scale: the more wrong you were, the more you get penalized
   A decision boundary is a line you draw to separate your data into two different classes
   If you have multiple classes, use one-vs-all classification
   If you need non-linear classification, choose neural networks or support vector machines

   # scikit Library:
   Learned how to apply algorithms using scikit library
   
 ## NonLinear predictions:
 
   ## Random Forest:

     Random forests (RF) are basically a bag containing n Decision Trees (DT) having a different set of hyper-parameters and trained on different subsets of data. Let’s say I have 100 decision trees in my Random forest bag!! As I just said, these decision trees have a different set of hyper-parameters and a different subset of training data, so the decision or the prediction given by these trees can vary a lot. Let’s consider that I have somehow trained all these 100 trees with their respective subset of data. Now I will ask all the hundred trees in my bag that what is their prediction on my test data. Now we need to take only one decision on one example or one test data, we do it by taking a simple vote. We go with what the majority of the trees have predicted for that example.


## Clustering

###### K Means Clustering ####

Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.
The way kmeans algorithm works is as follows:
Specify number of clusters K.
Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
Compute the sum of the squared distance between data points and all centroids.
Assign each data point to the closest cluster (centroid).
Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.

Learned various concepts and python prgramming that helps to grow in machine Learning
