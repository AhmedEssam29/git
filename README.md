
# Stock Market Prediction using Sentiment Analysis

Stock market prediction is widely acknowledged as an extremely challenging task.
As it influenced by a wide range of factors, including unpredictable elements such as political variables and the impact of social media platforms like Twitter, By considering these factors, we aim to address the existing challenges. 



![image](https://drive.google.com/uc?export=view&id=1hPsfWegCG_W3i7TLZdarArxuD9dVNd0w)
## Contents:
* [Problem Formulation](#Problem-Formulation)
* [Background](#Background)
  * [What's ARIMA(Autoregressive integrated moving average)?](#What's-ARIMA(Autoregressive-integrated-moving-average)?)
  * [Why CNN at time series?](#Why-CNN-at-time-series?)
  * [Why LSTM at time series?](#Why-LSTM-at-time-series?)
* [Methodology](#Methodology)
  * [The Project Structure](#The-Project-Structure)
  * [Dataset Collection](#Dataset-Collection)
  * [Data Preprocessing](#Data-Preprocessing)
  * [Sentiment Analysis](#Sentiment-Analysis)
  * [Data preparation](#Data-preparation)
    * [ARIMA Preparation](#ARIMA-Preparation)
    * [CNN-LSTM Preparation](#CNN-LSTM-Preparation)

  * [Modeling](#Modeling)
    * [CNN-LSTM Based approach](#CNN-LSTM-Based-approach)
    * [ARIMA (Autoregressive integrated moving average) approach](#ARIMA-(Autoregressive-integrated-moving-average)-approach)
* [Results](#Results)
* [Conclusion](#Conclusion)


  * [Applying the Sentiment analysis on the tweets](#Applying-the-Sentiment-analysis-on-the-tweets)
* [References and Related Work](#References-and-Related-Work)
* [Team Members](#Team-Members)

## problem formulation
The stock market is a focus for investors to maximize their potential
profits and consequently, the interest shown from the technical
and financial sides in stock market prediction is always on the rise.

However, stock market prediction is a problem known for its challenging
nature due to its dependency on diverse factors that affect 
the market, these factors are unpredictable and cannot be taken into
consideration such as political variables, and social media effects
such as twitter on the stock market.
## Background
### What's ARIMA(Autoregressive integrated moving average)?
> - ARIMA is a statistical analysis model that predicts the future of a variable with respect to time or for a better understanding of the time series dataset.
> - ARIMA is a generalization of Auto-Regressive, Moving Average, and integration terms.
> Those terms would be explained briefly in the following section.
> 1.	Auto-Regressive (AR): is a specific type of regression model, which means the current values are correlated with previous values in the time steps. To be more precise it’s partial Auto-correlation.
> <br><center>Y(t) = β1 + Ф1 Y(t-1) + Ф2 Y(t-2) +.. + Фp Y(t-p)</center>
> 	And, the (P) is the lagged order 
> 2.	Moving Average (MA): is analyzing the errors from the lagged observations and how they affect the current observation.
> <br><center>Y(t) = β2 + 𝟂1 𝞷(t-1) + 𝟂2 𝞷(t-2) +.. + 𝟂p 𝞷(t-p)</center>
> The 𝞷 terms are the errors observed, the 𝟂 is the weight of this error, and 𝟂 are calculated using a statistical correlation test.
> And, (q) represents the size of the moving average that has a significant impact on the current observation.<br>
> 3.	Integrated (I): the previous models can handle only the stationary time series dataset, which has a constant mean (μ), and variance (σ) without having seasonality. By taking the difference between consecutive timesteps, this transform will eliminate the trend and keep the mean constant. So, they defined (d) the order of differencing which means how many times we would apply the differencing process.
> <br><br>Till now, ARIMA can handle Non-stationary data with trends, but can’t handle the seasonality component, So, SARIMA (Seasonal Autoregressive Integrated Moving Average) was introduced as an extension adding more parameters to handle the seasonality.
> <br><center>![image](https://drive.google.com/uc?export=view&id=1K2-VRGdQZtigaXXw1AZXWMJJoUCJhanc)</center>
> <br >**SARIMA Parameters:**
> - p: non-seasonal or trend autoregressive (AR) order
> - d: non-seasonal or trend differencing<br>
> - q: non-seasonal or trend moving average (MA) order
> - P: seasonal AR order
> - D: seasonal differencing
> - Q: seasonal MA order
> - S: length of repeating seasonal pattern
> 
> To identify the values of AR and MA parameters we use ACF and PACF plots. Or by using grid search to go over all the combinations of parameters and chose the one that achieves the least loss according to a defined loss function such as Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) as both are penalized-likelihood information criteria.
And to identify the integrated parameters, they use a statistical test called the “Augmented Dickey-Fuller test”
> - Note: Due to the headache of identifying the model and its parameters, the python community has developed an automated library called “Auto Arima” that chooses the best model suitable for your dataset and identifies the best parameters to achieve the lowest loss.

## Why CNN at time series?
> **Convolutional Neural Networks (CNN):** <br> 
> The 1-d convolutional neural networks are used intensively in time series forecasting for a fixed window length, CNN has the ability to remove the noise and unrelated patterns along the time series data. Also, the convolutional and pooling layers both perform smoothing to the data instead of the traditional weighted average with require manual tuning.<br>
> ![image](https://drive.google.com/uc?export=view&id=1uX57pjUzbgduD-9nRgpuquUh7uOKGTXC)<br>
> Convolution learn from data through its filter’s weights update. CNN layers depend on the learning of the filters, so these filters after training will be able to detect the features.<br>
> ![image](https://drive.google.com/uc?export=view&id=1gDeLNYGX22hryuIUNbidhl5dadUUXXD0)<br>
> The pooling layer mainly smooths out and reduces the local noise by averaging the time series values.<br>
> ![image](https://drive.google.com/uc?export=view&id=1SID4bHM2Xuu2q4KPqNxYbRBnnYPjKQI9)<br>
> The data change after each layer:<br>
> ![image](https://drive.google.com/uc?export=view&id=1ySLDnYLasBa_rPSALGjutuRkmU6A2ZJl)<br>
> Figure 1: the change due to the CNN layer<br>
> ![image](https://drive.google.com/uc?export=view&id=1t57Teb0kQK5lHSm_3ZXugbvjc4Ziw9Fq)<br>
> Figure 2: the change due to the pooling layer<br>
## Why LSTM at time series?
 LSTM (Long Short-Term Memory) networks have the ability to learn long-term dependencies, especially, in a sequence prediction problem as time series, LSTM has outstanding performance on a large variety of data.
 ![image](https://drive.google.com/uc?export=view&id=1rX81D0b-WLr3MhiebNlPS9NIASQEHps1)
**Important Note:** The Bidirectional LSTM can learn in both direction forward and backward sequences, also the Bidirectional has complete information about all the points in the data.
## Methodology
### The Project Structure

## Methodology
### The Project Structure

![image](https://drive.google.com/uc?export=view&id=1hPsfWegCG_W3i7TLZdarArxuD9dVNd0w)

![image](https://i.ibb.co/N1kNsgk/my-project-1.png)


**As Shown at the above figure,** our project has thirteen stages:
- loading data 
- importing libraries
- apply Preprocessing 
- EDA and Sentiment analysis
- merging the data with yahoo finance data
- Modeling Using Two approaches
- TimeSeries forcast using:
  - ARIMA as Baseline
  - LSTM Deep learning architecture using 1D-CNN and BiCudaLSTM layers as our new approach
- Compare and evaluate our approach vs ARIMA
- Evaluate our approach against new data in real-time

- **Clustering** 
   - first we applied TF-IDF and BOW.
   - we applied K-MEAN & AGGLOMERATIVECLUSTERING

- **Classification**
  DNN & SVM & Randomforest
### Dataset Collection

The Stock Data:  We downloaded the stock data of FAANG companies ( Meta, Apple, Netflix, Amazon, Google) in Spreadsheet from Yahoo Finance website from September 30st, 2021 to September 30th, 2022.



![image](https://i.ibb.co/DbjhpdL/Picture1.png)### Data Preprocessing
> After scrapping the tweets, it was uncleaned text. So, we go through a cleaning process as shown:
> - Lowering the sentences.
> - Removing the attached links, hashtags, symbols, and numerical values.
> - Translate emojis into their meanings.
> - Tokenize sentences by Twitter.
> - Removing stop Words and punctuations
> - Applying lemmatization on each word.

![image](https://i.ibb.co/BGjBQpy/Picture2.png)


![image](https://i.ibb.co/BG8Kxvs/Picture3.png)
### Sentiment Analysis
> As the volume of tweets was huge, it was a challenge to label them manually, so the research team used 
a pre-trained model called “[twitter-slm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)” which was trained on 198 million tweets to get the polarity of each tweet.
The polarity score was converted into three values;1, 0, and -1 for each class as 1 represents a positive tweet, 0 represents a neutral tweet, and -1 represents a negative tweet.
![image](https://i.ibb.co/P4bKs29/Picture4.png)
> Due to each day having 350 tweets on average, it was aggregated by the average of polarities to get only a single number per day.### Data preparation
> the polarities were combined with the stock data to form one dataset to be used in time series models.
![image](https://i.ibb.co/X5qFbZX/Picture5.png)

## EDA

![image](https://i.ibb.co/y0XZ2Cr/Picture7.png)
![image](https://i.ibb.co/vJkTDq0/Picture8.png)
![image](https://i.ibb.co/TL6yBtZ/Picture9.png)
![image](https://i.ibb.co/RNQBnn7/Picture10.png)


## classification

![image](https://i.ibb.co/mDP8VBM/class.png)

## Clustering


![image](https://i.ibb.co/gSqyPTD/clust.png)
![image](https://i.ibb.co/SNdSHCt/clust-h.png)

#### **_ARIMA Preparation:_**
The input of the library should be in the form of y (predicted output) and X (features) as shown:
![image](https://i.ibb.co/JCJjywf/ARIMa.png)
Open, High, Low, Close, Adj Close, Volume, P_mean, P_sum, twt_count should be mapped to the Open and Close of the next data to be considered as the prediction.

and the testinf accuracy was 
![image](https://i.ibb.co/kmSnbtG/Sick.png)

#### **_ARIMA Preparation:_**
The input of the library should be in the form of y (predicted output) and X (features) as shown:
![image](https://i.ibb.co/JCJjywf/ARIMa.png)
Open, High, Low, Close, Adj Close, Volume, P_mean, P_sum, twt_count should be mapped to the Open and Close of the next data to be considered as the prediction.

and the testinf accuracy was 
![image](https://i.ibb.co/kmSnbtG/Sick.png)

#### **_ARIMA Preparation:_**
The input of the library should be in the form of y (predicted output) and X (features) as shown:
![image](https://i.ibb.co/JCJjywf/ARIMa.png)
Open, High, Low, Close, Adj Close, Volume, P_mean, P_sum, twt_count should be mapped to the Open and Close of the next data to be considered as the prediction.

and the testinf accuracy was 
![image](https://i.ibb.co/kmSnbtG/Sick.png)

#### **_CNN-LSTM Preparation:_**
In the time series problems with deep learning, the dataset should be reshaped to features and the target. The features consist of the number of previous days we look at and the target consists of the predicted future days and the number of predicted features. The dataset for this project is 1128 samples and the look-back days are 5 days, and the number of features is 7, so the features will be reshaped to (1118,5,7) and if the number of future days is 1 and the predicted features are 2 so the shape will be (1118,1,2).
The shape of features without the Twitter sentiment analysis is (1118,5,6). The last 5 days were dropped because they don’t have target value.
![image](https://i.ibb.co/M2DKqg8/LSTM.png)


**RESULTS**
![image](https://i.ibb.co/BTztx86/results.png)
### Conclusion
 While we compared the results of the two models we found:
 - The tweets have a positive impact as the accuracy increased.
 - Our model has a fast-learning curve, which achieved the Arima's accuracy in a few epochs, and without much optimization effort
 - expected to have a better performance if the model trained, again with more data and with more tuning to the parameter. taking into consideration that LSTM and CNN when getting more data would learn significantly.
