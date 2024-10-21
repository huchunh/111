# Sleep Onset and Wakeup Detection

# 1.Introduction 
Sleep is a fundamental biological need that impacts cognition and behavior, with specific effects on the regulation of mood, attention, memory, and emotion[1]. Sleep affects everything from child development to cognitive functioning. Even so, research into sleep has proved challenging, due to the lack of naturalistic data capture alongside accurate annotation. If data science could help researchers better analyze wrist-worn accelerometer data for sleep monitoring, sleep experts could more easily conduct large-scale studies of sleep, thus improving the understanding of sleep's importance and function. Machine learning techniques have been proposed for the analysis of sleep and identification of sleep stages. Applying ML techniques improved the classification accuracy of actigraphy-physical activity and sleep data in children[2]. With improved tools to analyze sleep data on a large scale, researchers can explore the relationship between sleep and mood/behavioral difficulties. This knowledge can lead to more targeted interventions and treatment strategies.

# 2.Literature Reviews 
In the previous studies, there are many machine learning approaches for the sleep researches and here are some introductions about them. These are really important references for our project.
Shahnawaz Qureshi and his team have studied on Evaluate Different Machine Learning Techniques for Classifying Sleep Stages on Single-Channel EEG. After getting the raw signals and preprocessing them, they focus four statistical parameters including standard deviation (Std), mean, variance and median, which are known as time domain features for EEG signals. For the Machine Learning Algorithm, they applied for Random Forest (RF), Support Vector Machine (SVM) and Bagging classifier. According to the results, they found that the overall accuracy obtained (97.73%) from RF classifier, (93.28%) from Bagging classifier and  (81.02%) from  Support Vector Machine (SVM).

In addition to these machine learning methods that we already have, there are also some researches that developed new models.
Xinyue Li and her team develop a novel machine learning unsupervised algorithm for sleep/wake identification using actigraphy, which is based on a two-state Hidden Markov Model (HMM). Their model is productive and has several advantages: Firstly, this is an unsupervised algorithm that does not require training data. Secondly, it can be directly applied to datasets from different devices and populations because it is data-driven. Last bu not least, the HMM based algorithm takes into account individual variations by analyzing each individual actigraphy separately.

HTWG Konstanz and his team  presented a implementation of deep learning methods for sleep stage detection by using three signals that can bel measured in a non-invasive way: heartbeat signal, respiratory signal, and movement signal. Since signals are measurements taken during the time, the problem is seen as time-series data classification. For detecting four sleep stages: REM (Rapid Eye Movement), Wake, Light sleep (Stage 1 and Stage 2), and Deep sleep (Stage 3 and Stage 4), the accuracy of the model is 55%, and F1 score is 44%.  For five stages: REM, Stage 1, Stage 2, Deep sleep (Stage 3 and 4), and Wake, the model gives an accuracy of 40% and F1 score of 37%.


# 3.Problem Definition 
Develop a model trained on wrist-worn accelerometer data in order to determine a person's sleep state. 
A single sleep period must be at least 30 minutes in length.
A single sleep period can be interrupted by bouts of activity that do not exceed 30 consecutive minutes.
No sleep windows can be detected unless the watch is deemed to be worn for the duration .
The longest sleep window during the night is the only one which is recorded.
If no valid sleep window is identifiable, neither an onset nor a wake-up event is recorded for that night.

# 4.Data Description 
The dataset comprises about 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wake-up, the end of sleep. Each series is a continuous recording of accelerometer data for a single subject spanning many days. Training series data includes series_id, step, timestamp, anglez, enmo. Every step represents a time. The onset and wake-up is recorded in step.
Database: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data. 



# 5.Proposed Methods and Experimental Results
We are working with time series data featuring two primary variables: anglez and enmo. These measure the angle between the body and arm and the body's movements, respectively. Given the predictive nature of our problem, initial attempts were made using traditional data mining algorithms such as regression, logistic regression, SVM, and naïve Bayes. However, these methods yielded unsatisfactory results for several reasons.

Firstly, our objective is to predict a specific time point, such as the moment a person wakes up (e.g., 23:05:40). Our data, recorded every five seconds, results in 720 data points per hour and thousands over a full day. Predicting a precise time point in this context is challenging. Additionally, relying solely on two features fails to capture time-sensitive information, rendering the prediction less accurate.

To address these challenges, we propose a two-step approach. This involves initially training a model on a simpler task, using the insights gained to inform a more complex, related task. The 'simpler' task here involves predicting the state of sleep at a given time point (sleeping or awake), potentially correlating with the two features over different time windows. We hypothesize that longer windows capture long-term trends, while shorter windows may indicate imminent movement. To this end, we have explored various statistical measures like mean squared error, variance, and permutation entropy over selected time windows.

After computing moving statistics for each data point and employing PCA for variance analysis and feature correlation, we are considering further feature transformation and the application of kernel tricks if necessary. Encouragingly, preliminary results with a simple random forest model on our test set have been promising, as evidenced by our confusion matrix.



With the initial step yielding results, the focus shifts to predicting wake-up and sleep onset times, now that we have insights into the sleep/awake state at each time point. Our approach involves segmenting continuous sleep states, as verified by our data. We employed clustering methods, specifically DBSCAN, using border points as wake-up and onset indicators. Additionally, we utilized change point detection algorithms like PELT and Bayesian Change-Point Detection from the Python 'ruptures' library.



Our evaluation method involves matching each set of predictions to ground-truth data within a defined tolerance. Predictions within this tolerance are considered true positives; those outside are false positives. Unmatched ground truths are false negatives. This allows us to construct a confusion matrix and evaluate performance metrics such as precision, recall, F1 score, and AUC.
