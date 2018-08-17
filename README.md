# Kaggle_DCASE2018_21st_Solution
This repository houses the codes for the 21st place solution for the DCASE 2018 challenge, held on Kaggle. This solution is only 0.02 points lesser than the best solution.

The solution comprises of a diversified system of deep convolutional neural networks with stacked fusion of spectral features for the DCASE 2018 Task 2, freesound general-purpose audio tagging.
My primary objective has been to design a solution which can churn out decent performance and be deployed within reasonable resource constraints. The two best performing submissions are the results of only two and three different CNNs, with their results being combined based on a boosted tree algorithm with fused spectral features. Experimental results show that the proposed system and preprocessing methods effectively learn acoustic characteristics from the audio recordings, and their ensemble models significantly reduce the error rate further, exhibiting a MAP@3 score of 0.933 and 0.932. 

The solution is described in the paper, which would be uploaded soon.
