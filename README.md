# Programming Project 3:

## GLM.py:
This program takes as input the dataset file names and the model name. The files should be kept in a folder named 'pp3data' under the current folder.
For each dataset file, it generates a graph plotting the error rates against the sampling size for alpha=10.

### 1) Logistic Regression : 

Input: 
```
 python GLM.py A.csv labels-A.csv Logistic
```

```
 python GLM.py usps.csv labels-usps.csv Logistic
```

### 2) Poisson Regression : 

Input:
```
 python GLM.py AP.csv labels-AP.csv Poisson
````
### 3) Ordinal Regression :

Input:
```
 python GLM.py AO.csv labels-AO.csv Ordinal
````

## Alpha.py:
This program takes as input the dataset file names and the model name. The files should be kept in a folder named 'pp3data' under the current folder.
For each dataset file, it generates a graph plotting the error rates against the values of alpha from 1 to 100. 
The program uses cross-valdiation and returns the value of alpha with the lowest error. The input format is same as GLM.py .

Input:
```
 python Alpha.py irlstest.csv labels-irlstest.csv Logistic
````
