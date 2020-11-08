# Datasets
train_X= https://d15dkfioxfhy94.cloudfront.net/tenxiitian_prod/AI-ML/Decision_trees/Project/train_X_dt.arff
train_Y= https://d15dkfioxfhy94.cloudfront.net/tenxiitian_prod/AI-ML/Decision_trees/Project/train_Y_dt.arff


# HELPER FUNCTIONS
## CSV
To read a csv file and convert into numpy array, you can use genfromtxt of the numpy package.
For Example:
```
train_data = np.genfromtxt(train_X_file_path, dtype=np.float64, delimiter=',', skip_header=1)
```
You can use the python csv module for writing data to csv files.
Refer to https://docs.python.org/2/library/csv.html.
For Example:
```
with open('sample_data.csv', 'w') as csv_file:
	writer = csv.writer(csv_file)
    writer.writerows(data)
```
## ARFF
For handling files in arff format, you can use the arff library in python.
Refer to https://pypi.org/project/liac-arff/.
1. To read an arff file and convert to numpy array, use arff.load
    For Example:
    ```
    arff_data = arff.load(open('sample_file.arff', 'r'))
    data = np.asarray(arff_data['data'])
    ```
2. To write a numpy array to arff file, use arff.dump
    For Example:
    ```
    np_array = np.array([["True"], ["False"], ["True"]])
    arff_data={'data':np_array, 'description':'', 'relation':'relation', 'attributes':[('class',['True','False'])]}
    with open('./sample_file.arff','w') as arff_file:
        arff.dump(arff_data,arff_file)
    ```
## PICKLE
You can use the pickle library (Reference: https://www.geeksforgeeks.org/saving-a-machine-learning-model/)
1. To store the training classifier into the model file.
For example:
	```
	pickle.dump(model, open('model_file.sav', 'wb'))
	```
2. To load the model from model file.
For example:
	```
	model = pickle.load(open('./model_file.sav', 'rb'))
	```