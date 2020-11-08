# HELPER FUNCTIONS
## CSV
To read a csv file and convert into numpy array, you can use genfromtxt of the numpy package.
For Example:
```
train_data = np.genfromtxt(train_X_file_path, dtype=str, delimiter='\n')
```
You can use the python csv module for writing data to csv files.
Refer to https://docs.python.org/2/library/csv.html.
For Example:
```
with open('sample_data.csv', 'w') as csv_file:
	writer = csv.writer(csv_file)
    writer.writerows(data)
```
## JSON
You can store the computed values of the Naive Bayes model into a JSON. You can use the json library.
Refer: https://www.geeksforgeeks.org/reading-and-writing-json-to-a-file-in-python/
For Example:
```
with open('model_file.json', 'w') as json_file:
    json.dump(model, json_file, indent=4)
```