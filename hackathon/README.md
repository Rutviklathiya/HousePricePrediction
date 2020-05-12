## ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict employee salaries absed on trainign data in 'hiring.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run extra.py using below command to start Flask API
```
python extra.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below link:
https://rlathiya.weebly.com/uploads/1/2/4/0/124059345/screenshot-from-2020-05-12-10-52-56_orig.png

Enter valid numerical values in all  input boxes and hit Predict.

Or you can use the chatbot to predict the price as below link:
https://rlathiya.weebly.com/uploads/1/2/4/0/124059345/screenshot-from-2020-05-12-10-57-20_orig.png

Also watch the exploratory data analysis as below link:
https://124059345-784427355905349158.preview.editmysite.com/uploads/1/2/4/0/124059345/screenshot-from-2020-05-12-10-57-49_orig.png

The maps iterface would look like this :
https://rlathiya.weebly.com/uploads/1/2/4/0/124059345/screenshot-from-2020-05-12-10-57-40_orig.png


4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
