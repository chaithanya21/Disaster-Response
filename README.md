# Disaster-Response

This project is all about analysing disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

<h2>Description</h2>

The Project Dataset Contains real messages that were sent during the disaster events. The main aim of the project is to create a **machine Learning Pipeline** to categorize the events so that these messages will be sent to an appropriate disaster relief agency.

The Project also includes a **web application** where an emergency worker can input a new message and get the Classification results in several categories.The Web Application also displays visualizations of the data.

<h2>Project Instructions </h2>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
 2. Run the following command in the app's directory to run your web app.
 `python run.py`
 
 3. Go to http://0.0.0.0:3001/
 
 <h2>Requiremnts</h2>
 
1.pandas

2.sqlalchemy

3.sklearn

4.Flask

Note that for sklearn if you are using the latest version you will get an error. Use 0.20.4 or lower version. To create same python environment use command conda create --name <env_name> --file requirement.txt

 <h2>Results</h2>

<h3>Home Page Of the Web Application</h3>

 <img src='https://github.com/chaithanya21/Disaster-Response/blob/master/IMAGES/HomePage.png' width=800 >
 
 <h2>Example Output</h2>
 
 <h3>Message Recieved</h3>
 
 <img src='https://github.com/chaithanya21/Disaster-Response/blob/master/IMAGES/Input_Message.png'>
 
 <h3>Classification Results</h3>
 
 <img src='https://github.com/chaithanya21/Disaster-Response/blob/master/IMAGES/Result.png'>
     
 
 
 
