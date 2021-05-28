# Disaster Response Pipeline Project

### Instructions:
0. Download pretrained embedding model. We used [pre-trained vectors](https://code.google.com/archive/p/word2vec/) trained on part of Google News dataset (about 100 billion words). The model contains 300-dimensional vectors. 
To download data please run the following commands
```
mkdir models/pretrained_nlp_models
cd models/pretrained_nlp_models
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
gzip -d GoogleNews-vectors-negative300.bin.gz
```
1. Run the following commands in the project's root directory to set up your database and model.

* To run ETL pipeline that cleans data and stores in database

```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

* To run ML pipeline that trains classifier and saves

```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

2. Run the following command in the app's directory to run your web app.
```
python run.py
```

3. Go to http://0.0.0.0:3001/
