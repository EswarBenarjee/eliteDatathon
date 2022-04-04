from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import numpy as np
import pandas as pd
import pickle
import xgboost
from sklearn import preprocessing 
from fancyimpute import KNN 
import sklearn
from sklearn.model_selection import cross_val_score

app = Flask(__name__)

@app.route('/')
def file():
   return render_template('index.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      df = pd.read_csv(f.filename)
      df['Lymphocytes'] = pd.to_numeric(df['Lymphocytes'], errors="coerce")
      X = df.iloc[:, :-1].values
      y = df.iloc[:, -1].values
      le = preprocessing.LabelEncoder()
      X[:, 0] = le.fit_transform(X[:, 0])
      knn_imputer = KNN()
      X[:, 2:9] = knn_imputer.fit_transform(X[:, 2:9])
      X[:, 9:] = knn_imputer.fit_transform(X[:, 9:])
      file2 = open("xg_pickle", "rb")
      xg = pickle.load(file2)
      return render_template("result.html", res=max(cross_val_score(xg, X, y))*100)

if __name__ == '__main__':
   app.run(debug = True)