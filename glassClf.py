import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC as svc
from sklearn.linear_model import LogisticRegression as lgreg
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import plot_confusion_matrix as pcm, plot_roc_curve as prc, plot_precision_recall_curve as prc
from sklearn.metrics import precision_score as ps, recall_score as rs

@st.cache()

def load():

    fp = "glass-types.csv"
    df = pd.read_csv(fp, header = None)

    df.drop(columns = 0, inplace = True)
    ch = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    cd = {}

    for i in df.columns:
        cd[i] = ch[i - 1]
        df.rename(cd, axis = 1, inplace = True)
    
    return df

gdf = load()

f = gdf.iloc[:, :-1]
t = gdf['GlassType']
xtrain, xtest, ytrain, ytest = tts(f, t, test_size = 0.3, random_state = 42)

elements = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

@st.cache()

def prediction(pm, e):

    pred = pm.predict([e])
    gtype = pred[0]

    if gtype == 1:
        return "Building Windows (Float Processed)"
    elif gtype == 2:
        return "Building Windows (Non-Float Processed)"
    elif gtype == 3:
        return "Vehicle Windows (Float Processed)"
    elif gtype == 4:
        return "Vehicle Windows (Non-Float Processed)"
    elif gtype == 5:
        return "Containers"
    elif gtype == 6:
        return "Tableware"
    else:
        return "Headlamp"

st.title("Glass Classifier")

st.sidebar.title("Options")

if st.sidebar.checkbox("Display raw data"):
    st.subheader("Glass Type DataFrame")
    st.write("|-----------------------------------------------Elements-------------------------------------------------|")
    st.dataframe(gdf)