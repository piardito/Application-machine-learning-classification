import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


st.title('Predictions')

st.markdown(":tada: Ceci <span style='color:blue;'>est</span> une appli ",
            unsafe_allow_html=True
)

f=st.sidebar.file_uploader("Uploader un fichier csv",["csv"])

if f is not None:
  df=pd.read_csv(f,sep=';')
  df=df.dropna()


  if st.sidebar.checkbox("Data Preview"):
    st.write(df.style.background_gradient(cmap='YlOrBr'))

  if st.sidebar.checkbox("Univariate"):
   col=st.selectbox("Choisissez une colonne",df.columns)
   chart =alt.Chart(df).mark_bar().encode(x=col,y='count()',tooltip=[col]).interactive()


   st.altair_chart(chart,use_container_width=True)

   if st.sidebar.checkbox("Classification"):

       target = st.selectbox("Choisissez la cible ", df.columns)
       X1 = df.loc[:, df.columns != target]
       X = pd.DataFrame(X1)
       y=df[target]
       X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=42)


   type = st.sidebar.selectbox("Algorithme",("RandomForest","LogisticRegression",'arbre de decision'))
   if type == "RandomForest" :
       n_estimators = st.number_input("Max number estimators", 5, 100, 10)
       max_depth = st.number_input("Profondeur des arbres", 2, 10, 2)
       Rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
       Rf.fit(X_train,y_train)

       y_pred=Rf.predict(X_test)

       st.title('RandomForest')
       fig,ax=plt.subplots()
       plot_confusion_matrix(Rf,X_test,y_test,ax=ax)
       score=accuracy_score(y_pred,y_test)

       fig,score

   if type == "LogisticRegression":
       Lr = LogisticRegression()
       Lr.fit(X_train, y_train)

       y_pred_lr = Lr.predict(X_test)
       st.title("Regression Logistique")
       fig1, ax1 = plt.subplots()
       plot_confusion_matrix(Lr, X_test, y_test, ax=ax1)
       score1 = accuracy_score(y_pred_lr, y_test)


       fig1,score1

   if type=='arbre de decision':
       method = st.selectbox("methode",('gini','entropy'))
       max_depth1 = st.number_input("Profondeur des arbres", 2, 10, 2)
       ad = DecisionTreeClassifier(random_state=0,criterion=method,max_depth=max_depth1)
       ad.fit(X_train, y_train)

       y_pred_ad = ad.predict(X_test)
       st.title("Arbre de décision")
       fig2, ax2 = plt.subplots()
       plot_confusion_matrix(ad, X_test, y_test, ax=ax2)
       score2 = accuracy_score(y_pred_ad, y_test)

       fig2, score2

   choix = st.sidebar.selectbox("Choix", ("RandomForest", "LogisticRegression", 'arbre de decision'))

   g = st.sidebar.file_uploader("Uploader un fichier", ["csv"])

   df1 = pd.read_csv(g, sep=';')

   target1 = st.selectbox("Choisissez la cible ", df1.columns)


   X2 = df1.loc[:, df1.columns != target1]

   if choix == "LogisticRegression":


    predictions=Lr.predict(X2)

    predictions


   if choix=="RandomForest":

    predictions1=Rf.predict(X2)

    predictions1

   if choix=="arbre de decision":

    predictions2 = ad.predict(X2)

    predictions2



















