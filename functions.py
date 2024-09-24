from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix
import streamlit as st

def Linearregression(X,Y,train_test_ratio=0.2,random_states=1):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=train_test_ratio, random_state=int(random_states))
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
            st.write("Model Score: ",model.score(X_test,y_test))
            st.write("Accuracy Score: ",accuracy_score(y_test, y_pred))

def Logisticregression(X,Y,train_test_ratio=0.2,random_states=1):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=train_test_ratio, random_state=int(random_states))
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy Score: ",confusion_matrix(y_test, y_pred))