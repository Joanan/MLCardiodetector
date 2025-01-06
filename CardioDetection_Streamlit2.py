import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

st.title('Heart Disease Prediction Model Trainer')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Perform your data preprocessing steps:
    data = data.drop_duplicates()

    # Identifying categorical and continuous variables
    cate_val = []
    cont_val = []

    for column in data.columns:
        if data[column].nunique() <= 10:
            cate_val.append(column)
        else:
            cont_val.append(column)

    cate_val.remove('target')  # Assuming 'target' is the label for prediction

    # Scaling numeric features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(data[cont_val]), columns=cont_val)

    # Encoding categorical features, keeping the structure intact for validation

    X_train_encoded = pd.get_dummies(data[cate_val],columns=cate_val)

    # Save the model columns
    model_columns = X_train_encoded.columns
    


    # Concatenate all processed features
    X_train_preprocessed = pd.concat([X_train_scaled.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)

    #X_train_preprocessed = pd.concat([X_train_scaled.reset_index(drop=True), label_encoders.reset_index(drop=True)], axis=1)
    #X = data[cont_val + cate_val]
    X = X_train_preprocessed
    
    #print(X.shape)
    y = data['target']

    # Encoding categorical features
    ##X_train_encoded = pd.get_dummies(data[cate_val], columns=cate_val)

    # Concatenate all processed features
    ##X_train_preprocessed = pd.concat([X_train_scaled.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
    ##X = X_train_preprocessed
    ##y = data['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': svm.SVC(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model_training = model.fit(X_train, y_train)
        trained_models[name] = model_training
        results[name] = {
            'Accuracy': accuracy_score(y_test, model_training.predict(X_test))
        }

    best_model = max(results, key=lambda x: results[x]['Accuracy'])
    st.write(f"Best model based on Accuracy: {best_model} with Accuracy: {results[best_model]['Accuracy']:.2f}")

    # Save models and scaler
    joblib.dump(scaler, '/downloads/scaler.pkl')
    joblib.dump(model_columns, '/downloads/label_encoders.pkl')
    joblib.dump(trained_models[best_model], '/downloads/best_model.pkl')

    # Allow user to input new data for validation
    st.subheader("Validate New Data")
    input_data = {column: st.number_input(f"Enter {column}", key=column) for column in cont_val}
    input_data.update({column: st.selectbox(f"Select {column}", options=sorted(list(data[column].unique())), key=column) for column in cate_val})
    input_df = pd.DataFrame([input_data])
 
    
    if st.button('Predict'):
        # Load scaler and model
        scaler = joblib.load('/downloads/scaler.pkl')
        label_encoders = joblib.load('/downloads/label_encoders.pkl')
        model = joblib.load('/downloads/best_model.pkl')
        
        #print (input_df.head())
        
        # Preprocess new data
        X_validation_scaled = pd.DataFrame(scaler.transform(input_df[cont_val]), columns=cont_val)
        
       # X_validation_encoded = pd.get_dummies(input_df[cate_val].iloc[0])
        X_validation_encoded = pd.get_dummies(input_df, columns=cate_val)
        X_validation_encoded = X_validation_encoded.reindex(columns=label_encoders, fill_value=False)
        X_validation_preprocessed = pd.concat([X_validation_scaled.reset_index(drop=True), X_validation_encoded.reset_index(drop=True)], axis=1)
        
        #print (X_validation_preprocessed.head())
        
        # Prediction
        prediction = model.predict(X_validation_preprocessed)
        st.write('Prediction:', 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
        #streamlit run /downloads/CardioDetection_Streamlit2.py

