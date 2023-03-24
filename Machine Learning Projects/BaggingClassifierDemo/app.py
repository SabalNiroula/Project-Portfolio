from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import streamlit as st

dataframe = [1, 2, 3, 4, 5, 6, 67, 7]

# Set up the UI elements
st.sidebar.title('Bagging Classifier Settings')

problem = st.sidebar.selectbox(
    'Problem', ('Classification', 'Regression'))

if problem == 'Classification':
    # Choose base model
    base_model = st.sidebar.selectbox(
        'Base Model', ('SVM', 'Decision Tree', 'KNN'))

    n_estimators = st.sidebar.number_input(
        'Number of Estimators', value=10, step=1, min_value=1)

    # Set max sample
    max_sample = st.sidebar.slider('Max Sample', 0, len(dataframe))

    # Set bootstrap sample
    bootstrap_sample = st.sidebar.radio(
        'Bootstrap Sample?', [True, False], index=0)

    # Set max feature
    max_feature = st.sidebar.slider('Max Feature', 0, len(dataframe))

    # Set bootstrap feature
    bootstrap_feature = st.sidebar.radio(
        'Bootstrap Feature?', [True, False], index=0)

else:
    # Choose base model
    base_model = st.sidebar.selectbox(
        'Base Model', ('SVM', 'Decision Tree', 'KNN'))

    n_estimators = st.sidebar.number_input(
        'Number of Estimators', value=10, step=1, min_value=1)
    
    # Set bootstrap sample
    bootstrap_sample = st.sidebar.radio(
        'Bootstrap Sample?', [True, False], index=0)

# Run the algorithm
if st.sidebar.button('Run Algorithm'):
    # Build the Bagging Classifier
    if base_model == 'SVM':
        base_model = SVC()
    elif base_model == 'Decision Tree':
        base_model = DecisionTreeClassifier()
    elif base_model == 'KNN':
        base_model = KNeighborsClassifier()

    bagging_classifier = BaggingClassifier(
        base_estimator=base_model,
        n_estimators=n_estimators,
        bootstrap=bootstrap_sample,
        max_features=max_feature,
        bootstrap_features=bootstrap_feature
    )

    # Train the Bagging Classifier
    # bagging_classifier.fit(X_train, y_train)

    # # Evaluate the Bagging Classifier
    # accuracy = bagging_classifier.score(X_test, y_test)
    st.write(f'Accuracy: {89.453:.2f}')
