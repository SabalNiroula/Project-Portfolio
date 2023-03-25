from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score
import streamlit as st
from sklearn.datasets import make_classification, make_regression
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Bagging Technique",
    page_icon="logo.jpg",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    n_features = X.shape[1]
    x_min, x_max = [], []
    for i in range(n_features):
        x_min.append(X[:, i].min() - 10*h)
        x_max.append(X[:, i].max() + 10*h)
    xx = np.meshgrid(*[np.arange(x_min[i], x_max[i], h)
                     for i in range(n_features)])    
    Z = clf.predict(np.c_[xx[0].ravel(), xx[1].ravel()]).reshape(xx[0].shape)
    fig = plt.figure(figsize=(6.45, 5))
    ax = fig.add_subplot(
        111, projection='3d') if n_features > 2 else fig.add_subplot(111)
    if n_features == 2:
        ax.contourf(xx[0], xx[1], Z, cmap=cmap, alpha=0.25)
        ax.contour(xx[0], xx[1], Z, colors='k', linewidths=0.7)
        ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=cmap, edgecolors='k')
        xx_0, xx_1 = np.meshgrid(xx[0], xx[1])
        ax.plot_surface(xx_0, xx_1, xx[2], facecolors=plt.cm.Paired_r(Z))
    st.pyplot(fig)

statement = ''

# Set up the UI elements
st.sidebar.title('Bagging Classifier Settings')
problem = st.sidebar.selectbox(
    'Problem', ('Classification', 'Regression'))

if problem == 'Classification':
    statement = 'cla'
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                               n_classes=2, random_state=42, class_sep=1.25)
    m, n = X.shape
    # Choose base model
    base_model = st.sidebar.selectbox(
        'Base Model', ('SVM', 'Decision Tree', 'KNN'))
    n_estimators = st.sidebar.number_input(
        'Number of Estimators', value=100, step=10, min_value=1)
    # Set max sample
    max_sample = st.sidebar.slider('Max Sample', 0, m)
    # Set bootstrap sample
    bootstrap_sample = st.sidebar.radio(
        'Bootstrap Sample?', [True, False], index=0)
    # Set max feature
    max_feature = st.sidebar.slider('Max Feature', 0, n)
    # Set bootstrap feature
    bootstrap_feature = st.sidebar.radio(
        'Bootstrap Feature?', [True, False], index=0)

else:
    statement = 'reg'
    X, y = make_regression(n_samples=500, n_features=2,
                           n_informative=1, bias=3, noise=5, random_state=42)
    m, n = X.shape
    # Choose base model
    base_model = st.sidebar.selectbox(
        'Base Model', ('SVM', 'Decision Tree', 'KNN'))
    n_estimators = st.sidebar.number_input(
        'Number of Estimators', value=100, step=10, min_value=1)
    # Set max sample
    max_sample = st.sidebar.slider('Max Sample', 0, m)
    # set the bootstrap sample
    bootstrap_sample = st.sidebar.radio(
        'Bootstrap Sample?', [True, False], index=0)

def create_and_train_model(base_model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    if statement == 'cla':
        bag = BaggingClassifier(
            base_estimator=base_model,
            n_estimators=n_estimators,
            bootstrap=bootstrap_sample,
            max_features=max_feature,
            bootstrap_features=bootstrap_feature)
        
        decision_tree = DecisionTreeClassifier()
    else:
        bag = BaggingRegressor(
            base_estimator=base_model,
            n_estimators=n_estimators,
            max_samples=max_sample,
            bootstrap=bootstrap_sample)

        decision_tree = DecisionTreeRegressor()

    bag.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)

    if statement == 'cla':  # classification problem
        y_pred1 = bag.predict(X_test)
        y_pred2 = decision_tree.predict(X_test)
    else:
        y_pred1 = bag.predict(X_test)
        y_pred2 = decision_tree.predict(X_test)

    return [(bag, r2_score(y_test, y_pred1)), (decision_tree, r2_score(y_test, y_pred2))]



# Run the algorithm
if st.sidebar.button('Run Algorithm'):
    # Build the Bagging Classifier
    if base_model == 'SVM':
        if statement == 'cla':
            base_model = SVC()
        else:
            base_model = SVR()
    elif base_model == 'Decision Tree':
        if statement == 'cla':
            base_model = DecisionTreeClassifier()
        else:
            base_model = DecisionTreeRegressor()
    elif base_model == 'KNN':
        if statement == 'cla':
            base_model = KNeighborsClassifier()
        else:
            base_model = KNeighborsRegressor()


    bag, dec = create_and_train_model(base_model=base_model, X=X, y=y);

    if  statement == 'cla':
        st.title('Bagging Classifier')
        plot_decision_boundary(clf=bag[0], X=X, Y=y)
        st.write(f'R-squared score: {bag[1]*100:.2f}')

        st.title(f'{base_model}')
        plot_decision_boundary(clf=dec[0], X=X, Y=y)
        st.write(f'R-squared score: {dec[1]*100:.2f}')

    else:
        st.title('Bagging Regressor')
        plot_decision_boundary(clf=bag[0], X=X, Y=y)
        st.write(f'R-squared score: {bag[1]*100:.2f}')

        st.title(f'{base_model}')
        plot_decision_boundary(clf=dec[0], X=X, Y=y)
        st.write(f'R-squared score: {dec[1]*100:.2f}')
