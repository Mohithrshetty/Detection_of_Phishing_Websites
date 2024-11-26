import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
from PIL import Image

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Details", "Model"])

if page == "Project Details":
    st.title('Detection of Phishing Websites Using URL and Visual Features')
    st.write(
        'This ML-based app is developed for detection of Phishing And Quishing Website (QR Phishing) Objective of the app is detecting phishing websites only using content data. Not URL Based Data!'
        ' You can see the details of approach, data set, and feature set if you click on _"Project Details"_. ')

    with st.expander("PROJECT DETAILS"):
        st.subheader('Approach')
        st.write('We used _supervised learning_ to classify phishing and legitimate websites. '
                 'We benefit from content-based approach and focus on html of the websites. '
                 'Also, We used scikit-learn for the ML models.'
                 )
        st.write('For this educational project, '
                 'We created our own data set and defined features, some from the literature and some based on manual analysis. '
                 'We used requests library to collect data, BeautifulSoup module to parse and extract features. ')
        # st.write('The source code and data sets are available in the below Github link:')
        st.write(' ')

        st.subheader('Data set')
        st.write('We used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
        st.write('Totally 31828 websites ==> **_15452_ legitimate** websites | **_16376_ phishing** websites')
        st.write('Data set was created in October 2024.')

        # ----- FOR THE PIE CHART ----- #
        labels = 'phishing', 'legitimate'
        phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
        legitimate_rate = 100 - phishing_rate
        sizes = [phishing_rate, legitimate_rate]
        explode = (0.1, 0)
        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)
        # ----- !!!!! ----- #

        st.write('Features + URL + Label ==> Dataframe')
        st.markdown('label is 1 for phishing, 0 for legitimate')
        number = st.slider("Select row number to display", 0, 100)
        st.dataframe(ml.legitimate_df.head(number))


        @st.cache_data
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')


        csv = convert_df(ml.df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='phishing_legitimate_structured_data.csv',
            mime='text/csv',
        )

        st.subheader('Features')
        st.write('We used only content-based features. I didn\'t use url-based features like length of url etc.'
                 'Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')

        st.subheader('Results')
        st.write(
            'We used 10 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
            'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
            'Comparison table is below:')
        st.table(ml.df_results)
        st.write('NB --> Gaussian Naive Bayes')
        st.write('SVM --> Support Vector Machine')
        st.write('DT --> Decision Tree')
        st.write('RF --> Random Forest')
        st.write('AB --> AdaBoost')
        st.write('NN --> Neural Network')
        st.write('KN --> K-Neighbours')
        st.write('LG --> Logistic Regression')
        st.write('GB --> Gradient Boosting')
        st.write('XGB --> XGBoost')

    with st.expander('EXAMPLE PHISHING URLs:'):
        st.write('https://kenny141.com/#abuse@optusnet.com.au')
        st.write('https://ipfs.io/ipfs/bafkreiavpbvl4eytaxgtd6kp5jhxocpwvzatklw7d2bprxfcjqgf7zgiim')
        st.write('https://ipfs.io/ipfs/bafkreibt27oeumdfrpnzdwfrfntqlrp2c7ui23ylsrojktp2mfxkw76rtu')
        st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE! SO, THE EXAMPLES SHOULD BE UPDATED!')

elif page == "Model":
    st.title('Detection of Phishing Websites Using URL and Visual Features')

    choice = st.selectbox("Please select your machine learning model",
                          ['Gaussian Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Random Forest',
                           'AdaBoost', 'Neural Network', 'K-Neighbours','Logistic Regression','Gradient Boosting','XGBoost']
                          )

    model = ml.models["NaiveBayes"]
    if choice == 'Gaussian Naive Bayes':
        model = ml.models["NaiveBayes"]
        st.write('GNB model is selected!')
    elif choice == 'Support Vector Machine':
        model = ml.models["SVM"]
        st.write('SVM model is selected!')
    elif choice == 'Decision Tree':
        model = ml.models["DecisionTree"]
        st.write('DT model is selected!')
    elif choice == 'Random Forest':
        model = ml.models["RandomForest"]
        st.write('RF model is selected!')
    elif choice == 'AdaBoost':
        model = ml.models["AdaBoost"]
        st.write('AB model is selected!')
    elif choice == 'Neural Network':
        model = ml.models["NeuralNetwork"]
        st.write('NN model is selected!')
    elif choice == 'Logistic Regression':
        model = ml.models["LogisticRegression"]
        st.write('LG model is selected!')
    elif choice == 'Gradient Boosting':
        model = ml.models["GradientBoosting"]
        st.write('GB model is selected!')
    elif choice == 'XGBoost':
        model = ml.models["XGBoost"]
        st.write('XBG model is selected!')
    else:
        model = ml.models["KNeighbors"]
        st.write('KN model is selected!')

    # Option to choose between URL input and file upload
    option = st.selectbox("Choose an option:", ["Enter URL", "Upload QR Scanner"])

    # Conditional inputs based on user selection
    if option == "Enter URL":
        url = st.text_input("Enter URL:")
    else:
        uploaded_image = st.file_uploader("Upload a QR Scanner:", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            data = decode(Image.open(uploaded_image))
            url = data[0].data.decode()
            print(data[0].data.decode())

    if st.button('Check!'):
        try:
            response = re.get(url, verify=False, timeout=4)
            if response.status_code != 200:
                print(". HTTP connection was not successful for the URL: ", url)
                st.warning("HTTP connection was not successful!") 
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = [fe.create_vector(soup)]  # it should be 2d array, so I added []
                result = model.predict(vector)
                if result[0] == 0:
                    st.success("This web page seems legitimate!")
                    st.balloons()
                else:
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.snow()

        except re.exceptions.RequestException as e:
            print("--> ", e)
