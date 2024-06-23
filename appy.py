import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('/Users/teresagodinho/Desktop/loan/loan_balanced_6040.csv')

# Data preprocessing
X = data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']]
y = data['loan_status']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Define the Random Forest Classifier with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
model = grid_search.best_estimator_

# Prepare data for linear regression to predict interest rates
X_interest = data[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]
y_interest = data['int_rate']

# Train a Linear Regression model for predicting interest rates
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_interest, y_interest)

# Standardize data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['annual_inc', 'loan_amnt']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Compute default probabilities for each client using the Random Forest Classifier
data['probability_of_default'] = model.predict_proba(data[['annual_inc', 'term', 'loan_amnt', 'home_ownership_OWN']])[:, 1]
data.sort_values(by='probability_of_default', ascending=False, inplace=True)

# Define the Streamlit app layout
st.title("LendSmart Dashboard")

# Main Page
tab1, tab2, tab3, tab4 = st.tabs(["Main Page", "Background Information", "New Client Default Prediction", "Client Risk Segmentation"])

with tab1:
    st.markdown(
        """
        This dashboard helps a US loan mortgage company identify and manage at-risk clients. 
        Using machine learning models and statistical analysis, it predicts loan defaults and provides actionable insights. 
        Amid rising US mortgage delinquency rates due to economic uncertainty (Financial Times), this tool enables early identification of potential defaults and better management of at-risk clients, 
        ensuring financial stability and improved loan portfolio management.
        """
    )

with tab2:
    st.markdown(
        """
        Explore various graphs that describe our dataset, which underpins the predictive tools used in the following tabs. 
        Gain insights into loan distributions, income levels, interest rates, and more.
        """
    )

    dropdown_selection = st.selectbox(
        "Select a graph to display:",
        ["Correlation Heatmap", "Distribution of Loan Status", "Distribution of Loan Amounts", "Distribution of Annual Incomes", "Distribution of Interest Rates"]
    )

    if dropdown_selection == 'Correlation Heatmap':
        correlation_matrix = data[['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'delinq_2yrs', 'home_ownership_OWN', 'home_ownership_RENT', 'open_acc', 'loan_status']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        ax.set_xticklabels(['Loan Amount', 'Loan Term', 'Interest Rate', 'Instalment', 'Annual Income', 'Delinquency in the Last 2 Years', 'Home Owner', 'Home Renter', 'Number of Open Accounts', 'Loan Status'], rotation=45)
        ax.set_yticklabels(['Loan Amount', 'Loan Term', 'Interest Rate', 'Instalment', 'Annual Income', 'Delinquency in the Last 2 Years', 'Home Owner', 'Home Renter', 'Number of Open Accounts', 'Loan Status'], rotation=0)
        st.pyplot(fig)

    elif dropdown_selection == 'Distribution of Loan Status':
        fig, ax = plt.subplots()
        data['loan_status'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Distribution of Loan Status')
        ax.set_xlabel('Loan Status')
        ax.set_ylabel('Number of Loans')
        st.pyplot(fig)

    elif dropdown_selection == 'Distribution of Loan Amounts':
        fig, ax = plt.subplots()
        data['loan_amnt'].plot(kind='hist', bins=50, color='blue', ax=ax)
        ax.set_title('Distribution of Loan Amounts')
        ax.set_xlabel('Loan Amount ($)')
        st.pyplot(fig)

    elif dropdown_selection == 'Distribution of Annual Incomes':
        fig, ax = plt.subplots()
        data['annual_inc'].plot(kind='hist', bins=50, color='purple', ax=ax)
        ax.set_title('Distribution of Annual Incomes')
        ax.set_xlabel('Annual Income ($)')
        st.pyplot(fig)

    elif dropdown_selection == 'Distribution of Interest Rates':
        fig, ax = plt.subplots()
        data['int_rate'].plot(kind='hist', bins=50, color='green', ax=ax)
        ax.set_title('Distribution of Interest Rates')
        ax.set_xlabel('Interest Rate (%)')
        st.pyplot(fig)

with tab3:
    st.markdown(
        """
        Enter your information to receive a personalized loan recommendation in seconds. 
        Our tool quickly evaluates your eligibility, helping you save time and determine the feasibility of your loan application. 
        If your loan is denied, you will receive a recommendation. If your loan is approved, we will suggest an interest rate.
        """
    )

    annual_income = st.number_input('Annual Income', value=120000, min_value=0, max_value=1000000)
    loan_term = st.number_input('Loan Term (months)', value=36, min_value=1, max_value=360)
    loan_amount = st.number_input('Loan Amount', value=300000, min_value=0, max_value=1000000)
    home_ownership = st.number_input('Home Ownership (OWN=1, RENT=0)', value=1, min_value=0, max_value=1)
    open_acc = st.number_input('Number of Open Accounts', value=5, min_value=0, max_value=50)
    delinq_2yrs = st.number_input('Delinquencies in Last 2 Years (1=YES, 0=NO)', value=0, min_value=0, max_value=50)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'annual_inc': [annual_income],
            'term': [loan_term],
            'loan_amnt': [loan_amount],
            'home_ownership_OWN': [home_ownership]
        })

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            result = 'Loan Denied'
            probability = f"{prediction_proba[0][1]*100:.2f}% probability of default"
            recommendations_title = 'Recommendations'
            recommendations = """
            - Reduce Loan Amount: A lower loan amount reduces the repayment burden, which can decrease the risk of default.
            - Extend Loan Term: Smaller monthly payments can be easier to manage, reducing the risk of default.
            """
            st.write(result)
            st.write(probability)
            st.write(recommendations_title)
            st.write(recommendations)
        else:
            result = 'Loan Accepted'
            probability = f"{prediction_proba[0][1]*100:.2f}% probability of default"

            # Predict the interest rate using the linear regression model
            input_data_for_rate = pd.DataFrame({
                'loan_amnt': [loan_amount],
                'open_acc': [open_acc],
                'delinq_2yrs': [delinq_2yrs],
                'term': [loan_term]
            })

            predicted_rate = lin_reg_model.predict(input_data_for_rate)
            recommended_rate = f"The suggested interest rate is {predicted_rate[0]:.2f}%."

            st.write(result)
            st.write(probability)
            st.write('Suggested Interest Rate')
            st.write(recommended_rate)

with tab4:
    st.markdown(
        """
        This heatmap visualizes the risk segmentation of clients based on their loan amounts and annual incomes. 
        Each cell represents the default probability for a specific segment, with colors ranging from green (low risk) to red (high risk). 
        By analyzing this heatmap, we can identify which client segments are more likely to default on their loans, 
        allowing for better risk management and targeted strategies.
        """
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    risk_levels = data.pivot_table(values='loan_status', 
                                   index=pd.cut(data['loan_amnt'], bins=range(0, 105000, 5000)), 
                                   columns=pd.cut(data['annual_inc'], bins=range(0, 1050000, 50000)), 
                                   aggfunc='mean')
    sns.heatmap(risk_levels, annot=True, cmap='RdYlGn_r', linewidths=0.5, ax=ax)
    ax.set_title('Client Risk Segmentation Heatmap')
    ax.set_xlabel('Annual Income')
    ax.set_ylabel('Loan Amount')
    st.pyplot(fig)

    st.markdown(
        """
        We're using our random forest model to calculate a new probability of default for all existing clients. 
        Based on these probabilities, we've also calculated suggested interest rates. 
        The goal is to improve the management of the company's at-risk clients.
        """
    )

    data['client'] = data.index + 1
    data['home_ownership'] = data['home_ownership_OWN'].map({1: 'OWN', 0: 'RENT'})
    data['suggested_interest_rate'] = lin_reg_model.predict(data[['loan_amnt', 'open_acc', 'delinq_2yrs', 'term']]).round(2)
    
    st.dataframe(data[['client', 'annual_inc', 'term', 'loan_amnt', 'home_ownership', 'delinq_2yrs', 'probability_of_default', 'int_rate', 'suggested_interest_rate']])

if __name__ == '__main__':
    st.set_page_config(layout="wide")
