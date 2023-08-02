import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import plotly.express as px
import plotly.graph_objects as go

def load_data():
    df = pd.read_csv('final_hotel_bookings.csv')
    return df

def overview(df):
    st.title("Overview")

    st.subheader("Distribution of Key Numerical Variables")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(df['lead_time'], kde=True, ax=axes[0, 0])
    sns.histplot(df['adr'], kde=False, bins=30, ax=axes[0, 1])
    sns.histplot(df['arrival_date_day_of_month'], kde=False, bins=30, ax=axes[1, 0])
    sns.histplot(df['arrival_date_week_number'], kde=False, bins=30, ax=axes[1, 1])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

def booking_analysis(df):
    st.title("Booking Analysis")

    st.subheader("Distribution of Total Bookings per Hotel Types")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(x='hotel', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Number of Bookings for Each Month")
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='arrival_date_month', data=df, order=months_order, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Number of Bookings in Seasons")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(x='season', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Top 10 Countries with the Most Bookings")
    top_countries = df['country'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_countries.index, y=top_countries.values, ax=ax)
    st.pyplot(fig)

def adr_analysis(df):
    st.title("ADR (Average Daily Rate) Analysis")

    st.subheader("Average Daily Rate (ADR) by Hotel Type")
    hotel_adr_mean = df.groupby('hotel')['adr'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='hotel', y='adr', data=hotel_adr_mean, ax=ax)
    st.pyplot(fig)

    st.subheader("Total ADR for Each Month")
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    total_adr_month = df.groupby('arrival_date_month')['adr'].sum().reindex(months_order)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=total_adr_month.index, y=total_adr_month.values, ax=ax)
    st.pyplot(fig)

    st.subheader("Average ADR for Top 10 Countries with the Most Bookings")
    top_countries_adr = df['country'].value_counts().index[:10]
    df_top_countries_adr = df[df['country'].isin(top_countries_adr)]
    avg_adr_countries = df_top_countries_adr.groupby('country')['adr'].mean()
    colors = ['b', 'g', 'grey', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_adr_countries.sort_values(ascending=False).plot(kind='bar', color=colors, ax=ax)
    st.pyplot(fig)

def cancellation_analysis(df):
    st.subheader('Total Bookings vs Total Cancellations (Top 10 Countries)')
    # Calculate the number and proportion of cancellations for each country
    country_cancellations = df[df['is_canceled'] == 1]['country'].value_counts()
    country_cancellations_proportion = df[df['is_canceled'] == 1]['country'].value_counts(normalize=True)

    # Calculate the total number of bookings for each country
    country_bookings = df['country'].value_counts()

    # Select the top 10 countries in terms of total bookings
    top_countries_bookings = country_bookings[:10]
    top_countries_cancellations = country_cancellations[top_countries_bookings.index]

    # Create a new DataFrame for plotting
    top_countries_df = pd.DataFrame({
        'Total_Bookings': top_countries_bookings,
        'Total_Cancellations': top_countries_cancellations
    })

    # Plot the total number of bookings vs total cancellations
    top_countries_df.plot(kind='bar', figsize=(10, 6))
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader('Proportion of Cancellations by Guest Type')
    # Calculate the proportion of bookings that were cancelled for each guest type
    guest_cancellations = df.groupby('guest_type')['is_canceled'].mean()

    # Plot the results
    guest_cancellations.plot(kind='bar', figsize=(10, 6))
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader('Total Bookings vs Total Cancellations (Market Segment)')
    # Calculate the total number of bookings and cancellations for each market segment
    market_segment_bookings = df['market_segment'].value_counts()
    market_segment_cancellations = df[df['is_canceled'] == 1]['market_segment'].value_counts()

    # Create a new DataFrame for plotting
    market_segment_df = pd.DataFrame({
        'Total_Bookings': market_segment_bookings,
        'Total_Cancellations': market_segment_cancellations
    }).sort_values(by='Total_Bookings', ascending=False)

    # Plot the total number of bookings vs total cancellations
    market_segment_df.plot(kind='bar', figsize=(10, 6))
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader('Total Bookings vs Total Cancellations by Deposit Type')
    # Calculate total bookings for each deposit type
    deposit_total_bookings = df['deposit_type'].value_counts()

    # Calculate total cancellations for each deposit type
    deposit_total_cancellations = df[df['is_canceled'] == 1]['deposit_type'].value_counts()

    # Combine total bookings and total cancellations into one dataframe
    deposit_data = pd.DataFrame({'Total_Bookings': deposit_total_bookings, 
                                 'Total_Cancellations': deposit_total_cancellations})

    # Calculate cancellation percent for each deposit type
    deposit_data['Cancellation_Percent'] = (deposit_data['Total_Cancellations'] / deposit_data['Total_Bookings']) * 100

    # Plot total bookings and total cancellations for each deposit type in one chart
    deposit_data[['Total_Bookings', 'Total_Cancellations']].plot(kind='bar', figsize=(10, 6))
    st.pyplot(plt.gcf())
    plt.clf()


def guest_analysis(df):
    st.title("Guest Analysis")

    st.subheader("Guest Type Distribution")
    guest_type_counts = df['guest_type'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(guest_type_counts, labels=guest_type_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("ADR Share per Guest Type")
    total_adr_per_category = df.groupby('guest_type')['adr'].sum()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(total_adr_per_category, labels=total_adr_per_category.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader('Repeated Guests by Deposit Type')
    # Filter data for repeated guests
    repeated_guests_data = df[df['is_repeated_guest'] == 1]

    # Count the number of repeated guests for each deposit type
    repeated_guests_by_deposit = repeated_guests_data['deposit_type'].value_counts()

    # Plot the number of repeated guests for each deposit type
    plt.figure(figsize=(8, 6))
    sns.barplot(x=repeated_guests_by_deposit.index, y=np.log1p(repeated_guests_by_deposit.values))
    plt.title('Repeated Guests by Deposit Type')
    plt.xlabel('Deposit Type')
    plt.ylabel('Number of Guests (Log Scale)')
    st.pyplot(plt.gcf())
    plt.clf()

def advanced_analysis(df):
    st.title("Advanced Analysis")

    st.subheader("Control Chart for Lead Time")
    mean_lead_time = df['lead_time'].mean()
    std_dev_lead_time = df['lead_time'].std()
    USL_lead_time = mean_lead_time + 3 * std_dev_lead_time
    LSL_lead_time = max(0, mean_lead_time - 3 * std_dev_lead_time)
    fig, ax = plt.subplots(figsize=(12, 9))
    df['lead_time'].plot(kind='line', ax=ax)
    ax.axhline(mean_lead_time, color='r', linestyle='dashed', linewidth=2)
    ax.axhline(USL_lead_time, color='g', linestyle='dashed', linewidth=2)
    ax.axhline(LSL_lead_time, color='g', linestyle='dashed', linewidth=2)
    ax.legend([f'Mean: {mean_lead_time:.2f}', f'Std. Dev.: {std_dev_lead_time:.2f}', f'USL: {USL_lead_time:.2f}'], loc='upper right')
    st.pyplot(fig)

    st.subheader("Cancellation Rate by Lead Time")
    bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660]
    df['lead_time_category'] = pd.cut(df['lead_time'], bins)
    cancellation_rates = df.groupby('lead_time_category')['is_canceled'].mean() * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=cancellation_rates.index, y=cancellation_rates.values, ax=ax)
    st.pyplot(fig)

    st.subheader("Sum of ADR for Each Category")
    adr_sum_grouped = df.groupby('is_canceled')['adr'].sum().reset_index()
    adr_sum_grouped['is_canceled'] = adr_sum_grouped['is_canceled'].map({0: 'Not Canceled', 1: 'Canceled'})
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(adr_sum_grouped['is_canceled'], adr_sum_grouped['adr'] / 1000000, color='skyblue')
    ax.set_title('Sum of ADR for Each Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Sum of ADR (Million)')
    for bar in bars:
        yval = round(bar.get_height(), 2)
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}M', ha='center', va='bottom')
    st.pyplot(fig)

    st.subheader("Sum of ADR for Different Scenarios")
    # Create a DataFrame to store these values
    # Calculate the sum of ADR for not canceled and canceled bookings
    adr_sum_not_canceled = df[df['is_canceled'] == 0]['adr'].sum()
    adr_sum_canceled = df[df['is_canceled'] == 1]['adr'].sum()

    # Simulate conversion of canceled bookings
    adr_sum_canceled_converted_25 = adr_sum_not_canceled + adr_sum_canceled * 0.25
    adr_sum_canceled_converted_50 = adr_sum_not_canceled + adr_sum_canceled * 0.5
    adr_sum_canceled_converted_75 = adr_sum_not_canceled + adr_sum_canceled * 0.75

    # Create a DataFrame to store these values
    simulation_df = pd.DataFrame({
        'Scenario': ['Actual Not Canceled', 'Convert 25%(+1.16m)', 'Convert 50%(+2.32m)', 'Convert 75%)'],
        'Sum ADR (in million)': [adr_sum_not_canceled / 1e6, adr_sum_canceled_converted_25 / 1e6,
                                adr_sum_canceled_converted_50 / 1e6, adr_sum_canceled_converted_75 / 1e6]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(simulation_df['Scenario'], simulation_df['Sum ADR (in million)'])
    ax.set_title('Sum of ADR for Different Scenarios')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Sum ADR (in million)')
    ax.set_xticklabels(simulation_df['Scenario'], rotation=45, ha='right')  # corrected line

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    st.pyplot(fig)



df = load_data()
page = st.sidebar.selectbox("Choose a page", ["Overview", "Booking Analysis", "ADR Analysis", "Cancellation Analysis", "Guest Analysis", "Advanced Analysis"])

if page == "Overview":
    overview(df)
elif page == "Booking Analysis":
    booking_analysis(df)
elif page == "ADR Analysis":
    adr_analysis(df)
elif page == "Cancellation Analysis":
    cancellation_analysis(df)
elif page == "Guest Analysis":
    guest_analysis(df)
elif page == "Advanced Analysis":
    advanced_analysis(df)
