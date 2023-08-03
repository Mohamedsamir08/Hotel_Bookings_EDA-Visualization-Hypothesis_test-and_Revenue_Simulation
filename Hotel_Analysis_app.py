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

def introduction():
    st.title('Introduction')
    st.image('Hotel_Image.jpg', caption='Hotel Booking Analysis', use_column_width=True)
    st.header('Hotel booking')
    st.markdown('''
    Content:

    ### EDA

    The dataset contains data from two different hotels "Resort hotel and City hotel".

    The data contains "bookings due to arrive between the 1st of July of 2015 and the 31st of August 2017".

    ### Topics covered and questions to answer from the data:

    1-What is the distribution of the data?
                
    2-What is the total count of bookings per hotel?
                
    3-Where are the guests coming from?
                
    4-What is the average revenue per country?
                
    5-What is the distribution of the number of bookings by season?
                
    6-What are the total bookings versus total cancellations for the top 10 countries?
                
    7-What are the total bookings versus total cancellations by market segment?
                
    8-What are the total bookings versus total cancellations by deposit type?
                
    9-What is the distribution of guest types and average daily rates (ADR)?
                
    10-What is the percentage of repeated guests by deposit type?
                
    11-How many bookings were canceled?
                
    12-Which month has the highest number of cancellations?
                
    13-Does the lead time affect cancellation rates?
                
    14-What is the total average daily rate (ADR) for 'Not Canceled' and 'Canceled' bookings?
                
    15-Could the revenue increase if the hotel could reduce the cancellation rate?
                
    ''')

def overview(df):
    st.title("Numerical Variables Overview")

    st.subheader("Distribution of Key Numerical Variables")
    st.markdown("""
    Below are histograms showing the distribution of key numerical variables in the dataset. 

    - **Lead Time Distribution**: Most of the bookings have a lead time of less than 200 days. However, there are bookings with a lead time of up to 700 days. 
    - **ADR (Average Daily Rate) Distribution**: The ADR has a somewhat right-skewed distribution, with most of the rates less than 200. However, there are a few bookings with a higher rate.
    - **Arrival Date (Day of Month) Distribution**: The arrival day of the month is fairly distributed, with slight dips at the end of the month. except for the end of the month, which shows that we recieve a higher number of guests by the end of each month.
    - **Arrival Date (Week Number) Distribution**: The arrival week number shows a bimodal distribution, with peaks around week 30 (mid-July) and week 40 (early October). This suggests that there are more bookings during the summer.
    """)
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
    st.markdown("""
    - The distribution of bookings between the two types of hotels in the dataset shows that City Hotel has significantly more bookings than Resort Hotel.
    """)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(x='hotel', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Number of Bookings for Each Month")
    st.markdown("""- The number of bookings varies across the months, with August being the month with the most bookings and January being the month with the least bookings.""")
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='arrival_date_month', data=df, order=months_order, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Number of Bookings in Seasons")
    st.markdown("""- Most bookings are made in Summer, followed by Spring, Fall, and Winter.""")

    fig, ax = plt.subplots(figsize=(8,6))
    sns.countplot(x='season', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Top 10 Countries with the Most Bookings")
    st.markdown("""- the majority of bookings are from guests in Portugal (PRT), followed by Great Britain (GBR), France (FRA), Spain (ESP), and Germany (DEU). The other countries in the top 10 are Ireland (IRL), Italy (ITA), Belgium (BEL), Brazil (BRA), and the Netherlands (NLD).""")

    top_countries = df['country'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_countries.index, y=top_countries.values, ax=ax)
    st.pyplot(fig)

def adr_analysis(df):
    st.title("ADR (Average Daily Rate) Analysis")

    st.subheader("Average Daily Rate (ADR) by Hotel Type")
    st.markdown("""- The average daily rate (ADR) is higher for City Hotel compared to Resort Hotel.""")

    hotel_adr_mean = df.groupby('hotel')['adr'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='hotel', y='adr', data=hotel_adr_mean, ax=ax)
    st.pyplot(fig)

    st.subheader("Total ADR for Each Month")
    st.markdown("""- The average daily rate (ADR) is higher for City Hotel compared to Resort Hotel.""")
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    total_adr_month = df.groupby('arrival_date_month')['adr'].sum().reindex(months_order)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=total_adr_month.index, y=total_adr_month.values, ax=ax)
    st.pyplot(fig)

    st.subheader("Average ADR for Top 10 Countries with the Most Bookings")
    st.markdown("""- Guests from Portugal (PRT), which has the highest number of bookings, have a lower average ADR compared to some other countries.""")
    top_countries_adr = df['country'].value_counts().index[:10]
    df_top_countries_adr = df[df['country'].isin(top_countries_adr)]
    avg_adr_countries = df_top_countries_adr.groupby('country')['adr'].mean()
    colors = ['b', 'g', 'grey', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_adr_countries.sort_values(ascending=False).plot(kind='bar', color=colors, ax=ax)
    st.pyplot(fig)

def cancellation_analysis(df):
    st.subheader('Total Bookings vs Total Cancellations (Top 10 Countries)')
    st.markdown("""- The top 10 countries with the highest number of bookings also have a significant number of cancellations. Portugal (PRT) has the highest number of bookings and cancellations, indicating a high demand from this country but also a high likelihood of cancellation.""")
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
    st.markdown("""- Couples have the highest cancellation rate, followed by families, groups, and single guests.""")
    # Calculate the proportion of bookings that were cancelled for each guest type
    guest_cancellations = df.groupby('guest_type')['is_canceled'].mean()

    # Plot the results
    guest_cancellations.plot(kind='bar', figsize=(10, 6))
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader('Total Bookings vs Total Cancellations (Market Segment)')
    st.markdown("""- The 'Online TA' market segment has the highest number of bookings and cancellations, followed by 'Offline TA/TO' and 'Groups'. The 'Complementary', 'Aviation', and 'Undefined' segments have the least number of bookings and cancellations.""")

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
    st.markdown("""- The 'No Deposit' type has the highest number of bookings and cancellations, but the cancellation percentage is lower compared to the 'Non Refund' type, which has a cancellation rate of almost 100%. 'Refundable' deposits have the lowest number of bookings and cancellations, but their cancellation rate is similar to 'No Deposit'.""")
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
    st.markdown("""- Most of the guests are Couple, followed by Single, Family, and Group.""")
    guest_type_counts = df['guest_type'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(guest_type_counts, labels=guest_type_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader("ADR Share per Guest Type")
    st.markdown("""- Couple guests contribute the most to the total ADR, followed by Single, Family, and Group.""")
    total_adr_per_category = df.groupby('guest_type')['adr'].sum()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(total_adr_per_category, labels=total_adr_per_category.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    st.subheader('Repeated Guests by Deposit Type')
    st.markdown("""- The 'No Deposit' type also has the highest number of repeated guests, followed by 'Non Refund' and 'Refundable'. The percentage of repeated guests is highest for 'No Deposit', followed by 'Refundable' and 'Non Refund'.""")
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
    st.markdown("""- The mean lead time is around 104 days, with a standard deviation of 107 days. The upper control limit (USL) is around 425 days, and the lower control limit (LSL) is 0 days. That means that the majority of bookings are made within a year before the stay.""")
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
    st.markdown("""- Previous graph suggests that there is a positive relationship between lead time and cancellation. the longer the time between booking and actual stay, the more likely the booking is to be cancelled.""")
    bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660]
    df['lead_time_category'] = pd.cut(df['lead_time'], bins)
    cancellation_rates = df.groupby('lead_time_category')['is_canceled'].mean() * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=cancellation_rates.index, y=cancellation_rates.values, ax=ax)
    st.pyplot(fig)

    st.subheader("Sum of ADR for Each Category")
    st.markdown("""- The total ADR for 'Not Canceled' bookings is higher than for 'Canceled' bookings. This might indicate that more revenue is lost due to cancellations.""")
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
    st.markdown("""- From the chart, you can clearly see the potential increase in the sum of ADR if a certain percentage of the canceled bookings were converted to not canceled. This can provide an estimate of the potential revenue increase if the hotel can reduce the cancellation rate.
                - As we can see, there is significant potential to increase revenue by reducing the cancellation rate. Converting even a quarter of the cancelled bookings could result in an increase in total ADR by about 1.16 million. The potential increase grows with the conversion rate, reaching about 3.48 million when 75 percent of cancelled bookings are converted. 
                """)
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

    st.markdown("""- Based on the cancellation analysis, hypothesis test results, and Average Daily Rate (ADR) for canceled bookings we've conducted, we can suggest several strategies to mitigate the cancellation rate:

    1- **Targeted Marketing for High Cancellation Countries:**
    As Portugal accounts for a large proportion of both total bookings and cancellations, it could be beneficial to focus on this market to understand the reasons behind the high cancellation rate. This could involve surveys or market research to identify any issues or concerns Portuguese customers might have, and then addressing those issues in your marketing and service offerings.

    2- **Reducing Lead Time:**
    The Chi-square test indicated a significant relationship between lead time and cancellation rate. Therefore, strategies to reduce lead time could help decrease cancellations. This could involve offering incentives for last-minute bookings or implementing a dynamic pricing model where prices decrease as the booking date approaches.

    3- **Investing in Cancellation Prevention:**
    The analysis of ADR for canceled bookings showed that reducing the cancellation rate could lead to a significant increase in revenue. This indicates that investing in cancellation prevention could be highly profitable. This could involve improving the booking process, enhancing customer service, or offering flexible cancellation policies to prevent customers from cancelling their bookings in the first place.

    4- **Offering Flexible Plans:**
    Offering more flexible booking options may decrease the likelihood of cancellations. This could include options such as free cancellation up to a certain number of days before the stay, or the option to reschedule the booking without additional fees.

    5- **Loyalty Programs:**
    Implementing a loyalty program could also help reduce cancellation rates. If customers feel valued and receive additional benefits from a loyalty program, they may be less likely to cancel their bookings.""")



df = load_data()
page = st.sidebar.selectbox("Choose Analysis Type", ["Introduction", "Overview", "Booking Analysis", "ADR Analysis", "Cancellation Analysis", "Guest Analysis", "Advanced Analysis"])

if page == "Introduction":
    introduction()
elif page == "Overview":
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
