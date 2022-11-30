# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:13:50 2022

@author: habdallah
"""


#==============================================================================
#Importing Data
#==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf 
from PIL import Image
from plotly import graph_objs as go
import pandas_datareader.data as web
import numpy as np
from plotly.subplots import make_subplots

#==============================================================================
# Summary 
#==============================================================================

def tab1():
   
    #Adding header and data source that will appear at the top of the tab.
    st.header('Summary')

    #We create a dictionary to get stock info
    @st.cache
    def GetCompanyInfo(ticker):
        return yf.Ticker(ticker).info
    
    if ticker != '':
        
        info = GetCompanyInfo(ticker)
        
        #Company description as a text and logo image
        st.write('**1. Info:**')
        st.image(info['logo_url'])
        st.write(info['longBusinessSummary'])
        
        #Specify the page format
        col1,col2 = st.columns([2,2])
        
        #To organize Tables next to each other i used 'with col'
        with col1:
            #Key statistics in a Table Format 
            st.write('**2. Key Statistics:**')
            keys = ['previousClose', 'open', 'bid', 'ask', 'marketCap', 'volume']
            company_stats = {}  # Dictionary
            for key in keys:
                company_stats.update({key:info[key]})
            company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
            st.dataframe(company_stats)
        with col2:
            #Major stock holders

            st.write('**3. Major Holder:**')
            st.dataframe(yf.Ticker(ticker).major_holders)
        
        #Create options to select the time period
        Period = ['1mo', '3mo','6mo','ytd','1y','3y','5y','max']
        default  = Period.index('1y')
        Period =  st.radio('Date Range', Period,horizontal = True,index = default)
        #interval 1 day
        data = yf.download(ticker, period = Period,interval = '1d')
        @st.cache
        def GetStockData(tickers, start_date, end_date):
            return pd.concat([yf.Ticker(ticker, start_date, end_date).info for tick in tickers])
        
        if ticker != '-':   
              data = yf.download(ticker, period = Period)
        #Tracing Area chart and Bar chart, and combining both together through secondary_y      
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'],name="Stock Value",showlegend=True,fill='tozeroy',fillcolor=('Green'),line=dict(color='green')),secondary_y=(False))
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'],name="Volume",showlegend=True,marker= dict(color='orange')),secondary_y=(True))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(title="Summary Graph", yaxis_title="Close Price")
        fig.update_layout(width = 1000 , height = 600)
        
        st.plotly_chart(fig)
    
#==============================================================================
# Chart
#==============================================================================

def tab2():
    
    #Adding header that will appear at the top of the tab.
    st.header("Chart")
        
    # Add table to show stock data
    @st.cache
    def GetStockData(tickers, start_date, end_date):
        return pd.concat([yf.Ticker(ticker, start_date, end_date).info for ticker in tickers])

    #Specify the page format
    col1,col2 = st.columns([2,2])
    
    with col2:
        #Create options to select the time period
        Date =  ['1d','1mo', '3mo','6mo','ytd','1y','3y','5y','max']
        default  = Date.index('1y')
        Date = st.radio("Date Range", Date,index = default,horizontal=True)
        
    with col1:
        #Create options to select the graph
        Graph = st.radio("Select Graph", ["Line","Candle"],horizontal=True)
        
        #Create options to select the time interval
        interval = ['1d','1mo']
        interval= st.radio("Select Interval", interval,horizontal=True)
        
        #getting the stock data 
        data = yf.download(ticker, period = Date,interval = interval)

#Line Chart    
    if Graph == "Line":     
       if ticker != '-':
           data = yf.download(ticker, period = Date,interval = interval)
           
           fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.07, subplot_titles=('Stock Trend', 'Volume'), 
               row_width=[0.2, 0.7])
           fig.add_trace(go.Scatter(x=data.index, y=data['Close'],name="Stock Trend",showlegend=True),row= 1,col = 1)
           fig.add_trace(go.Bar(x=data.index, y=data['Volume'],name="Volume",showlegend=True),row=2,col = 1)
           fig.update(layout_xaxis_rangeslider_visible=False)
           fig.update_layout(title="Line Plot", yaxis_title="Close Price")
           fig.update_layout(width = 1000 , height = 600)
           
           fig.update(layout_xaxis_rangeslider_visible=True)

           st.plotly_chart(fig)

#Cadle Stick
    elif Graph == "Candle":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.03, subplot_titles=('Stock Trend', 'Volume'), 
               row_width=[0.2, 0.7])

        fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                        low=data["Low"], close=data["Close"], name="Stock Trend"), 
                        row=1, col=1)
        fig.update_layout(title="Candlestick Plot", yaxis_title="Close Price")
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'],name="Volume",showlegend=True), row=2, col=1)
        
        #Moving average
        fig.add_trace(go.Scatter(x=data.index,y=data['Close'].rolling(window=50).mean(),marker_color='purple',name='Moving Average'))
        
        fig.update(layout_xaxis_rangeslider_visible=True)
    
        st.plotly_chart(fig)



#==============================================================================
# Financials 
#==============================================================================       
def tab3():
    #Adding header that will appear at the top of the tab.

     st.header("Financials")
     col1,col2 = st.columns([2,2])


#Creating Selection options for financials type and term and organizing them through 'col:' option
     with col1:
         
         Financials =  ['Income Statement', 'Balance Sheet','Cash Flow']
         fin =  st.radio("Show", Financials,horizontal=True)
         
     with col2:
         Term = ['Annual', 'Quarterly']
         term= st.radio(" ", Term)
     
     if ticker != '-':
#use st.text(fin) to get the name of the selected financial when chosen
            st.text(fin)
            #using if Statment we set a condition for each financial option to be downloaded
            #filled missing values by 0 , and tried to adjust table's measures.
            if fin == 'Income Statement' and term == 'Quarterly':
                st.dataframe(yf.Ticker(ticker).quarterly_financials.fillna(0),height=560,use_container_width=True)
            elif fin == 'Income Statement' and term == 'Annual':
                st.dataframe(yf.Ticker(ticker).financials.fillna(0),height=560,use_container_width=True)
                
            elif fin == 'Balance Sheet' and term == 'Quarterly':
                st.dataframe(yf.Ticker(ticker).quarterly_balance_sheet,height=850,use_container_width=True)        
            elif fin == 'Balance Sheet' and term == 'Annual':
                st.dataframe(yf.Ticker(ticker).balance_sheet,height=850,use_container_width=True)        

            elif fin == 'Cash Flow' and term == 'Quarterly':
                st.dataframe(yf.Ticker(ticker).quarterly_cashflow,height=700,use_container_width=True)          
            elif fin == 'Cash Flow' and term == 'Annual':
                st.dataframe(yf.Ticker(ticker).cashflow.fillna(0),height=700,use_container_width=True)             
        
            
                     
#==============================================================================
# Monte Carlo Simulation 
#==============================================================================       
def tab4():
    
    #Adding header that will appear at the top of the tab.
    st.header("Monte Carlo Simulation")
    
    #Dropdown to select number of simulations & time horizon
    col1,col2 = st.columns([2,2])
    select_sims =  [200, 500,1000]
    sim =  col1.radio("Select simulations", select_sims,horizontal=True)
    select_horizon = [30, 60,90]
    horizon= col2.radio("Select Horizon", select_horizon,horizontal=True)
    
    
    
    # Getting the Required stock price from Yahoo finance
    stock_price = web.DataReader(ticker, 'yahoo',start_date, end_date)
   
    #close price, daily return & daily volatility 
    close_price = stock_price['Close']
    daily_return = close_price.pct_change()
    daily_volatility = np.std(daily_return)

    
    #Monte Carlo simulation
    np.random.seed(123)
    simulations = sim
    time_horizone = horizon

    # Run the simulation
    simulation_df = pd.DataFrame()

    for i in range(simulations):
    
    # The list to store the next stock price
       next_price = []
    
       #Next stock price
       last_price = close_price[-1]
    
       for j in range(time_horizone):
           #Future return around the mean (0) and daily_volatility
           future_return = np.random.normal(0, daily_volatility)

           #Random future price
           future_price = last_price * (1 + future_return)

           # Save the price and go next
           next_price.append(future_price)
           last_price = future_price
    
       # Store the result of the simulation
       simulation_df[i] = next_price
    
    # Plotting the simulation stock price in the future
    #mfig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10, forward=True)

    plt.plot(simulation_df)
    plt.title('Monte Carlo simulation')
    plt.xlabel('Day')
    plt.ylabel('Price')

    plt.axhline(y=close_price[-1], color='red')
    plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
    ax.get_legend().legendHandles[0].set_color('red')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    # Ending price 
    ending_price = simulation_df.iloc[-1:, :].values[0, ]
    
    # Stock Price at 95% confidence interval
    future_price_95ci = np.percentile(ending_price, 5)
    
    # Finding out Value at Risk(VAR)
  
    VaR = close_price[-1] - future_price_95ci
    st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')  
    
    
    
    
    
#==============================================================================
# News
#==============================================================================       
def tab5():

    st.header("Stock news")
    def GetStocknews(ticker):
        return yf.Ticker(ticker).news
                
    if ticker != '-':
        news = GetStocknews(ticker)       
        for i in news:
            st.write(f'{i["title"]}\n{i["link"]}\n-Publisher: {i["publisher"]}')   
    
#==============================================================================
# Main body
#==============================================================================

def run():
    st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")
    st.title ("Financial dashboard")
     
    image = Image.open('C:/Users/habdallah/Desktop/MBD-S1/FP_Section4 (1)/FP_Section4/streamlit/img/investing.jpg')
    st.image(image)
    
    # Getting the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    
    
    #Ticker selection on the sidebar
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    
  
    #select Start & end dates
    global start_date, end_date
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col2.date_input("End date", datetime.today().date())
    
    
    #Refresh the form
    with st.sidebar:
         with st.form(key = "Refresh Form"):
              st.form_submit_button(label = "Update")
                      
    #radio box to select the tabs 
    select_tab = st.radio("Select tab", ['Summary','Chart','Financials','Monte Carlo Simulation','News'],horizontal=True)
    
    #Display the selected tab
    if select_tab == 'Summary':
        tab1()
        
    elif select_tab == 'Chart':
        tab2()
        
    elif select_tab == 'Financials':
        tab3()
        
    elif select_tab == 'Monte Carlo Simulation':
        tab4()
        
    elif select_tab == 'News':
        tab5()

        
    
    
if __name__ == "__main__":
    run()
    

