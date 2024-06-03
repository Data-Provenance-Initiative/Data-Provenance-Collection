import pandas as pd
import altair as alt
from statsmodels.tsa.ar_model import AutoReg
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

#disable user warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_lagged_features(df, lags):
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    return df.dropna()


def forecast_and_plot(df, agent, lags):
    # Filter the DataFrame for the specific agent
    agent_df = df[df['agent'] == agent].copy()
    
    # Convert 'period' to timestamp if it's a Period object
    agent_df.loc[:, 'period'] = agent_df['period'].apply(lambda x: x.to_timestamp() if isinstance(x, pd.Period) else x)
    
    # Reshape the data
    pivoted_df = agent_df.pivot_table(index='period', columns='status', values='count')
    
    # Normalize the counts to percentages
    pivoted_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0) * 100
    
    # Doing these for each status individually
    status_dfs = {}
    for status in pivoted_df.columns:
        status_df = pivoted_df[[status]].reset_index()
        status_df.columns = ['ds', 'y']
        status_df.set_index('ds', inplace=True)
        status_dfs[status] = status_df
    
    # Fit model
    models = {}
    for status, status_df in status_dfs.items():
        model = AutoReg(status_df['y'], lags=lags)
        models[status] = model.fit()
    
    # Periods to predict (months)
    n_periods = 12  
    
    # Make future predictions
    future_periods = pd.date_range(start=agent_df['period'].max(), periods=n_periods, freq='M')
    predictions = {}
    conf_intervals = {}
    for status, model in models.items():
        forecast = model.predict(start=len(status_df), end=len(status_df) + n_periods - 1)
        conf_int = model.get_prediction(start=len(status_df), end=len(status_df) + n_periods - 1).conf_int()
        predictions[status] = forecast.values
        conf_intervals[status] = conf_int
    # Combine the predictions into a single DataFrame
    predicted_df = pd.DataFrame(predictions, index=future_periods)
    predicted_df = predicted_df.reset_index().melt(id_vars='index', var_name='status', value_name='count')
    predicted_df.columns = ['period', 'status', 'count']

    predicted_df['agent'] = agent
    
    # Concatenate the original and predicted DataFrames
    combined_df = pd.concat([agent_df, predicted_df], ignore_index=True)
    
    # Define the color scheme for the statuses
    status_colors = {'no_robots': 'gray', 'none': 'blue', 'some': 'orange', 'all': 'red'}
    
    chart = robots_util.plot_robots_time_map_altair(
        combined_df, 
        agent_type=agent, 
        period_col='period', 
        status_col='status', 
        val_col='count', 
        title='Restriction Status over Time', 
        ordered_statuses=['no_robots', 'none', 'some', 'all'], 
        status_colors=status_colors,
        datetime_swap=True,
    )
     # map the confidence intervals to the predicted df
    for status, conf_int in conf_intervals.items():
        # Ensure the length of the confidence intervals matches the number of future periods
        predicted_df.loc[predicted_df['status'] == status, 'lower'] = conf_int['lower'].values
        predicted_df.loc[predicted_df['status'] == status, 'upper'] = conf_int['upper'].values

    return chart, predicted_df

def forecast_and_plot_prophet(df, agent, lags):
    # Pick agent
    agent_df = df[df['agent'] == agent].copy()
    
    # Convert 'period' to timestamp if it's a Period object
    agent_df.loc[:, 'period'] = agent_df['period'].apply(lambda x: x.to_timestamp() if isinstance(x, pd.Period) else x)
    
    # Reshape the data
    pivoted_df = agent_df.pivot_table(index='period', columns='status', values='count')
    
    # Normalize the counts to percentages
    pivoted_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0) * 100
    
    # Create separate DataFrames for each status
    status_dfs = {}
    for status in pivoted_df.columns:
        status_df = pivoted_df[[status]].reset_index()
        status_df.columns = ['ds', 'y']
        status_df = create_lagged_features(status_df, lags)
        status_dfs[status] = status_df
    
    # Train time series models for each status
    models = {}
    for status, status_df in status_dfs.items():
        model = Prophet()
        for lag in lags:
            model.add_regressor(f'lag_{lag}')
        model.fit(status_df)
        models[status] = model
    
    # Define the number of future periods
    n_periods = 12 
    
    # Make future predictions
    future_periods = pd.date_range(start=agent_df['period'].max(), periods=n_periods, freq='M')
    predictions = {}
    for status, model in models.items():
        future_df = pd.DataFrame({'ds': future_periods})
        for lag in lags:
            future_df[f'lag_{lag}'] = status_dfs[status][f'lag_{lag}'].iloc[-1]
        forecast = model.predict(future_df)
        predictions[status] = forecast['yhat'].values
    
    # Combine the predictions into a single DataFrame
    predicted_df = pd.DataFrame(predictions, index=future_periods)
    predicted_df = predicted_df.reset_index().melt(id_vars='index', var_name='status', value_name='predicted_value')
    predicted_df.columns = ['period', 'status', 'predicted_value']
    # add agent column
    predicted_df['agent'] = agent
    # add tokens column
    predicted_df['tokens'] = predicted_df['predicted_value']
    # Define the color scheme for the statuses
    status_colors = {'no_robots': 'gray', 'none': 'blue', 'some': 'orange', 'all': 'red'}
    
    chart = robots_util.plot_robots_time_map_altair(
        predicted_df, 
        agent_type=agent, 
        period_col='period', 
        status_col='status', 
        val_col='tokens',  # "count" / "tokens"
        title='Restriction Status over Time', 
        ordered_statuses=['no_robots', 'none', 'some', 'all'], 
        status_colors=status_colors,
        datetime_swap=True,
    )
    
    return chart

def forecast_and_plot_arima(df, agent, lags):
    # Pick agent
    agent_df = df[df['agent'] == agent].copy()
    
    # Convert 'period' to timestamp if it's a Period object
    agent_df.loc[:, 'period'] = agent_df['period'].apply(lambda x: x.to_timestamp() if isinstance(x, pd.Period) else x)
    
    # Reshape the data
    pivoted_df = agent_df.pivot_table(index='period', columns='status', values='count')
    
    # Normalize the counts to percentages
    pivoted_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0) * 100
    
    # Create separate DataFrames for each status
    status_dfs = {}
    for status in pivoted_df.columns:
        status_df = pivoted_df[[status]].reset_index()
        status_df.columns = ['ds', 'y']
        status_df = create_lagged_features(status_df, lags)
        status_dfs[status] = status_df
    
    # Train ARIMA models for each status
    models = {}
    for status, status_df in status_dfs.items():
        model = ARIMA(status_df['y'], order=(max(lags), 0, 0))
        models[status] = model.fit()
    
    # Define the number of future periods
    n_periods = 12 
    
    # Make future predictions
    future_periods = pd.date_range(start=agent_df['period'].max(), periods=n_periods, freq='M')
    predictions = {}
    for status, model in models.items():
        forecast = model.forecast(steps=n_periods)
        predictions[status] = forecast.values
    
    # Combine the predictions into a single DataFrame
    predicted_df = pd.DataFrame(predictions, index=future_periods)
    predicted_df = predicted_df.reset_index().melt(id_vars='index', var_name='status', value_name='predicted_value')
    predicted_df.columns = ['period', 'status', 'predicted_value']
    # add agent column
    predicted_df['agent'] = agent
    # add tokens column
    predicted_df['tokens'] = predicted_df['predicted_value']
    # Define the color scheme for the statuses
    status_colors = {'no_robots': 'gray', 'none': 'blue', 'some': 'orange', 'all': 'red'}
    
    chart = robots_util.plot_robots_time_map_altair(
        predicted_df, 
        agent_type=agent, 
        period_col='period', 
        status_col='status', 
        val_col='tokens', 
        title='Restriction Status over Time', 
        ordered_statuses=['no_robots', 'none', 'some', 'all'], 
        status_colors=status_colors,
        datetime_swap=True,
    )
    
    return chart

def forecast_and_plot_sarima(df, agent, lags, seasonal_order):
    # Pick agent
    agent_df = df[df['agent'] == agent].copy()
    
    # Convert 'period' to timestamp if it's a Period object
    agent_df.loc[:, 'period'] = agent_df['period'].apply(lambda x: x.to_timestamp() if isinstance(x, pd.Period) else x)
    
    # Reshape the data
    pivoted_df = agent_df.pivot_table(index='period', columns='status', values='count')
    
    # Normalize the counts to percentages
    pivoted_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0) * 100
    
    # Create separate DataFrames for each status
    status_dfs = {}
    for status in pivoted_df.columns:
        status_df = pivoted_df[[status]].reset_index()
        status_df.columns = ['ds', 'y']
        status_df = create_lagged_features(status_df, lags)
        status_dfs[status] = status_df
    
    # Train SARIMA models for each status
    models = {}
    for status, status_df in status_dfs.items():
        model = SARIMAX(status_df['y'], order=(max(lags), 0, 0), seasonal_order=seasonal_order)
        models[status] = model.fit(disp=False)
    
    # Define the number of future periods
    n_periods = 12 
    
    # Make future predictions
    future_periods = pd.date_range(start=agent_df['period'].max(), periods=n_periods, freq='M')
    predictions = {}
    for status, model in models.items():
        forecast = model.get_forecast(steps=n_periods)
        predictions[status] = forecast.predicted_mean.values
    
    # Combine the predictions into a single DataFrame
    predicted_df = pd.DataFrame(predictions, index=future_periods)
    predicted_df = predicted_df.reset_index().melt(id_vars='index', var_name='status', value_name='predicted_value')
    predicted_df.columns = ['period', 'status', 'predicted_value']
    # add agent column
    predicted_df['agent'] = agent
    # add tokens column
    predicted_df['tokens'] = predicted_df['predicted_value']
    # Define the color scheme for the statuses
    status_colors = {'no_robots': 'gray', 'none': 'blue', 'some': 'orange', 'all': 'red'}
    
    chart = robots_util.plot_robots_time_map_altair(
        predicted_df, 
        agent_type=agent, 
        period_col='period', 
        status_col='status', 
        val_col='tokens', 
        title='Restriction Status over Time', 
        ordered_statuses=['no_robots', 'none', 'some', 'all'], 
        status_colors=status_colors,
        datetime_swap=True,
    )
    
    return chart

def analyze_robots(
    df, analysis_type, chosen_corpus,
    lags, seasonal_order=None, display=False):
    """
    Analyzes robot data for different agents using specified forecasting methods.

    Parameters:
    - analysis_type (str): Type of analysis to perform. Options are 'autoregression', 'prophet', 'arima', 'sarima'.
    - lags (list): List of lag values to be used in the forecasting models.
    - seasonal_order (tuple, optional): Seasonal order parameters for SARIMA model. Default is None.
    - display (bool, optional): If True, displays the predicted DataFrame. Default is False.

    Returns:
    - None: Displays the chart for each agent.
    """
    agents = df['agent'].unique()
    
    for agent in agents:
        print(f"CHOSEN_CORPUS: {chosen_corpus}")
        print(f"AGENT: {agent}")
        
        if analysis_type == 'autoregression':
            chart, predicted_df = forecast_and_plot(df, agent, lags)
        elif analysis_type == 'prophet':
            chart = forecast_and_plot_prophet(df, agent, lags)
        elif analysis_type == 'arima':
            chart = forecast_and_plot_arima(df, agent, lags)
        elif analysis_type == 'sarima':
            chart = forecast_and_plot_sarima(df, agent, lags, seasonal_order)
        else:
            raise ValueError("Invalid analysis type specified.")
        
        if display & (analysis_type == 'autoregression'):
            display(predicted_df)
        chart.show()