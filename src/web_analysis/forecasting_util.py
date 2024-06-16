import os
import pandas as pd
import altair as alt
from statsmodels.tsa.ar_model import AutoReg
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import typing

from . import robots_util
from analysis import visualization_util

# disable user warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_lagged_features(df, lags):
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    return df.dropna()


# def forecast_and_plot(
#     df,
#     agent,
#     lags,
#     status_colors,
#     ordered_statuses,
#     val_col,
#     detailed=False
# ):
#     # Filter the DataFrame for the specific agent
#     agent_df = df[df["agent"] == agent].copy()

#     # Convert the column
#     def convert_period_to_timestamp(x):
#         # if isinstance(x, pd.Period):
#         #     print('yes')
#         #     return x.to_timestamp()
#         # return x
#         return x.to_timestamp()

#     agent_df["period"] = agent_df["period"].apply(convert_period_to_timestamp)

#     # Convert 'period' to timestamp if it's a Period object
#     # agent_df.loc[:, 'period'] = agent_df['period'].apply(lambda x: x.to_timestamp() if isinstance(x, pd.Period) else x)

#     # print(agent_df.dtypes)
#     # Reshape the data
#     pivoted_df = agent_df.pivot_table(index="period", columns="status", values=val_col)

#     # Normalize the counts to percentages
#     pivoted_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0) * 100

#     # Doing these for each status individually
#     status_dfs = {}
#     for status in pivoted_df.columns:
#         status_df = pivoted_df[[status]].reset_index()
#         status_df.columns = ["ds", "y"]
#         status_df.set_index("ds", inplace=True)
#         status_dfs[status] = status_df

#     # Fit model
#     models = {}
#     for status, status_df in status_dfs.items():
#         model = AutoReg(status_df["y"], lags=lags)
#         models[status] = model.fit()

#     # Periods to predict (months)
#     n_periods = 6

#     # Make future predictions
#     # print(agent_df.dtypes)
#     future_periods = pd.date_range(
#         start=agent_df["period"].max(), periods=n_periods, freq="M"
#     )
#     predictions = {}
#     conf_intervals = {}
#     for status, model in models.items():
#         forecast = model.predict(
#             start=len(status_df), end=len(status_df) + n_periods - 1
#         )
#         conf_int = model.get_prediction(
#             start=len(status_df), end=len(status_df) + n_periods - 1
#         ).conf_int()
#         predictions[status] = forecast.values
#         conf_intervals[status] = conf_int
#     # Combine the predictions into a single DataFrame
#     predicted_df = pd.DataFrame(predictions, index=future_periods)
#     predicted_df = predicted_df.reset_index().melt(
#         id_vars="index", var_name="status", value_name=val_col
#     )
#     predicted_df.columns = ["period", "status", val_col]

#     predicted_df["agent"] = agent

#     # Concatenate the original and predicted DataFrames
#     combined_df = pd.concat([agent_df, predicted_df], ignore_index=True)

#     if detailed:
#         chart = robots_util.plot_robots_time_map_altair_detailed(
#             combined_df,
#             agent_type=agent,
#             period_col="period",
#             status_col="status",
#             val_col=val_col,
#             title="Restriction Status over Time",
#             ordered_statuses=ordered_statuses,
#             status_colors=status_colors,
#             datetime_swap=True,
#         )
#     else:
#         chart = robots_util.plot_robots_time_map_altair(
#             combined_df,
#             agent_type=agent,
#             period_col="period",
#             status_col="status",
#             val_col=val_col,
#             title="Restriction Status over Time",
#             ordered_statuses=ordered_statuses,
#             status_colors=status_colors,
#             datetime_swap=True,
#         )
#     # map the confidence intervals to the predicted df
#     for status, conf_int in conf_intervals.items():
#         # Ensure the length of the confidence intervals matches the number of future periods
#         predicted_df.loc[predicted_df["status"] == status, "lower"] = conf_int[
#             "lower"
#         ].values
#         predicted_df.loc[predicted_df["status"] == status, "upper"] = conf_int[
#             "upper"
#         ].values

#     return chart, predicted_df


def forecast_and_plot(
    df: pd.DataFrame,
    agent: str,
    lags: typing.List[int],
    status_colors: typing.Dict[str, str],
    ordered_statuses: typing.List[str],
    val_col: str = "count",
    detailed: bool = False,
    n_periods: int = 6,
    **plot_kwargs: dict,
) -> alt.Chart:
    # Filter the DataFrame for the specific agent
    agent_df = df[df["agent"] == agent].copy()

    # Convert 'period' to timestamp if it's a Period object
    agent_df["period"] = agent_df["period"].apply(lambda x: x.to_timestamp())

    # Reshape the data
    pivoted_df = agent_df.pivot_table(index="period", columns="status", values=val_col)

    # Normalize the counts to percentages
    pivoted_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0) * 100

    # Doing these for each status individually
    status_dfs = {}
    for status in pivoted_df.columns:
        status_df = pivoted_df[[status]].reset_index()
        status_df.columns = ["ds", "y"]
        status_df.set_index("ds", inplace=True)
        status_dfs[status] = status_df

    # Fit model
    models = {}
    for status, status_df in status_dfs.items():
        model = AutoReg(status_df["y"], lags=lags)
        models[status] = model.fit()

    # Make future predictions
    future_periods = pd.date_range(
        start=agent_df["period"].max(), periods=n_periods, freq="M"
    )

    predictions = {}
    conf_intervals = {}

    # Make predictions per status
    for status, model in models.items():
        forecast = model.predict(
            start=len(status_df), end=len(status_df) + n_periods - 1
        )
        conf_int = model.get_prediction(
            start=len(status_df), end=len(status_df) + n_periods - 1
        ).conf_int()
        predictions[status] = forecast.values
        conf_intervals[status] = conf_int

    # Combine the predictions into a single DataFrame
    predicted_df = pd.DataFrame(predictions, index=future_periods)
    predicted_df = predicted_df.reset_index().melt(
        id_vars="index", var_name="status", value_name=val_col
    )
    predicted_df.columns = ["period", "status", val_col]

    predicted_df["agent"] = agent

    # Concatenate the original and predicted DataFrames
    combined_df = pd.concat([agent_df, predicted_df], ignore_index=True)

    forecast_startdate = predicted_df["period"].min()

    if detailed:
        chart = robots_util.plot_robots_time_map_altair_detailed(
            combined_df,
            agent_type=agent,
            period_col="period",
            status_col="status",
            val_col=val_col,
            ordered_statuses=ordered_statuses,
            status_colors=status_colors,
            datetime_swap=True,
            forecast_startdate=forecast_startdate,
            **plot_kwargs,
        )
    else:
        chart = robots_util.plot_robots_time_map_altair(
            combined_df,
            agent_type=agent,
            period_col="period",
            status_col="status",
            val_col=val_col,
            title="Restriction Status over Time",
            ordered_statuses=ordered_statuses,
            status_colors=status_colors,
            datetime_swap=True,
            forecast_startdate=forecast_startdate,
            **plot_kwargs,
        )
    # map the confidence intervals to the predicted df
    for status, conf_int in conf_intervals.items():
        # Ensure the length of the confidence intervals matches the number of future periods
        predicted_df.loc[predicted_df["status"] == status, "lower"] = conf_int[
            "lower"
        ].values
        predicted_df.loc[predicted_df["status"] == status, "upper"] = conf_int[
            "upper"
        ].values

    return chart, predicted_df


def forecast_and_plot_prophet(
    df: pd.DataFrame,
    agent: str,
    lags: typing.List[int],
    status_colors: typing.Dict[str, str],
    ordered_statuses: typing.List[str],
    n_periods: int = 6,
    **plot_kwargs: dict,
) -> alt.Chart:
    # Pick agent
    agent_df = df[df["agent"] == agent].copy()

    # Convert 'period' to timestamp if it's a Period object
    agent_df["period"] = agent_df["period"].apply(lambda x: x.to_timestamp())

    # Reshape the data
    pivoted_df = agent_df.pivot_table(index="period", columns="status", values="count")

    # Normalize the counts to percentages
    pivoted_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0) * 100

    # Create separate DataFrames for each status
    status_dfs = {}
    for status in pivoted_df.columns:
        status_df = pivoted_df[[status]].reset_index()
        status_df.columns = ["ds", "y"]
        status_df = create_lagged_features(status_df, lags)
        status_dfs[status] = status_df

    # Train time series models for each status
    models = {}
    for status, status_df in status_dfs.items():
        model = Prophet()
        for lag in lags:
            model.add_regressor(f"lag_{lag}")
        model.fit(status_df)
        models[status] = model

    # Make future predictions
    future_periods = pd.date_range(
        start=agent_df["period"].max(), periods=n_periods, freq="M"
    )
    predictions = {}

    # Make predictions per status
    for status, model in models.items():
        future_df = pd.DataFrame({"ds": future_periods})
        for lag in lags:
            future_df[f"lag_{lag}"] = status_dfs[status][f"lag_{lag}"].iloc[-1]
        forecast = model.predict(future_df)
        predictions[status] = forecast["yhat"].values

    # Combine the predictions into a single DataFrame
    predicted_df = pd.DataFrame(predictions, index=future_periods)
    predicted_df = predicted_df.reset_index().melt(
        id_vars="index", var_name="status", value_name="predicted_value"
    )
    predicted_df.columns = ["period", "status", "predicted_value"]
    # add agent column
    predicted_df["agent"] = agent
    # add tokens column
    predicted_df["tokens"] = predicted_df["predicted_value"]
    # Define the color scheme for the statuses

    combined_df = pd.concat([agent_df, predicted_df], ignore_index=True)

    forecast_startdate = predicted_df["period"].min()

    chart = robots_util.plot_robots_time_map_altair_detailed(
        combined_df,
        agent_type=agent,
        period_col="period",
        status_col="status",
        val_col="tokens",  # "count" / "tokens"
        ordered_statuses=ordered_statuses,
        status_colors=status_colors,
        datetime_swap=True,
        forecast_startdate=forecast_startdate,
        **plot_kwargs,
    )

    return chart


def forecast_and_plot_arima(
    df: pd.DataFrame,
    agent: str,
    lags: typing.List[int],
    status_colors: typing.Dict[str, str],
    ordered_statuses: typing.List[str],
    n_periods: int = 6,
    **plot_kwargs: dict,
) -> alt.Chart:
    # Pick agent
    agent_df = df[df["agent"] == agent].copy()

    # Convert 'period' to timestamp if it's a Period object
    agent_df["period"] = agent_df["period"].apply(lambda x: x.to_timestamp())

    # Reshape the data
    pivoted_df = agent_df.pivot_table(index="period", columns="status", values="count")

    # Normalize the counts to percentages
    pivoted_df = pivoted_df.div(pivoted_df.sum(axis=1), axis=0) * 100

    # Create separate DataFrames for each status
    status_dfs = {}
    for status in pivoted_df.columns:
        status_df = pivoted_df[[status]].reset_index()
        status_df.columns = ["ds", "y"]
        status_df = create_lagged_features(status_df, lags)
        status_dfs[status] = status_df

    # Train ARIMA models for each status
    models = {}
    for status, status_df in status_dfs.items():
        model = ARIMA(status_df["y"], order=(max(lags), 0, 0))
        models[status] = model.fit()

    # Make future predictions
    future_periods = pd.date_range(
        start=agent_df["period"].max(), periods=n_periods, freq="M"
    )
    predictions = {}
    for status, model in models.items():
        forecast = model.forecast(steps=n_periods)
        predictions[status] = forecast.values

    # Combine the predictions into a single DataFrame
    predicted_df = pd.DataFrame(predictions, index=future_periods)
    predicted_df = predicted_df.reset_index().melt(
        id_vars="index", var_name="status", value_name="predicted_value"
    )

    predicted_df.columns = ["period", "status", "predicted_value"]
    # add agent column
    predicted_df["agent"] = agent
    # add tokens column
    predicted_df["tokens"] = predicted_df["predicted_value"]
    # Define the color scheme for the statuses
    # status_colors = {
    #     "no_robots": "gray",
    #     "none": "blue",
    #     "some": "orange",
    #     "all": "red",
    # }

    combined_df = pd.concat([agent_df, predicted_df], ignore_index=True)

    forecast_startdate = predicted_df["period"].min()

    chart = robots_util.plot_robots_time_map_altair(
        combined_df,
        agent_type=agent,
        period_col="period",
        status_col="status",
        val_col="tokens",
        ordered_statuses=ordered_statuses,
        status_colors=status_colors,
        datetime_swap=True,
        forecast_startdate=forecast_startdate,
        **plot_kwargs,
    )

    return chart


def forecast_and_plot_sarima(
    df: pd.DataFrame,
    agent: str,
    order: typing.Tuple[int, int, int],
    seasonal_order: typing.Tuple[int, int, int, int],
    status_colors: typing.Dict[str, str],
    ordered_statuses: typing.List[str],
    n_periods: int = 12,
    excel_file: str = "forecasted_robots_data.xlsx",
    chosen_corpus: str = "c4",
    **plot_kwargs: dict,
) -> alt.Chart:
    filtered_df = df[df["agent"] == agent].copy()
    grouped_df = (
        filtered_df.groupby(["period", "status"])["tokens"].sum().unstack(fill_value=0)
    )
    grouped_df = grouped_df[ordered_statuses]
    total_counts = grouped_df.sum(axis=1)
    percent_df = grouped_df.div(total_counts, axis=0).reset_index()
    percent_df["period"] = percent_df["period"].apply(lambda x: x.to_timestamp())
    # Reshape the data
    pivoted_df = percent_df.melt(
        id_vars="period", var_name="status", value_name="percentage"
    )
    pivoted_df["tokens"] = pivoted_df["percentage"]
    pivoted_df["agent"] = agent

    # Fit SARIMA or ARIMA model for each status
    predicted_data = []
    for status in ordered_statuses:
        status_data = pivoted_df[pivoted_df["status"] == status]
        if seasonal_order is not None:
            model = SARIMAX(
                status_data["percentage"], order=order, seasonal_order=seasonal_order
            )
        else:
            model = ARIMA(status_data["percentage"], order=order)
        models = model.fit(disp=False)
        # Make future predictions
        future_periods = pd.date_range(
            start=percent_df["period"].max() + pd.DateOffset(months=1),
            periods=n_periods,
            freq="M",
        )
        forecast = models.forecast(steps=n_periods)
        forecast[forecast < 0] = 0  # Clip negative values to zero
        conf_int = models.get_forecast(steps=n_periods).conf_int()
        predicted_df = pd.DataFrame(
            {"period": future_periods, "status": status, "percentage": forecast}
        )
        predicted_df["agent"] = agent
        predicted_df["tokens"] = predicted_df["percentage"]
        predicted_df["lower"] = conf_int.iloc[:, 0]
        predicted_df["upper"] = conf_int.iloc[:, 1]
        predicted_data.append(predicted_df)

    predicted_data = pd.concat(predicted_data)
    predicted_data["total_percentage"] = predicted_data.groupby("period")[
        "percentage"
    ].transform("sum")
    predicted_data["percentage"] = (
        predicted_data["percentage"] / predicted_data["total_percentage"]
    )
    predicted_data.drop("total_percentage", axis=1, inplace=True)
    combined_df = pd.concat([pivoted_df, predicted_data], ignore_index=True)
    forecast_startdate = predicted_data["period"].min()

    merged_df = pd.merge(
        percent_df.melt(
            id_vars=["period"], var_name="status", value_name="Actual Percentage"
        ),
        predicted_data[["period", "status", "percentage"]].rename(
            columns={"percentage": "Forecasted Percentage"}
        ),
        on=["period", "status"],
        how="outer",
    )
    merged_df["Agent"] = agent

    # Reorder columns
    merged_df = merged_df[
        ["period", "status", "Actual Percentage", "Forecasted Percentage", "Agent"]
    ]

    if os.path.exists(excel_file):
        # If the file exists, append the new data to a new sheet
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a") as writer:
            merged_df.to_excel(writer, sheet_name=chosen_corpus, index=False)
    else:
        # If the file doesn't exist, create a new Excel file with the data
        with pd.ExcelWriter(excel_file) as writer:
            merged_df.to_excel(writer, sheet_name=chosen_corpus, index=False)

    chart = robots_util.plot_robots_time_map_altair(
        combined_df,
        agent_type=agent,
        period_col="period",
        status_col="status",
        val_col="tokens",
        title="Restriction Status over Time",
        ordered_statuses=ordered_statuses,
        status_colors=status_colors,
        datetime_swap=True,
        forecast_startdate=forecast_startdate,
        configure=False,
        **plot_kwargs,
    )
    return chart


def analyze_robots(
    df,
    agent,
    analysis_type,
    chosen_corpus,
    lags,
    seasonal_order=None,
    display=False,
    status_colors=None,
    ordered_statuses=None,
    val_col="tokens",
    detailed=False,
):
    """
    Analyzes robot data for different agents using specified forecasting methods.

    Parameters:
    - analysis_type (str): Type of analysis to perform. Options are 'autoregression', 'prophet', 'arima', 'sarima'.
    - lags (list): List of lag values to be used in the forecasting models.
    - seasonal_order (tuple, optional): Seasonal order parameters for SARIMA model. Default is None.
    - display (bool, optional): If True, displays the predicted DataFrame. Default is False.
    - status_colors (dict, optional): Dictionary mapping status values to colors. Default is None.
    - ordered_statuses (list, optional): List of status values in the order they should be displayed. Default is None.
    - val_col (str, optional): Column to use for the value in the plot. Default is 'tokens'.

    Returns:
    - None: Displays the chart for each agent.
    """
    # agents = df["agent"].unique()

    # for agent in agents:
    print(f"CHOSEN_CORPUS: {chosen_corpus}")
    print(f"AGENT: {agent}")

    if analysis_type == "autoregression":
        chart, predicted_df = forecast_and_plot(
            df, agent, lags, status_colors, ordered_statuses, val_col, detailed
        )
    elif analysis_type == "prophet":
        chart = forecast_and_plot_prophet(df, agent, lags)
    elif analysis_type == "arima":
        chart = forecast_and_plot_arima(df, agent, lags)
    elif analysis_type == "sarima":
        chart = forecast_and_plot_sarima(df, agent, lags, seasonal_order)
    else:
        raise ValueError("Invalid analysis type specified.")

    if display & (analysis_type == "autoregression"):
        display(predicted_df)
    chart.show()


def forecast_company_comparisons_autoregression(df, lags, val_col, n_periods=6):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame

    df["period"] = df["period"].apply(lambda x: x.to_timestamp())

    # Calculate the percentage of tokens for the 'Restrictive' status
    total_tokens = df.groupby(["period", "agent"])[val_col].sum().reset_index()
    restrictive_tokens = (
        df[df["status"] == "all"]
        .groupby(["period", "agent"])[val_col]
        .sum()
        .reset_index()
    )
    data = pd.merge(
        total_tokens, restrictive_tokens, on=["period", "agent"], how="left"
    ).fillna(0)
    data["percent_Restrictive"] = (data["tokens_y"] / data["tokens_x"]) * 100
    data = data[["period", "agent", "percent_Restrictive"]]

    # Fit model and make predictions for each agent
    agents = data["agent"].unique()
    predicted_data = []
    for agent in agents:
        agent_data = data[data["agent"] == agent]
        agent_data = agent_data.set_index("period")

        # Reshape the data
        pivoted_df = agent_data.pivot_table(
            index="period", columns="agent", values="percent_Restrictive"
        )

        # Fit model
        model = AutoReg(pivoted_df.values.flatten(), lags=lags)
        models = model.fit()

        # Make future predictions
        future_periods = pd.date_range(
            start=agent_data.index.max(), periods=n_periods, freq="M"
        )
        forecast = models.predict(
            start=len(pivoted_df), end=len(pivoted_df) + n_periods - 1
        )
        conf_int = models.get_prediction(
            start=len(pivoted_df), end=len(pivoted_df) + n_periods - 1
        ).conf_int()

        predicted_df = pd.DataFrame(
            {"period": future_periods, "agent": agent, "percent_Restrictive": forecast}
        )
        # print(predicted_df)
        # predicted_df["lower"] = conf_int["lower"]
        # predicted_df["upper"] = conf_int["upper"]
        predicted_data.append(predicted_df)

    # Concatenate the original and predicted data
    predicted_data = pd.concat(predicted_data)
    combined_df = pd.concat([data, predicted_data])

    return combined_df


def forecast_company_comparisons_sarima(
    df: pd.DataFrame,
    val_col: str,
    n_periods: int = 12,
    order: typing.Tuple[int, int, int] = (2, 1, 1),
    seasonal_order: typing.Tuple[int, int, int, int] = (1, 1, 1, 4),
) -> pd.DataFrame:
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame

    def convert_period_to_timestamp(x):
        return x.to_timestamp()

    df["period"] = df["period"].apply(convert_period_to_timestamp)

    # Calculate the percentage of tokens for the 'Restrictive' status
    total_tokens = df.groupby(["period", "agent"])[val_col].sum().reset_index()
    restrictive_tokens = (
        df[df["status"] == "all"]
        .groupby(["period", "agent"])[val_col]
        .sum()
        .reset_index()
    )
    data = pd.merge(
        total_tokens, restrictive_tokens, on=["period", "agent"], how="left"
    ).fillna(0)
    data["percent_Restrictive"] = (data["tokens_y"] / data["tokens_x"]) * 100
    data = data[["period", "agent", "percent_Restrictive"]]

    pivoted_df = data.pivot_table(
        index="period", columns="agent", values="percent_Restrictive"
    )

    predicted_data = []
    for agent in pivoted_df.columns:
        agent_data = pivoted_df[agent]

        if seasonal_order is not None:
            model = SARIMAX(agent_data, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(agent_data, order=order)

        models = model.fit(disp=False)

        # Make future predictions
        future_periods = pd.date_range(
            start=data["period"].max() + pd.DateOffset(months=1),
            periods=n_periods,
            freq="M",
        )
        forecast = models.forecast(steps=n_periods)
        conf_int = models.get_forecast(steps=n_periods).conf_int()

        predicted_df = pd.DataFrame(
            {"period": future_periods, "agent": agent, "percent_Restrictive": forecast}
        )
        predicted_df["lower"] = conf_int.iloc[:, 0]
        predicted_df["upper"] = conf_int.iloc[:, 1]
        predicted_data.append(predicted_df)

    predicted_data = pd.concat(predicted_data)
    combined_df = pd.concat([data, predicted_data])

    return combined_df


def plot_and_forecast_tos_sarima(
    df: pd.DataFrame,
    period_col: str,
    status_col: str,
    val_col: str,
    title: str = "",
    ordered_statuses: typing.List[str] = None,
    status_colors: typing.Dict[str, str] = None,
    datetime_swap: bool = False,
    legend_cols: int = 1,
    vertical_line_dates: typing.List[str] = [],
    label_fontsize: int = 14,
    title_fontsize: int = 16,
    width: int = 1000,
    height: int = 200,
    forecast_startdate: str = None,
    configure: bool = False,
    n_periods: int = 12,
    order: typing.Tuple[int, int, int] = (2, 1, 2),
    seasonal_order: typing.Tuple[int, int, int, int] = (1, 1, 1, 4),
    excel_file: str = "forecasted_tos_data.xlsx",
    chosen_corpus: str = "c4",
    **plot_kwargs,
):
    # Group by 'period' and 'status', and sum up the 'count'
    grouped_df = (
        df.groupby([period_col, status_col])[val_col].sum().unstack(fill_value=0)
    )

    # Reorder the columns as desired
    if ordered_statuses is None:
        ordered_statuses = grouped_df.columns.tolist()

    grouped_df = grouped_df[ordered_statuses]

    # Calculate the total counts for each period
    total_counts = grouped_df.sum(axis=1)

    # Calculate the percentage of each status per period
    percent_df = grouped_df.div(total_counts, axis=0).reset_index()

    if datetime_swap:
        percent_df[period_col] = pd.to_datetime(percent_df[period_col])
    else:
        percent_df[period_col] = percent_df[period_col].dt.to_timestamp()

    # Reshape the data
    pivoted_df = percent_df.melt(
        id_vars=period_col, var_name=status_col, value_name="percentage"
    )
    pivoted_df[val_col] = pivoted_df["percentage"]

    # Fit SARIMA or ARIMA model for each status
    predicted_data = []
    for status in ordered_statuses:
        status_data = pivoted_df[pivoted_df[status_col] == status]
        if seasonal_order is not None:
            model = SARIMAX(
                status_data["percentage"], order=order, seasonal_order=seasonal_order
            )
        else:
            model = ARIMA(status_data["percentage"], order=order)
        models = model.fit(disp=False)
        # Make future predictions
        future_periods = pd.date_range(
            start=percent_df[period_col].max() + pd.DateOffset(months=1),
            periods=n_periods,
            freq="M",
        )
        forecast = models.forecast(steps=n_periods)
        forecast[forecast < 0] = 0  # Clip negative values to zero
        conf_int = models.get_forecast(steps=n_periods).conf_int()
        predicted_df = pd.DataFrame(
            {period_col: future_periods, status_col: status, "percentage": forecast}
        )
        predicted_df[val_col] = predicted_df["percentage"]
        predicted_df["lower"] = conf_int.iloc[:, 0]
        predicted_df["upper"] = conf_int.iloc[:, 1]
        predicted_data.append(predicted_df)

    predicted_data = pd.concat(predicted_data)
    predicted_data["total_percentage"] = predicted_data.groupby(period_col)[
        "percentage"
    ].transform("sum")
    predicted_data["percentage"] = (
        predicted_data["percentage"] / predicted_data["total_percentage"]
    )
    predicted_data.drop("total_percentage", axis=1, inplace=True)

    combined_df = pd.concat([pivoted_df, predicted_data], ignore_index=True)
    forecast_startdate = predicted_data[period_col].min()

    merged_df = pd.merge(
        pivoted_df.rename(columns={"percentage": "Actual Percentage"}),
        predicted_data[[period_col, status_col, "percentage"]].rename(
            columns={"percentage": "Forecasted Percentage"}
        ),
        on=[period_col, status_col],
        how="outer",
    )

    merged_df = merged_df[
        [period_col, status_col, "Actual Percentage", "Forecasted Percentage"]
    ]

    if os.path.exists(excel_file):
        # If the file exists, append the new data to a new sheet
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a") as writer:
            merged_df.to_excel(writer, sheet_name=chosen_corpus, index=False)
    else:
        with pd.ExcelWriter(excel_file) as writer:
            merged_df.to_excel(writer, sheet_name=chosen_corpus, index=False)

    chart = visualization_util.create_stacked_area_chart(
        df=combined_df,
        period_col=period_col,
        status_col=status_col,
        percentage_col="percentage",
        title=title,
        ordered_statuses=ordered_statuses,
        status_colors=status_colors,
        legend_cols=legend_cols,
        vertical_line_dates=vertical_line_dates,
        label_fontsize=label_fontsize,
        title_fontsize=title_fontsize,
        width=width,
        height=height,
        forecast_startdate=forecast_startdate,
        configure=configure,
        **plot_kwargs,
    )

    return chart


def forecast_restrictive_tokens_sarima(
    df: pd.DataFrame,
    n_periods: int = 12,
    order: typing.Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: typing.Tuple[int, int, int, int] = (1, 1, 1, 6),
) -> pd.DataFrame:
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame

    # Combine 'rand' and 'head' columns to calculate total restrictive tokens
    df["total_tokens"] = df["rand"] + df["head"]

    df["percent_restrictive"] = df["total_tokens"]

    df.set_index("period", inplace=True)

    model = SARIMAX(
        df["percent_restrictive"], order=order, seasonal_order=seasonal_order
    )
    model_fit = model.fit(disp=False)
    print(model_fit.summary())

    future_periods = pd.date_range(
        start=df.index.max() + pd.DateOffset(months=1), periods=n_periods, freq="M"
    )
    forecast = model_fit.forecast(steps=n_periods)
    conf_int = model_fit.get_forecast(steps=n_periods).conf_int()
    predicted_df = pd.DataFrame(
        {
            "period": future_periods,
            "forecasted_restrictive_tokens": forecast,
            "lower": conf_int.iloc[:, 0],
            "upper": conf_int.iloc[:, 1],
        }
    )

    combined_df = pd.concat([df.reset_index(), predicted_df])

    return combined_df
