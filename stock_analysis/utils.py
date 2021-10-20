"""Utility functions for stock analysis."""
from functools import wraps
import re
import pandas as pd


# Functions related to Dataframe validation and cleaning


def string_handler(item):
    """
    Create a string out of an item if isn't it already.

    Parameters:
        - item: The variable to make sure it is a string.

    Returns:
        The input as a string.
    """
    return (
        str(item) 
        if not isinstance(item, str) else 
        item
    )


def iter_handler(items):
    """
    Create a list out of an item if it isn't a list or tuple already.

    Parameters:
        - items: The variable to make sure it is a list.

    Returns:
        The input as a list or tuple.
    """
    return (
        [items]
        if not isinstance(items, (list, tuple)) else
        items 
    )

def _sanitize_label(label):
    """
    Clean up a label by removing non-letter, non-space characters 
    and putting in all lowercase with underscores replacing spaces.

    Parameters:
        - label: The text you want to fix.

    Returns:
        The sanitized label.
    """
    return (
        re
        .sub(r'[^\w\s]', '', label)
        .lower()
        .replace(' ', '_')
    )


def sanitize_labels(method):
    """
    Decorator around a method that returns a dataframe to
    clean up all labels in said dataframe (column names and index name)
    by removing non-letter, non-space characters and
    putting in all lowercase with underscores replacing spaces.

    Parameters:
        - method: The method to wrap.

    Returns:
        A decorated method or function.
    """
    @wraps(method)
    def method_wrapper(self, *args, **kwargs):
        df = method(self, *args, **kwargs)
        renamed_index = _sanitize_label(df.index.name)
        return (
            df
            .reindex(index=df.index.rename(renamed_index))
            .rename(columns={
                column: _sanitize_label(column)
                for column in df.columns
            })
        )
    return method_wrapper


def validate_df(columns, instance_method=True):
    """
    Decorator that raises a `ValueError` if input isn't a pandas
    `DataFrame` or doesn't contain the proper columns. Note the `DataFrame`
    must be the first positional argument passed to this method.

    Parameters:
        - columns: A set of required column names.
                   For example, {'open', 'high', 'low', 'close'}.
        - instance_method: Whether or not the item being decorated is
                           an instance method. Pass `False` to decorate
                           static methods and functions.

    Returns:
        A decorated method or function.
    """
    def method_wrapper(method):
        @wraps(method)
        def validate_wrapper(self, *args, **kwargs):
            df = (self, *args)[0 if not instance_method else 1]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    'Must pass in a pandas `DataFrame`'
                )
            if columns.difference(df.columns):
                raise ValueError(
                    f'DataFrame must contain the following columns: {columns}'
                )
            return method(self, *args, **kwargs)
        return validate_wrapper
    return method_wrapper


# Functions related to Dataframe grouping


def group_stocks(mapping):
    """
    Create a new dataframe with many assets and a new column 
    indicating the asset that row's data belongs to.

    Parameters:
        - mapping: A key-value mapping of the form {asset_name: asset_df}

    Returns:
        A new `pandas.DataFrame` object
    """
    return pd.concat([
        asset_df.assign(name=asset_name)
        for asset_name, asset_df in mapping.items()
    ])


@validate_df(columns={'name'}, instance_method=False)
def describe_group(df):
    """
    Run `describe()` on the asset group created with `group_stocks()`.

    Parameters:
        - df: The group dataframe resulting from `group_stocks()`

    Returns:
        The transpose of the grouped description statistics.
    """
    return (
        df
        .groupby('name')
        .describe()
        .T
    )


@validate_df(columns=set(), instance_method=False)
def make_portfolio(df, date_level='date'):
    """
    Make a portfolio of assets by grouping by date and summing all columns.

    Note: the caller is responsible for making sure the dates line up across
    assets and handling when they don't.
    """
    return (
        df
        .groupby(level=date_level)
        .sum()
    )

def create_pivot_table(data, columns, column_values):
    """
    Create a new subset of a given DataFrame() by creating 
    a pivot table from it.

    Parameters:
        - data: The base DataFrame().
        - columns: The columns of the new DataFrame().
        - column_values: The values to put in each column.

    Returns:
        A DataFrame() object.
    """
    return data.pivot_table(
        index=data.index, 
        columns=columns,
        values=column_values, 
    )


def query_df(df, col_name, col_value):
    """
    Create a new subset of a given DataFrame() by querying it.

    Parameters:
        - df: The base DataFrame().
        - col_name: The name of the column.
        - col_vale: The values to match in that column.

    Returns:
        A DataFrame() object.
    """
    return df.query(f'{col_name} == "{col_value}"')


# Functions related to calculations over Dataframes


def calc_correlation(data1, data2):
    """
    Calculate the correlations between 2 DataFrames().

    Parameters:
        - data1: The first dataframe.
        - data2: The second dataframe.

    Returns:
        A Series() object.
    """
    return (
        data1.corrwith(data2).
        loc[lambda x: x.notnull()]
    )


def calc_diff(data, column1='open', column2='close'):
    """
    Calculate the difference between a column and the previous value
    of another column of a dataframe.

    Parameters:
        - data: The base DataFrame().
        - column1: The first column.
        - column2: The second column.

    Returns:
        A Series() object.
    """
    return data[column1] - data[column2].shift()


def calc_moving_average(data, func, named_arg, period):
    """
    Calculate the moving average of a Series() for a given period.

    Parameters:
        - data: The base Series()
        - func: The window calculation function.
        - named_arg: The name of the argument `periods` is being passed as.
        - period: The rule/span or list of them to pass to the
                    resampling/smoothing function, like '20D' 
                    for 20-day periods.

    Returns:
        A Series() object.
    """
    return (
        data
        .pipe(
            func=func, 
            **{named_arg: period}
        )
        .mean()
    )


# Functions related to Dataframe resampling


def resample_df(data, resample, agg_dict):
    """
    Resample a DataFrame() and run functions on columns specified in a dict.

    Parameters:
        - df: DataFrame() to be resampled.
        - resample: The period to use for resampling the data.
        - agg_dict: A dictionary that specifies the operations to be done
                    for each column after resampling.
    
    Returns:
        The resampled DataFrame()
    """
    return (
        data
        .resample(resample)
        .agg({
                col: agg_dict[col]
                for col in data.columns
                if col in agg_dict
        })
    )


def resample_series(data, period):
    """
    Resample a Series() by a specified period.
    
    Parameters:
        - df: DataFrame to be resampled.
        - data: The period to use for resampling the data.
    
    Returns:
        The resampled Series()
    """
    return (
        data
        .resample(period)
        .sum()
    )


def resample_index(index, period='Q'):
    """
    Resample an Index() by a specified period.

    Parameters:
        - index: The Index() to be resampled.
        - period: The period to use for resampling the data, if desired.
    
    Returns:
        The resampled Index()
    """
    return pd.date_range(
        start=index.min(),
        end=index.max(),
        freq=period,
    )
