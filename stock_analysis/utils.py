"""Utility functions for stock analysis."""
from functools import wraps
import re
import pandas as pd


# Functions related to Dataframe validation and cleaning

def _string_handler(item):
    """
    Static method for making a string out of an item if isn't it
    already.

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


def _iter_handler(items):
    """
    Static method for making a list out of an item if it isn't a list or
    tuple already.

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
    Clean up a label by removing non-letter, non-space characters and
    putting in all lowercase with underscores replacing spaces.

    Parameters:
        - label: The text you want to fix.

    Returns:
        The sanitized label.
    """
    return re\
        .sub(r'[^\w\s]', '', label)\
        .lower()\
        .replace(' ', '_')


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
        return df\
            .reindex(
                index=df.index.rename(renamed_index)
            )\
            .rename(
                columns=dict(
                    (column, _sanitize_label(column)) 
                    for column in df.columns
                ),
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
    Create a new dataframe with many assets and a new column indicating
    the asset that row's data belongs to.

    Parameters:
        - mapping: A key-value mapping of the form {asset_name: asset_df}

    Returns:
        A new `pandas.DataFrame` object
    """
    return pd.concat(
        list(
            asset_df.assign(name=asset_name)
            for asset_name, asset_df in mapping.items()
        )
    )


@validate_df(columns={'name'}, instance_method=False)
def describe_group(df):
    """
    Run `describe()` on the asset group created with `group_stocks()`.

    Parameters:
        - df: The group dataframe resulting from `group_stocks()`

    Returns:
        The transpose of the grouped description statistics.
    """
    return df\
        .groupby('name')\
        .describe()\
        .T


@validate_df(columns=set(), instance_method=False)
def make_portfolio(df, date_level='date'):
    """
    Make a portfolio of assets by grouping by date and summing all columns.

    Note: the caller is responsible for making sure the dates line up across
    assets and handling when they don't.
    """
    return df\
        .groupby(level=date_level)\
        .sum()

def create_pivot_table(data, columns, column_values):
    return data.pivot_table(
        index=data.index, 
        columns=columns,
        values=column_values, 
    )


def query_df(df, col_name, col_value):
    return df.query(f'{col_name} == "{col_value}"')


# Functions related to calculations over Dataframes

def calc_correlation(data1, data2):
    return data1.corrwith(data2).loc[lambda x: x.notnull()]


def calc_diff(data, column1='open', column2='close'):
    return data[column1] - data[column2].shift()


def calc_moving_average(data, func, named_arg, period):
    return data\
        .pipe(
            func=func, 
            **{named_arg: period}
        )\
        .mean()


# Functions related  Dataframe resampling


def resample_df(data, resample, agg_dict):
    """
    Resample a dataframe and run functions on columns specified in a dict.

    Parameters:
        - df: DataFrame to be resampled.
        - resample: The period to use for resampling the data, if desired.
        - agg_dict: A dictionary that specifies the operations to be done
                    for each column after resampling.
    Returns:
        The resampled dataframe
    """
    return data\
            .resample(resample)\
            .agg(
                dict(
                    (col, agg_dict[col])
                    for col in data.columns
                    if col in agg_dict
                )
            )


def resample_series(data, period):
    return data\
        .resample(period)\
        .sum()


def resample_index(index, period='Q'):
    return pd.date_range(
        start=index.min(),
        end=index.max(),
        freq=period,
    )
