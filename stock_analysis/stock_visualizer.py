"""Visualize financial instruments."""
import math
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from .utils import validate_df


def create_pivot_table(data, columns, column_values):
    return data.pivot_table(
        index=data.index, 
        columns=columns,
        values=column_values, 
    )


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


def validate_period(named_arg, period):
    return (
        period
        if named_arg == 'rule' else
        int(period.strip('D'))
    )


def validate_name(named_arg):
    return (
        'MA' 
        if named_arg == 'rule' else 
        'EWMA'
    )


class Visualizer:
    """Class with utility methods"""
    def __init__(self):
        pass

    def plot_reference_line(self, ax, x=None, y=None, **kwargs):
        """
        Method for adding reference lines to plots.

        Parameters:
            - ax: The matplotlib `Axes` object to add the reference line to.
            - x, y: The x, y value to draw the line at as a
                    single value or numpy array-like structure.
                        - For horizontal: pass only `y`
                        - For vertical: pass only `x`
                        - For AB line: pass both `x` and `y`
                        for all coordinates on the line
            - kwargs: Additional keyword arguments to pass to the plotting
                      function.

        Returns:
            The matplotlib `Axes` object passed in.
        """
        if (x is None) and (y is None):
            raise ValueError(
                'You must provide an `x` or a `y` at a minimum.'
            )
        if (x is not None) and (y is None):
            ax.axvline(x, **kwargs) # Vertical line
        elif (x is None) and (y is not None):
            ax.axhline(y, **kwargs) # Horizontal line
        elif x.shape and y.shape:
            # In case numpy array-like structures are passed -> AB line
            ax.plot(x, y, **kwargs) 
        else:
            raise ValueError(
                'If providing only `x` or `y`, it must be a single value.'
            )
        ax.legend()
        return ax

    def plot_shaded_region(self, ax, x=tuple(), y=tuple(), **kwargs):
        """
        Static method for shading a region on a plot.

        Parameters:
            - ax: The matplotlib `Axes` object to add the shaded region to.
            - x: Tuple with the `xmin` and `xmax` bounds for the rectangle
                 drawn vertically.
            - y: Tuple with the `ymin` and `ymax` bounds for the rectangle
                 drawn horizontally.
            - kwargs: Additional keyword arguments to pass to the plotting
                      function.

        Returns:
            The matplotlib `Axes` object passed in.
        """
        if not x and not y:
            raise ValueError(
                'You must provide an x or a y min/max tuple at a minimum.'
            )
        elif x and y:
            raise ValueError('You can only provide `x` or `y`.')
        elif x and not y:
            ax.axvspan(*x, **kwargs) # Vertical span
        elif not x and y:
            ax.axhspan(*y, **kwargs) # Horizontal span
        return ax

    def plot_curve(self, data, **kwargs):
        """
        Visualize the evolution over time of a column.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        ax = sns.lineplot(
            data=data,
            **kwargs
        )
        ax.set_xticklabels(
            labels=data.index.strftime('%Y-%b'),
            rotation=45,
        )
        return ax

    def plot_moving_averages(self, data, ax, periods, func, named_arg, **kwargs):
        """
        Helper method for plotting moving averages for different periods.

        Parameters:
            - ax: The matplotlib `Axes` object to add the curves to.
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
                        resampling/smoothing function, like '20D' 
                        for 20-day periods
                        (for resampling) or 20 for a 20-day span (smoothing)
            - func: The window calculation function.
            - named_arg: The name of the argument `periods` is being passed as.
            - kwargs: Additional arguments to pass down to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        ax = self.plot_curve(
            data=data,
            ax=ax,
            **kwargs
        )
        for period in periods:
            moving_avg = calc_moving_average(
                data=data, 
                func=func, 
                named_arg=named_arg, 
                period=validate_period(named_arg, period),
            )
            ax = self.plot_curve(
                data=moving_avg,
                ax=ax,
                linestyle='--',
                label=f'{period} {validate_name(named_arg)}',
            )
        return ax

    def plot_boxplot(self, data, column, **kwargs):
        """
        Generate box plots for all columns.

        Parameters:
            - kwargs: Additional keyword arguments to pass down
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return sns.boxplot(
            data=data,
            y=data.loc[:,column],
            **kwargs
        )

    def plot_histogram(self, data, column, **kwargs):
        """
        Generate the histogram of a given column.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return sns.histplot(
            data=data,
            x=data.loc[:,column], 
            **kwargs
        )

    def plot_pairplot(self, data, **kwargs):
        """
        Generate a seaborn pairplot for this asset.

        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn pairplot
        """
        return sns.pairplot(
            data=data, 
            **kwargs
        )

    def plot_jointplot(self, data1, data2, column, **kwargs):
        """
        Generate a seaborn jointplot for given column in asset compared to
        another asset.

        Parameters:
            - other: The other asset's dataframe
            - column: The column name to use for the comparison.
            - kwargs: Keyword arguments to pass down to `sns.jointplot()`

        Returns:
            A seaborn jointplot
        """
        return sns.jointplot(
            x=data1.loc[:,column],
            y=data2.loc[:,column],
            **kwargs
        )

    def plot_area_between(self, y1, y2, title, label_higher, label_lower, figsize, legend_x):
        """
        Visualize the difference between assets.

        Parameters:
            - y1, y2: Data to be plotted with fill between y2 - y1.
            - title: The title for the plot.
            - label_higher: String label for when y2 is higher than y1.
            - label_lower: String label for when y2 is lower than y1.
            - figsize: A tuple of (width, height) for the plot dimensions.
            - legend_x: Where to place the legend below the plot.

        Returns:
            A matplotlib `Axes` object.
        """
        fig = plt.figure(figsize=figsize)
        is_higher = y2 - y1 > 0
        zipped = zip(
            (is_higher, np.invert(is_higher)), # filters
            ('g', 'r'), # colors
            (label_higher, label_lower), # labels
        )
        for exclude_mask, color, label in zipped:
            plt.fill_between(
                x=y2.index, 
                y1=y1, 
                y2=y2, 
                figure=fig,
                where=exclude_mask, 
                color=color, 
                label=label
            )
        plt.legend(
            bbox_to_anchor=(legend_x, -0.1), 
            framealpha=0, 
            ncol=2
        )
        plt.suptitle(title)
        return fig.axes[0]

    def plot_difference(self, data, axes, period, name):
        daily_effect = calc_diff(data)
        monthly_effect = resample_series(
            data=daily_effect, 
            period=period, 
        )
        ax = self.plot_curve(
            data=daily_effect,
            ax=axes[0],
        )
        ax = axes[1].bar(
            x=monthly_effect.index,
            height=monthly_effect,
            width=10,
            color=np.where(monthly_effect >= 0, 'g', 'r'),
        )
        for ax in axes:
            ax = self.plot_reference_line(
                ax=ax,
                y=0, 
                color='black', 
                linewidth=1
            )
        return ax

    def create_plot_layout(self, subplot_number=1, col_number=1):
        """
        Helper method for getting an autolayout of subplots (1 per group).

        Returns:
            The matplotlib `Figure` and `Axes` objects to plot with.
        """
        def remove_excess_axes(fig, axes):
            for idx, ax in enumerate(axes.flatten()):
                if subplot_number <= idx < len(axes.flatten()):
                    ax.set_visible(False)
            return fig, axes

        row_number = math.ceil(subplot_number / col_number)
        fig, axes = plt.subplots(
            nrows=row_number, 
            ncols=col_number, 
            figsize=(15, 5*row_number)
        )
        if subplot_number == 1:
            return fig, axes
        else:
            return remove_excess_axes(fig, axes)


class StockVisualizer:
    """Class for visualizing a single asset."""
    @validate_df(columns={'open', 'high', 'low', 'close'})
    def __init__(self, df):
        self.df = df
        self.viz = Visualizer()

    def plot_correlation_heatmap(self, other):
        """
        Plot the correlations between this asset and
        another one with a heatmap.

        Parameters:
            - other: The other dataframe.

        Returns:
            A seaborn heatmap
        """
        correlations = calc_correlation(
            data1=self.df.pct_change(),
            data2=other.pct_change(),
        )
        size = len(correlations)
        corr_matrix = np.zeros((size, size),  float)
        mask_matrix = np.ones_like(corr_matrix) # Create mask to only show diagonal
        np.fill_diagonal(corr_matrix, correlations)
        np.fill_diagonal(mask_matrix, 0)
        return sns.heatmap(
            data=corr_matrix,
            annot=True,
            xticklabels=self.df.columns,
            yticklabels=self.df.columns,
            center=0,
            mask=mask_matrix,
            vmin=-1,
            vmax=1
        )

    def plot_candlestick(self, date_range=None, resample=None, volume=False, **kwargs):
        """
        Create a candlestick plot for the OHLC data with optional aggregation,
        subset of the date range, and volume.

        Parameters:
            - date_range: String or `slice()` of dates to pass to `loc[]`, 
                          if `None`
                          the plot will be for the full range of the data.
            - resample: The offset to use for resampling the data, if desired.
            - volume: Whether to show a bar plot for volume traded 
                      under the candlesticks
            - kwargs: Additional keyword arguments to pass down 
                      to `mplfinance.plot()`

        Note: `mplfinance.plot()` doesn't return anything. 
              To save your plot, pass in `savefig=file.png`.
        """
        def _plot_candlestick(data, volume=False, **kwargs):
            mpf.plot(
                data=data, 
                type='candle', 
                volume=volume, 
                **kwargs
            )
        agg_dict = {
            'open': 'first', 
            'close': 'last',
            'high': 'max', 
            'low': 'min', 
            'volume': 'sum'
        }
        custom_range = slice(
            self.df.index.min(),
            self.df.index.max()
        )
        plot_data = self.df.loc[
            custom_range 
            if not date_range else 
            date_range
        ]
        return _plot_candlestick(
            data=(
                resample_df(plot_data, resample, agg_dict)
                if resample else
                plot_data
            ),
            volume=volume,
            **kwargs
        )

    def plot_after_hours_trades(self):
        """
        Visualize the effect of after-hours trading on this asset.

        Returns:
            A matplotlib `Axes` object.
        """
        _, ax_row = plt.subplots(1, 2, figsize=(15, 3))
        return self.viz.plot_difference(
            data=self.df,
            axes=ax_row,
            period='1M',
            name='Asset'
        )

    def plot_between_open_close(self, figsize=(10, 4)):
        """
        Visualize the daily change in price from open to close.

        Parameters:
            - figsize: A tuple of (width, height) for the plot dimensions.

        Returns:
            A matplotlib `Axes` object.
        """
        ax = self.viz.plot_area_between(
            y1=self.df.open, 
            y2=self.df.close, 
            figsize=figsize,
            legend_x=0.67, 
            title='Daily price change (open to close)',
            label_higher='Price rose', 
            label_lower='Price fell'
        )
        ax.set_ylabel('price')
        return ax

    def plot_between_closes(self, other_df, figsize=(10, 4)):
        """
        Visualize the difference in closing price between assets.

        Parameters:
            - other_df: The dataframe with the other asset's data.
            - figsize: A tuple of (width, height) for the plot dimensions.

        Returns:
            A matplotlib `Axes` object.
        """
        ax = self.viz.plot_area_between(
            y1=other_df.close, 
            y2=self.df.close, 
            figsize=figsize, 
            legend_x=0.7,
            title='Differential between asset closing price (this - other)',
            label_higher='Asset is higher', 
            label_lower='Asset is lower'
        )
        ax.set_ylabel('price')
        return ax

    def plot_moving_averages(self, column, periods, type_, **kwargs):
        """
        Add curve(s) for the exponentially smoothed moving average of a column.

        Parameters:
            - ax: The matplotlib `Axes` object to add the curves to.
            - column: The name of the column to plot.
            - periods: The span or list of spans for smoothing,
                        like 20 for 20-day periods.
            - kwargs: Additional arguments to pass down 
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        if type_ == 'MA':
            func=pd.DataFrame.resample
            named_arg='rule'
        if type_ == 'EWMA':
            func=pd.DataFrame.ewm
            named_arg='span'
        
        _, ax = plt.subplots(1, 1)
        ax = self.viz.plot_moving_averages(
            data=self.df.loc[:,column], 
            ax=ax,
            periods=periods,
            func=func, 
            named_arg=named_arg, 
            **kwargs
        )
        return ax


class AssetGroupVisualizer:
    """Class for visualizing groups of assets in a single dataframe."""
    @validate_df(columns={'open', 'high', 'low', 'close'})
    def __init__(self, df, group_by='name'):
        self.df = df
        self.group_by = group_by
        self.viz = Visualizer()

    @property
    def grouped_df(self):
        return self.df.groupby(self.group_by)

    @property
    def asset_names(self):
        return self.df.loc[:,self.group_by].unique()

    def plot_curve(self, column, **kwargs):
        """
        Visualize the evolution over time of a column for all assets in group.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        fig, ax = self.viz.create_plot_layout()
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        return sns.lineplot(
            x=self.df.index,
            y=column,
            hue=self.group_by,
            data=self.df,
            ax=ax,
            **kwargs
        )

    def plot_boxplot(self, column, **kwargs):
        """
        Generate box plots for a given column in all assets.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return sns.boxplot(
            x=self.group_by,
            y=column,
            data=self.df,
            **kwargs
        )

    def plot_histogram(self, column, **kwargs):
        """
        Generate the histogram of a given column for all assets in group.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        _, ax_layout = self.viz.create_plot_layout(
            subplot_number=len(self.asset_names),
            col_number=2,
        )
        for ax, (name, data) in zip(
            ax_layout.flatten(), 
            self.grouped_df
        ):
            ax = self.viz.plot_histogram(
                data=data,
                column=column,
                ax=ax,
                kde=True, 
            )
            ax.set_title(f'{name} - {column}')
        return ax_layout

    def plot_moving_averages(self, column, periods, type_, **kwargs):
        """
        Add curve(s) for the exponentially smoothed moving average of a column.

        Parameters:
            - ax: The matplotlib `Axes` object to add the curves to.
            - column: The name of the column to plot.
            - periods: The span or list of spans for smoothing,
                        like 20 for 20-day periods.
            - kwargs: Additional arguments to pass down 
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        def _plot_moving_averages(column, periods, func, named_arg, **kwargs):
            _, ax_layout = self.viz.create_plot_layout(
                subplot_number=len(self.asset_names),
                col_number=2,
            )
            for ax, asset_name in zip(
                ax_layout.flatten(), 
                self.asset_names
            ):
                subset = self.df\
                    .query(f'{self.group_by} == "{asset_name}"')\
                    .loc[:,column]
                ax = self.viz.plot_moving_averages(
                    data=subset, 
                    ax=ax,
                    periods=periods,
                    func=func, 
                    named_arg=named_arg, 
                    label=asset_name,
                )
            plt.tight_layout()
            return ax
        
        if type_ == 'MA':
            func=pd.DataFrame.resample
            named_arg='rule'
        if type_ == 'EWMA':
            func=pd.DataFrame.ewm
            named_arg='span'
        return _plot_moving_averages(
            column=column,
            periods=periods, 
            func=func, 
            named_arg=named_arg, 
            **kwargs,
        )

    def plot_after_hours_trades(self):
        """
        Visualize the effect of after-hours trading on this asset group.

        Returns:
            A matplotlib `Axes` object.
        """
        _, ax_layout = self.viz.create_plot_layout(
            subplot_number=2*len(self.asset_names),
            col_number=2,
        )
        for ax_row, (name, data) in zip(
            ax_layout, 
            self.grouped_df
        ):
            ax = self.viz.plot_difference(
                data=data,
                axes=ax_row,
                period='1M',
                name=name,
            )
        plt.tight_layout()
        return ax

    def plot_pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset group.

        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn pairplot
        """
        return self.viz.plot_pairplot(
            data=create_pivot_table(
                data=self.df,
                columns=self.group_by,
                column_values='close',
            ),
            diag_kind='kde',
            **kwargs
        )

    def plot_heatmap(self, pct_change=True, **kwargs):
        """
        Generate a seaborn heatmap for correlations between assets.

        Parameters:
            - pct_change: Whether or not to show the correlations of the
                          daily percent change in price or just use
                          the closing price.
            - kwargs: Keyword arguments to pass down to `sns.heatmap()`

        Returns:
            A seaborn heatmap
        """
        _pivot_table = create_pivot_table(
            data=self.df,
            columns=self.group_by,
            column_values='close',
        )
        pivot_table = (
            _pivot_table.pct_change()
            if pct_change else
            _pivot_table
        )
        return sns.heatmap(
            data=pivot_table.corr(), 
            annot=True, 
            center=0, 
            vmin=-1, 
            vmax=1, 
            **kwargs
        )
