"""Visualize financial instruments."""
import math
import functools
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from .utils import \
    validate_df, calc_correlation, calc_diff, calc_moving_average, \
    resample_df, resample_series, resample_index, \
    create_pivot_table, query_df

    
def set_ax_parameters(method):
    """
    Decorator around a method that returns a matplotlib `Axes` object.
    The method formats the x axes and sets the x and y label names. 
    It also sets the legend as visible.
    
    Parameters:
        - method: The method to wrap.

    Returns:
        A decorated method or function.
    """
    @functools.wraps(method)
    def method_wrapper(self, *args, **kwargs):
        df = {**kwargs}['data']
        new_index = resample_index(df.index)
        ax = method(self, *args, **kwargs)
        ax.set_xticks(new_index)
        ax.set_xticklabels(
            labels=new_index.strftime('%Y-%b'),
            rotation=30,
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        return ax
    return method_wrapper


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


def remove_excess_axes(fig, axes, subplot_number):
    for idx, ax in enumerate(axes.flatten()):
        if subplot_number <= idx < len(axes.flatten()):
            ax.set_visible(False)
    return fig, axes


class Visualizer:
    """Class with base visualization methods"""
    def __init__(self):
        pass

    def create_plot_layout(self, subplot_number=1, col_number=1):
        """
        Method that creates an autolayout of subplots.

        Parameters:
            - subplot_number: The number of subplots to create.
            - col_number: The number of columns of the layout.

        Returns:
            The matplotlib `Figure` and `Axes` objects to plot with.
        """
        row_number = math.ceil(subplot_number / col_number)
        fig, axes = plt.subplots(
            nrows=row_number, 
            ncols=col_number, 
            figsize=(15, 5*row_number)
        )
        if subplot_number == 1:
            return fig, axes
        else:
            return remove_excess_axes(fig, axes, subplot_number)

    def plot_shaded_region(self, ax, x=tuple(), y=tuple(), **kwargs):
        """
        Method that shades a region on a plot.

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

    def plot_reference_line(self, ax, x=None, y=None, **kwargs):
        """
        Method that adds a reference line to a plot.

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
            ax.plot(x, y, **kwargs) # AB line
        else:
            raise ValueError(
                'If providing only `x` or `y`, it must be a single value.'
            )
        return ax

    @set_ax_parameters
    def plot_curve(self, data, **kwargs):
        """
        Method that plots the evolution over time of a Series().

        Parameters:
            - data: The base Series()
            - kwargs: Additional keyword arguments to pass down
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        ax = data.plot(
            kind='line',
            **kwargs
        )
        return ax

    @set_ax_parameters
    def plot_bar(self, data, **kwargs):
        """
        Method that generates a bar plot for a Series().

        Parameters:
            - data: The base Series()
            - kwargs: Additional keyword arguments to pass down
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return data.plot(
            kind='bar',
            width=10,
            **kwargs
        )

    def plot_boxplot(self, data, **kwargs):
        """
        Method that generates a boxplot for a Series().

        Parameters:
            - data: The base Series()
            - kwargs: Additional keyword arguments to pass down
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return sns.boxplot(
            data=data,
            **kwargs
        )

    def plot_histogram(self, data, **kwargs):
        """
        Method that generates an histogram for a Series().

        Parameters:
            - data: The base Series()
            - kwargs: Additional keyword arguments to pass down
                        to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return sns.histplot(
            data=data,
            **kwargs
        )

    def plot_pairplot(self, data, **kwargs):
        """
        Method that generates a seaborn pairplot for a Series().

        Parameters:
            - data: The base Series()
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn pairplot
        """
        return sns.pairplot(
            data=data, 
            **kwargs
        )

    def plot_jointplot(self, data, data2, column, **kwargs):
        """
        Generate a seaborn jointplot for given column in a DataFrame() 
        compared to another DataFrame().

        Parameters:
            - data: The first DataFrame()
            - data2: The second DataFrame()
            - column: The column name to use for the comparison.
            - kwargs: Keyword arguments to pass down to `sns.jointplot()`

        Returns:
            A seaborn jointplot
        """
        return sns.jointplot(
            x=data.loc[:,column],
            y=data2.loc[:,column],
            **kwargs
        )

    def plot_heatmap(self, data, **kwargs):
        """
        Generate a seaborn heatmap for a DataFrame().

        Parameters:
            - data: The DataFrame() to use for the comparison.
            - kwargs: Keyword arguments to pass down to `sns.heatmap()`

        Returns:
            A seaborn heatmap
        """
        return sns.heatmap(
            data=data, 
            annot=True, 
            center=0, 
            vmin=-1, 
            vmax=1, 
            **kwargs
        )

    def plot_candlestick(self, data, volume=False, **kwargs):
        """
        Create a candlestick plot for a Series().

        Parameters:
            - data: The base Series()
            - volume: Whether to show a bar plot for volume traded 
                      under the candlesticks
            - kwargs: Additional keyword arguments to pass down 
                      to `mplfinance.plot()`

        Note: `mplfinance.plot()` doesn't return anything. 
              To save the plot, pass in `savefig=FILE_NAME.png`.
        """
        mpf.plot(
            data=data, 
            type='candle', 
            volume=volume, 
            **kwargs
        )
    
    def plot_moving_averages(self, data, ax, periods, func, named_arg, **kwargs):
        """
        Method for plotting moving averages of a Series()
        for different periods.

        Parameters:
            - data: The base Series()
            - ax: The matplotlib `Axes` object to add the curves to.
            - periods: The rule/span or list of them to pass to the
                        resampling/smoothing function, like '20D' 
                        for 20-day periods.
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

    @set_ax_parameters
    def plot_area_between(self, data, data2, title, labels, figure):
        """
        Method that fills the difference between 2 Series().

        Parameters:
            - data, data2: Data to be plotted with fill between data2 - data.
            - title: The title for the plot.
            - label_higher: String label for when data2 is higher than data.
            - label_lower: String label for when data2 is lower than data.
            - figsize: A tuple of (width, height) for the plot dimensions.

        Returns:
            A matplotlib `Axes` object.
        """
        is_higher = data2 - data > 0
        for exclude_mask, color, label in zip(
            (is_higher, np.invert(is_higher)), # filters
            ('g', 'r'), # colors
            labels, # labels
        ):
            plt.fill_between(
                x=data.index, 
                y1=data, 
                y2=data2, 
                figure=figure,
                where=exclude_mask, 
                color=color, 
                label=label,
            )
        plt.legend(
            bbox_to_anchor=(0.67, -0.1), 
            framealpha=0, 
            ncol=2
        )
        plt.suptitle(title)
        plt.tight_layout()
        return figure.axes[0]

    def plot_difference(self, data, period, axes, **kwargs):
        """
        Method that plots the difference between 2 columns
        of a DataFrame(), specifically a line plot for the daily difference
        and a bar plot for the monthly difference.

        Parameters:
            - data: the base DataFrame()
            - axes: The 2 matplotlib `Axes` objects to add 
                    the daily and monthly effects respectively.
            - period: The resampling period for the monthly effect.

        Returns:
            A matplotlib `Axes` object.
        """
        daily_effect = calc_diff(data)
        monthly_effect = resample_series(
            data=daily_effect, 
            period=period, 
        )
        ax = self.plot_curve(
            data=daily_effect,
            ax=axes[0],
            title='After-Hours Trading - Daily Effect',
            **kwargs
        )
        ax = self.plot_bar(
            data=monthly_effect,
            ax=axes[1],
            title='After-Hours Trading - Monthly Effect',
            color=np.where(monthly_effect >= 0, 'g', 'r'),
            **kwargs
        )
        for ax in axes:
            ax = self.plot_reference_line(
                ax=ax,
                y=0, 
                color='black', 
                linewidth=1
            )
        return ax


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
            - other: The other asset's DataFrame().

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
        return self.viz.plot_heatmap(
            data=corr_matrix,
            xticklabels=self.df.columns,
            yticklabels=self.df.columns,
            mask=mask_matrix,
        )

    def plot_candlestick(self, date_range=None, resample=None, volume=False, **kwargs):
        """
        Create a candlestick plot for an asset, with optional aggregation,
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
        """
        agg_dict = {
            'open': 'first', 
            'close': 'last',
            'high': 'max', 
            'low': 'min', 
            'volume': 'sum'
        }
        if not date_range:
            custom_range = slice(
                self.df.index.min(),
                self.df.index.max()
            )
        else:
            custom_range = date_range
        if resample:
            plot_data = resample_df(
                data=self.df.loc[custom_range],
                resample=resample, 
                agg_dict=agg_dict, 
            )
        else:
            plot_data = self.df.loc[custom_range]
        return self.viz.plot_candlestick(
            data=plot_data,
            volume=volume,
            **kwargs
        )

    def plot_between_open_close(self):
        """
        Visualize the daily change in price from open to close for an asset.

        Returns:
            A matplotlib `Axes` object.
        """
        fig, _ = self.viz.create_plot_layout()
        return self.viz.plot_area_between(
            data=self.df.open, 
            data2=self.df.close, 
            figure=fig,
            title='Daily price change (open to close)',
            labels=['Price rose', 'Price fell'],
        )

    def plot_between_closes(self, other_df):
        """
        Visualize the difference in closing prices between 2 assets.

        Parameters:
            - other_df: The other asset's DataFrame().

        Returns:
            A matplotlib `Axes` object.
        """
        fig, _ = self.viz.create_plot_layout()
        return self.viz.plot_area_between(
            data=other_df.close, 
            data2=self.df.close, 
            figure=fig, 
            title='Differential between asset closing price (this - other)',
            labels=['Asset is higher', 'Asset is lower']
        )

    def plot_after_hours_trades(self):
        """
        Visualize the effect of after-hours trading for an asset.

        Returns:
            A matplotlib `Axes` object.
        """
        _, ax_row = self.viz.create_plot_layout(col_number=2)
        return self.viz.plot_difference(
            data=self.df,
            axes=ax_row,
            period='1M',
            label='Asset'
        )

    def plot_moving_averages(self, column, periods, type_, **kwargs):
        """
        Add curve(s) for the moving average of a column for an asset.

        Parameters:
            - column: The name of the column to plot.
            - type_: The type of calculation to use,
                     like 'EWMA' for exponentially weighted moving average.
            - periods: The periods or list of periods to use,
                     like '20D' for 20-day periods.
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
        _, ax = self.viz.create_plot_layout()
        return self.viz.plot_moving_averages(
            data=self.df.loc[:,column], 
            ax=ax,
            periods=periods,
            func=func, 
            named_arg=named_arg, 
            **kwargs
        )


class AssetGroupVisualizer:
    """Class for visualizing groups of assets in a single dataframe."""
    @validate_df(columns={'open', 'high', 'low', 'close'})
    def __init__(self, df, group_by='name'):
        self.df = df
        self.group_by = group_by
        self.viz = Visualizer()

    @property
    def asset_names(self):
        return self.df.loc[:,self.group_by].unique()

    @property
    def asset_number(self):
        return len(self.asset_names)

    def group_df(self, col_value):
        return query_df(
            df=self.df,
            col_name=self.group_by,
            col_value=col_value,
        )

    def plot_curve(self, column, **kwargs):
        """
        Visualize the evolution over time of a column for all assets.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            _, ax = self.viz.create_plot_layout()
        for asset_name in self.asset_names:
            grouped_df = self.group_df(col_value=asset_name)
            ax = self.viz.plot_curve(
                data=grouped_df.loc[:,column],
                ax=ax,
                label=asset_name,
            )
        return ax

    def plot_boxplot(self, column, **kwargs):
        """
        Generate box plots for a given column for all assets.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return self.viz.plot_boxplot(
            data=self.df,
            x=self.group_by,
            y=column,
            **kwargs
        )

    def plot_histogram(self, column, **kwargs):
        """
        Generate the histogram of a given column for all assets.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        _, ax_layout = self.viz.create_plot_layout(
            subplot_number=self.asset_number,
            col_number=2,
        )
        for ax, asset_name in zip(ax_layout.flatten(), self.asset_names):
            grouped_df = self.group_df(col_value=asset_name)
            ax = self.viz.plot_histogram(
                data=grouped_df,
                x=grouped_df.loc[:,column], 
                ax=ax,
                kde=True, 
                **kwargs
            )
            ax.set_title(f'{asset_name}')
        return ax_layout

    def plot_pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for all assets.

        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn pairplot
        """
        pivot_table = create_pivot_table(
            data=self.df,
            columns=self.group_by,
            column_values='close',
        )
        return self.viz.plot_pairplot(
            data=pivot_table,
            diag_kind='kde',
            **kwargs
        )

    def plot_correlation_heatmap(self, pct_change=True, **kwargs):
        """
        Generate a seaborn heatmap for correlations between all assets.

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
        if pct_change:
            pivot_table = _pivot_table.pct_change().corr()
        else:
            pivot_table = _pivot_table.corr()
        return self.viz.plot_heatmap(
            data=pivot_table,
            **kwargs
        )

    def plot_after_hours_trades(self):
        """
        Visualize the effect of after-hours trading for all assets.

        Returns:
            A matplotlib `Axes` object.
        """
        _, ax_layout = self.viz.create_plot_layout(
            subplot_number=2*self.asset_number,
            col_number=2,
        )
        for ax_row, asset_name in zip(ax_layout, self.asset_names):
            grouped_df = self.group_df(col_value=asset_name)
            ax = self.viz.plot_difference(
                data=grouped_df,
                axes=ax_row,
                period='1M',
                label=asset_name,
            )
        plt.tight_layout()
        return ax

    def plot_moving_averages(self, column, periods, type_, **kwargs):
        """
        Add curve(s) for the moving average of a column for all assets.

        Parameters:
            - column: The name of the column to plot.
            - type_: The type of calculation to use,
                     like 'EWMA' for exponentially weighted moving average.
            - periods: The periods or list of periods to use,
                     like '20D' for 20-day periods.
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
        _, ax_layout = self.viz.create_plot_layout(
            subplot_number=self.asset_number,
            col_number=2,
        )
        for ax, asset_name in zip(ax_layout.flatten(), self.asset_names):
            grouped_df = self.group_df(col_value=asset_name)
            ax = self.viz.plot_moving_averages(
                data=grouped_df.loc[:,column], 
                ax=ax,
                periods=periods,
                func=func, 
                named_arg=named_arg, 
                label=asset_name,
                **kwargs
            )
        plt.tight_layout()
        return ax
