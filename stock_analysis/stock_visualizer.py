"""Visualize financial instruments."""
import math
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from .utils import validate_df


def resample_df(df, resample, agg_dict):
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
    return df\
            .resample(resample)\
            .agg(
                dict(
                    (col, agg_dict[col])
                    for col in df.columns
                    if col in agg_dict
                )
            )


class Visualizer:
    """Base visualizer class not intended for direct use."""
    @validate_df(columns={'open', 'high', 'low', 'close'})
    def __init__(self, df):
        """Visualizer has a `pandas.DataFrame` object as an attribute."""
        self.df = df

    @staticmethod
    def plot_reference_line(ax, x=None, y=None, **kwargs):
        """
        Static method for adding reference lines to plots.

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
        if not x and not y:
            raise ValueError(
                'You must provide an `x` or a `y` at a minimum.'
            )
        elif x and not y:
            ax.axvline(x, **kwargs) # Vertical line
        elif not x and y:
            ax.axhline(y, **kwargs) # Horizontal line
        elif x.shape and y.shape:
            # In case numpy array-like structures are passed -> AB line
            ax.plot(x, y, **kwargs) 
        else:
            raise ValueError(
                'If providing only `x` or `y`, it must be a single value.'
            )
        return ax.legend()

    @staticmethod
    def plot_shaded_region(ax, x=tuple(), y=tuple(), **kwargs):
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

    @staticmethod
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
    
    def validate_periods(self, periods):
        return list(
            period if isinstance(period, str) else str(period)
            for period in self._iter_handler(periods)
        )


class StockVisualizer(Visualizer):
    """Visualizer for a single stock."""
    def plot_evolution_over_time(self, column, **kwargs):
        """
        Visualize the evolution over time of a column.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return self.df.plot(kind='line', y=column, **kwargs)

    def plot_boxplot(self, **kwargs):
        """
        Generate box plots for all columns.

        Parameters:
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return self.df.plot(kind='box', **kwargs)

    def plot_histogram(self, column, **kwargs):
        """
        Generate the histogram of a given column.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return self.df.plot(kind='hist', y=column, **kwargs)

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
        agg_dict = {
            'open': 'sum', 
            'close': 'sum',
            'high': 'sum', 
            'low': 'sum', 
            'volume': 'sum'
        }
        daily_effect = self.df.open - self.df.close.shift()
        monthly_effect = resample_df(
            df=daily_effect, 
            resample='1M', 
            agg_dict=agg_dict
        )
        monthly_effect.index = monthly_effect.index.strftime('%Y-%b') # Mutate in-place
        fig, axes = plt.subplots(1, 2, figsize=(15, 3))
        daily_effect\
            .plot(
                ax=axes[0],
                kind='line',
                title='After-hours trading - Daily effect'
            )
        monthly_effect\
            .plot(
                ax=axes[1],
                kind='bar',
                title='After-hours trading - Monthly effect',
                color=np.where(monthly_effect >= 0, 'g', 'r'),
                rot=90
            )\
            .axhline(
                0, 
                color='black', 
                linewidth=1
            )
        for ax in axes.flatten():
            ax.set_ylabel('price ($)')
        return axes

    @staticmethod
    def plot_area_between(y1, y2, title, label_higher, label_lower, figsize, legend_x):
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
            ('top', 'right'), # spines
        )
        for exclude_mask, color, label, spine in zipped:
            plt\
                .fill_between(
                    y2.index, 
                    y2, 
                    y1, 
                    figure=fig,
                    where=exclude_mask, 
                    color=color, 
                    label=label
                )
            fig\
                .axes[0]\
                .spines[spine]\
                .set_visible(False)
        plt.legend(
            bbox_to_anchor=(legend_x, -0.1), 
            framealpha=0, 
            ncol=2
        )
        plt.suptitle(title)
        return fig.axes[0]

    def plot_between_open_close(self, figsize=(10, 4)):
        """
        Visualize the daily change in price from open to close.

        Parameters:
            - figsize: A tuple of (width, height) for the plot dimensions.

        Returns:
            A matplotlib `Axes` object.
        """
        ax = self.plot_area_between(
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
        ax = self.plot_area_between(
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

    def _window_calc(self, column, periods, name, func, named_arg, **kwargs):
        """
        Helper method for plotting a series and adding reference lines using
        a window calculation.

        Parameters:
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
                       resampling/smoothing function, like '20D' 
                       for 20-day periods
                       (for resampling) or 20 for a 20-day span (smoothing)
            - name: The name of the window calculation (to show in the legend).
            - func: The window calculation function.
            - named_arg: The name of the argument `periods` is being passed as.
            - kwargs: Additional arguments to pass down to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        ax = self.df.plot(
            y=column, 
            **kwargs
        )
        for period in self.validate_periods(periods):
            self.df.loc[:,column]\
                .pipe(
                    func, 
                    **{named_arg: period}
                )\
                .mean()\
                .plot(
                    ax=ax,
                    linestyle='--',
                    label=f'{period + "D"} {name}'
                )
        plt.legend()
        return ax

    def plot_pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset.

        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn pairplot
        """
        return sns.pairplot(self.df, **kwargs)

    def plot_jointplot(self, other, column, **kwargs):
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
            x=self.df[column],
            y=other[column],
            **kwargs
        )

    def plot_correlation_heatmap(self, other):
        """
        Plot the correlations between this asset and
        another one with a heatmap.

        Parameters:
            - other: The other dataframe.

        Returns:
            A seaborn heatmap
        """
        correlations = self.df.pct_change()\
            .corrwith(other.pct_change())\
            .loc[lambda x: x.notnull()]
        size = len(correlations)
        correlation_matrix = np.zeros((size, size),  float)
        mask_matrix = np.ones_like(correlation_matrix) # Create mask to only show diagonal
        np.fill_diagonal(correlation_matrix, correlations)
        np.fill_diagonal(mask_matrix, 0)
        return sns.heatmap(
            data=correlation_matrix,
            annot=True,
            xticklabels=self.df.columns,
            yticklabels=self.df.columns,
            center=0,
            mask=mask_matrix,
            vmin=-1,
            vmax=1
        )

    def plot_moving_average(self, column, periods, **kwargs):
        """
        Add line(s) for the moving average of a column.

        Parameters:
            - column: The name of the column to plot.
            - periods: The rule or list of rules for resampling,
                       like '20D' for 20-day periods.
            - kwargs: Additional arguments to pass down 
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return self._window_calc(
            column, 
            periods, 
            name='MA',
            func=pd.DataFrame.resample, 
            named_arg='rule', 
            **kwargs
        )

    def plot_exp_smoothing(self, column, periods, **kwargs):
        """
        Add line(s) for the exponentially smoothed moving average of a column.

        Parameters:
            - column: The name of the column to plot.
            - periods: The span or list of spans for smoothing,
                       like 20 for 20-day periods.
            - kwargs: Additional arguments to pass down 
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        return self._window_calc(
            column, 
            periods, 
            name='EWMA',
            func=pd.DataFrame.ewm, 
            named_arg='span', 
            **kwargs
        )


class AssetGroupVisualizer(Visualizer):
    """Class for visualizing groups of assets in a single dataframe."""
    def __init__(self, df, group_by='name'):
        """This object also keeps track of which column it needs to group by."""
        super().__init__(df)
        self.group_by = group_by

    def plot_evolution_over_time(self, column, **kwargs):
        """
        Visualize the evolution over time of a column for all assets in group.

        Parameters:
            - column: The name of the column to visualize.
            - kwargs: Additional keyword arguments to pass down
                      to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        fig, ax = self._get_layout()
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

    def _get_layout(self):
        """
        Helper method for getting an autolayout of subplots (1 per group).

        Returns:
            The matplotlib `Figure` and `Axes` objects to plot with.
        """
        subplot_number = self.df\
            .loc[:,self.group_by]\
            .nunique()
        row_number = math.ceil(subplot_number / 2)
        fig, axes = plt.subplots(
            nrows=row_number, 
            ncols=2, 
            figsize=(15, 5 * row_number)
        )
        if row_number > 1:
            axes = axes.flatten()
        if subplot_number < len(axes):
            # Remove excess axes from autolayout
            for i in range(subplot_number, len(axes)):
                fig.delaxes(axes[i]) 
        return fig, axes

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
        fig, axes = self._get_layout()
        for ax, (name, data) in zip(
            axes, 
            self.df.groupby(self.group_by)
        ):
            sns.histplot(
                data[column], 
                kde=True, 
                ax=ax
            )
            ax.set_title(f'{name} - {column}')
        return axes

    def _window_calc(self, column, periods, name, func, named_arg, **kwargs):
        """
        Helper method for plotting a series and adding reference lines using
        a window calculation.

        Parameters:
            - column: The name of the column to plot.
            - periods: The rule/span or list of them to pass to the
                       resampling/smoothing function, like '20D' 
                       for 20-day periods
                       (for resampling) or 20 for a 20-day span (smoothing)
            - name: The name of the window calculation (to show in the legend).
            - func: The window calculation function.
            - named_arg: The name of the argument `periods` is being passed as.
            - kwargs: Additional arguments to pass down to the plotting function.

        Returns:
            A matplotlib `Axes` object.
        """
        fig, axes = self._get_layout()
        for ax, asset_name in zip(
            axes, 
            self.df[self.group_by].unique()
        ):
            subset = self.df\
                .query(f'{self.group_by} == "{asset_name}"')
            ax = subset.plot(
                y=column, 
                ax=ax, 
                label=asset_name, 
                **kwargs
            )
            for period in self._iter_handler(periods):
                validPeriod = \
                    period \
                    if isinstance(period, str) else \
                    str(period)
                subset[column]\
                    .pipe(func, **{named_arg: period})\
                    .mean()\
                    .plot(
                        ax=ax,
                        linestyle='--',
                        label=f'{validPeriod + "D"} {name}'
                    )
            ax.legend()
        plt.tight_layout()
        return ax

    def plot_after_hours_trades(self):
        """
        Visualize the effect of after-hours trading on this asset group.

        Returns:
            A matplotlib `Axes` object.
        """
        num_categories = self.df[self.group_by].nunique()
        fig, axes = plt.subplots(
            num_categories,
            2,
            figsize=(15, 3 * num_categories)
        )
        for ax, (name, data) in zip(
            axes, 
            self.df.groupby(self.group_by)
        ):
            after_hours = data.open - data.close.shift()
            monthly_effect = after_hours\
                .resample('1M')\
                .sum()
            after_hours\
                .plot(
                    ax=ax[0],
                    title=f'{name} Open Price - Prior Day\'s Close'
                )\
                .set_ylabel('price')
            monthly_effect.index = monthly_effect.index.strftime('%Y-%b')
            monthly_effect\
                .plot(
                    ax=ax[1],
                    kind='bar',
                    title=f'{name} after-hours trading monthly effect',
                    color=np.where(monthly_effect >= 0, 'g', 'r'),
                    rot=90
                )\
                .axhline(
                    0, 
                    color='black', 
                    linewidth=1
                )
            ax[1].set_ylabel('price')
        plt.tight_layout()
        return axes

    def create_pivot_table(self, column):
        return self.df.pivot_table(
            values=column, 
            index=self.df.index, 
            columns=self.group_by
        )

    def plot_pairplot(self, **kwargs):
        """
        Generate a seaborn pairplot for this asset group.

        Parameters:
            - kwargs: Keyword arguments to pass down to `sns.pairplot()`

        Returns:
            A seaborn pairplot
        """
        return sns.pairplot(
            data=self.create_pivot_table('close'),
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
        pivot_table = (
            self.create_pivot_table('close').pct_change()
            if pct_change else
            self.create_pivot_table('close')
        )
        return sns.heatmap(
            data=pivot_table.corr(), 
            annot=True, 
            center=0, 
            vmin=-1, 
            vmax=1, 
            **kwargs
        )
