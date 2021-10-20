"""Classes for technical analysis of assets."""
import math
from .utils import validate_df


class StockAnalyzer:
    """Class for providing metrics for technical analysis of a stock."""
    @validate_df(columns={'open', 'high', 'low', 'close'})
    def __init__(self, df):
        """Create a `StockAnalyzer` object by passing in 
        a `pandas.DataFrame` of OHLC data."""
        self.df = df

    @property
    def _max_periods(self):
        """Get the maximum number of trading periods that can be used 
        in calculations."""
        return self.df.shape[0]

    @property
    def close(self):
        """Get the close column of the data."""
        return self.df.close

    @property
    def pct_change(self):
        """Get the percent change of the close column."""
        return self.close.pct_change()

    @property
    def pivot_point(self):
        """Calculate the pivot point for support/resistance calculations."""
        return (self.last_close + self.last_high + self.last_low) / 3

    @property
    def last_close(self):
        """Get the value of the last close in the data."""
        return self.df\
            .last('1D')\
            .close\
            .iat[0]

    @property
    def last_high(self):
        """Get the value of the last high in the data."""
        return self.df\
            .last('1D')\
            .high\
            .iat[0]

    @property
    def last_low(self):
        """Get the value of the last low in the data."""
        return self.df\
            .last('1D')\
            .low\
            .iat[0]

    def calc_resistance(self, level=1):
        """
        Calculate the resistance at the given level.

        Parameters:
            - level: The resistance level (1, 2, or 3)

        Returns:
            The resistance value.
        """
        if level == 1:
            resistance = (2*self.pivot_point) - self.last_low
        elif level == 2:
            resistance = self.pivot_point + (self.last_high - self.last_low)
        elif level == 3:
            resistance = self.last_high + 2*(self.pivot_point - self.last_low)
        else:
            raise ValueError('Not a valid level. Must be 1, 2, or 3')
        return resistance
    
    def calc_support(self, level=1):
        """
        Calculate the support at the given level.

        Parameters:
            - level: The support level (1, 2, or 3)

        Returns:
            The support value.
        """
        if level == 1:
            support = (2*self.pivot_point) - self.last_high
        elif level == 2:
            support = self.pivot_point - (self.last_high - self.last_low)
        elif level == 3:
            support = self.last_low - 2*(self.last_high - self.pivot_point)
        else:
            raise ValueError('Not a valid level. Must be 1, 2, or 3')
        return support

    def calc_daily_std(self, periods=252):
        """
        Calculate the daily standard deviation of percent change.

        Parameters:
            - periods: The number of periods to use for the calculation;
                       default is 252 for the trading days in a year.
                       Note if you provide a number greater than the number
                       of trading periods in the data, `self._max_periods`
                       will be used instead.

        Returns:
            The standard deviation
        """
        valid_periods = min(periods, self._max_periods)*(-1)
        return self.pct_change[valid_periods:].std()

    def calc_annualized_volatility(self):
        """Calculate the annualized volatility."""
        return self.calc_daily_std()*math.sqrt(252)

    def calc_rolling_volatility(self, periods=252):
        """
        Calculate the rolling volatility.

        Parameters:
            - periods: The number of periods to use for the calculation;
                       default is 252 for the trading days in a year.
                       Note if you provide a number greater than the number
                       of trading periods in the data, `self._max_periods`
                       will be used instead.

        Returns:
            A `pandas.Series` object.
        """
        valid_periods = min(periods, self._max_periods)
        volatility = self.close\
            .rolling(valid_periods)\
            .std()
        return volatility / math.sqrt(valid_periods)

    def calc_correlation_with(self, other):
        """
        Calculate the correlations between this dataframe and another
        for matching columns.

        Parameters:
            - other: The other dataframe.

        Returns:
            A `pandas.Series` object.
        """
        this_pct_change = self.df.pct_change()
        other_pct_change = other.pct_change()
        return this_pct_change.corrwith(other_pct_change)

    def calc_cv(self):
        """
        Calculate the coefficient of variation for the asset. Note
        that the lower this is, the better the risk/return tradeoff.
        """
        return self.close.std() / self.close.mean()

    def calc_qcd(self):
        """Calculate the quantile coefficient of dispersion."""
        q1, q3 = self.close.quantile([0.25, 0.75])
        return (q3 - q1) / (q3 + q1)

    def calc_beta(self, index):
        """
        Calculate the beta of the asset.

        Parameters:
            - index: The dataframe for the index to compare to.

        Returns:
            Beta, a float.
        """
        asset_pct_change = self.pct_change
        index_pct_change = index.close.pct_change()
        index_variance = index_pct_change.var()
        return asset_pct_change.cov(index_pct_change) / index_variance

    def calc_cumulative_returns(self):
        """Calculate the series of cumulative returns for plotting."""
        return (1 + self.pct_change).cumprod()

    @staticmethod
    def calc_portfolio_return(df):
        """
        Calculate the return assuming no distribution per share.

        Parameters:
            - df: The asset's dataframe.

        Returns:
            The return, as a float.
        """
        start_price =  df.close[0]
        end_price = df.close[-1]
        return (end_price - start_price) / start_price
    
    def calc_alpha(self, index, risk_free_ror):
        """
        Calculates the asset's alpha.

        Parameters:
            - index: The index to compare to.
            - risk_free_ror: The risk-free rate of return.
                   Consult 
                   https://www.treasury.gov/resource-center/data-chart-center/
                   interest-rates/pages/TextView.aspx?data=yield
                   for US Treasury Bill historical rates.

        Returns:
            Alpha, as a float.
        """
        rf_ror = risk_free_ror / 100
        asset_ror = self.calc_portfolio_return(self.df)
        index_ror = self.calc_portfolio_return(index)
        beta = self.calc_beta(index)
        return asset_ror - rf_ror - beta*(index_ror - rf_ror)

    def is_bear_market(self):
        """
        Determine if a stock is in a bear market, meaning its
        return in the last 2 months is a decline of 20% or more.
        """
        period = self.df.last('2M')
        return self.calc_portfolio_return(period) <= -0.2

    def is_bull_market(self):
        """
        Determine if a stock is in a bull market, meaning its
        return in the last 2 months is an increase of 20% or more.
        """
        period = self.df.last('2M')
        return self.calc_portfolio_return(period) >= 0.2

    def calc_sharpe_ratio(self, risk_free_ror):
        """
        Calculates the asset's Sharpe ratio.

        Parameters:
            - risk_free_ror: The risk-free rate of return
                   Consult 
                   https://www.treasury.gov/resource-center/data-chart-center/
                   interest-rates/pages/TextView.aspx?data=yield
                   for US Treasury Bill historical rates.

        Returns:
            The Sharpe ratio, as a float.
        """
        asset_ror = self\
            .calc_cumulative_returns()\
            .last('1D')\
            .iat[0]
        stdev_asset_ror = self.calc_cumulative_returns().std()
        return (asset_ror - risk_free_ror) / stdev_asset_ror


class AssetGroupAnalyzer:
    """Analyzes many assets in a dataframe."""
    @validate_df(columns={'open', 'high', 'low', 'close'})
    def __init__(self, df, groupby_criteria='name'):
        """
        Create an `AssetGroupAnalyzer` object by passing in 
        a `pandas.DataFrame` and column to group by.
        """
        self.df = df
        if groupby_criteria not in self.df.columns:
            raise ValueError(
                f'Column "{groupby_criteria}" not in dataframe.'
            )
        self.groupby_criteria = groupby_criteria
        self.analyzers = self._composition_handler()

    def _composition_handler(self):
        """
        Create a dictionary mapping each group to its analyzer,
        taking advantage of composition instead of inheritance.
        """
        return {
            group: StockAnalyzer(data)
            for group, data in self.df.groupby(self.groupby_criteria)
        }

    def analyze(self, function_name, **kwargs):
        """
        Run a `StockAnalyzer` method on all assets in the group.

        Parameters:
            - function_name: The name of the method to run.
            - kwargs: Additional keyword arguments to pass to the function.

        Returns:
            A dictionary mapping each asset to the result of the
            calculation of that function.
        """
        if not hasattr(StockAnalyzer, function_name):
            raise ValueError(
                f'StockAnalyzer has no "{function_name}" method.'
            )
        if not kwargs:
            kwargs = dict()
        return {
            group: getattr(analyzer, function_name)(**kwargs)
            for group, analyzer in self.analyzers.items()
        }
