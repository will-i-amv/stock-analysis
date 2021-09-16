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
            res = (2 * self.pivot_point) - self.last_low
        elif level == 2:
            res = self.pivot_point + (self.last_high - self.last_low)
        elif level == 3:
            res = self.last_high + 2 * (self.pivot_point - self.last_low)
        else:
            raise ValueError('Not a valid level. Must be 1, 2, or 3')
        return res

    def calc_support(self, level=1):
        """
        Calculate the support at the given level.

        Parameters:
            - level: The support level (1, 2, or 3)

        Returns:
            The support value.
        """
        if level == 1:
            sup = (2 * self.pivot_point) - self.last_high
        elif level == 2:
            sup = self.pivot_point - (self.last_high - self.last_low)
        elif level == 3:
            sup = self.last_low - 2 * (self.last_high - self.pivot_point)
        else:
            raise ValueError('Not a valid level. Must be 1, 2, or 3')
        return sup

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
        validPeriods = min(periods, self._max_periods) * -1
        return self.pct_change[validPeriods:].std()

    def annualized_volatility(self):
        """Calculate the annualized volatility."""
        return self.calc_daily_std() * math.sqrt(252)

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
        validPeriods = min(periods, self._max_periods)
        numerator = self.close\
            .rolling(validPeriods)\
            .std()
        denominator = math.sqrt(validPeriods)
        return numerator / denominator

    def calc_correlation_with(self, other):
        """
        Calculate the correlations between this dataframe and another
        for matching columns.

        Parameters:
            - other: The other dataframe.

        Returns:
            A `pandas.Series` object.
        """
        return self.df.pct_change()\
            .corrwith(
                other.pct_change()
            )

    def cv(self):
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
        numerator = self.pct_change\
            .cov(
                index.close.pct_change()
            )
        denominator = index.close.pct_change().var()
        return numerator / denominator

    def cumulative_returns(self):
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
        start =  df.close[0]
        end = df.close[-1]
        return (end - start) / start

    def calc_alpha(self, index, r_f):
        """
        Calculates the asset's alpha.

        Parameters:
            - index: The index to compare to.
            - r_f: The risk-free rate of return.
                   Consult 
                   https://www.treasury.gov/resource-center/data-chart-center/
                   interest-rates/pages/TextView.aspx?data=yield
                   for US Treasury Bill historical rates.

        Returns:
            Alpha, as a float.
        """
        r = self.calc_portfolio_return(self.df)
        r_f /= 100
        r_m = self.calc_portfolio_return(index)
        beta = self.calc_beta(index)
        return r - r_f - beta * (r_m - r_f)

    def is_bear_market(self):
        """
        Determine if a stock is in a bear market, meaning its
        return in the last 2 months is a decline of 20% or more.
        """
        validPeriod = self.df.last('2M')
        return self.calc_portfolio_return(validPeriod) <= -0.2

    def is_bull_market(self):
        """
        Determine if a stock is in a bull market, meaning its
        return in the last 2 months is an increase of 20% or more.
        """
        validPeriod = self.df.last('2M')
        return self.calc_portfolio_return(validPeriod) >= 0.2

    def calc_sharpe_ratio(self, r_f):
        """
        Calculates the asset's Sharpe ratio.

        Parameters:
            - r_f: The risk-free rate of return
                   Consult 
                   https://www.treasury.gov/resource-center/data-chart-center/
                   interest-rates/pages/TextView.aspx?data=yield
                   for US Treasury Bill historical rates.

        Returns:
            The Sharpe ratio, as a float.
        """
        numerator = self.cumulative_returns().last('1D').iat[0] - r_f
        denominator = self.cumulative_returns().std()
        return numerator / denominator


class AssetGroupAnalyzer:
    """Analyzes many assets in a dataframe."""
    @validate_df(columns={'open', 'high', 'low', 'close'})
    def __init__(self, df, group_by='name'):
        """
        Create an `AssetGroupAnalyzer` object by passing in 
        a `pandas.DataFrame` and column to group by.
        """
        self.df = df
        if group_by not in self.df.columns:
            raise ValueError(
                f'`group_by` column "{group_by}" not in dataframe.'
            )
        self.group_by = group_by
        self.analyzers = self._composition_handler()

    def _composition_handler(self):
        """
        Create a dictionary mapping each group to its analyzer,
        taking advantage of composition instead of inheritance.
        """
        return dict(
            (group, StockAnalyzer(data))
            for group, data in self.df.groupby(self.group_by)
        )

    def analyze(self, func_name, **kwargs):
        """
        Run a `StockAnalyzer` method on all assets in the group.

        Parameters:
            - func_name: The name of the method to run.
            - kwargs: Additional keyword arguments to pass to the function.

        Returns:
            A dictionary mapping each asset to the result of the
            calculation of that function.
        """
        if not hasattr(StockAnalyzer, func_name):
            raise ValueError(
                f'StockAnalyzer has no "{func_name}" method.'
            )
        if not kwargs:
            kwargs = dict()
        return dict(
            (
                group, 
                getattr(analyzer, func_name)(**kwargs)
            )
            for group, analyzer in self.analyzers.items()
        )
