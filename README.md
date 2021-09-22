# Stock Analysis

This is pandas-based package intended for automating the technical analysis of a stock, index or a cryptocurrency such as bitcoin. 

## Setup

* Clone this repo:

```shell
$ git clone https://github.com/will-i-amv/stock-analysis.git 
```

* Go to the cloned repo's root directory and execute the following commands:

```shell
$ pip3 install -e stock-analysis # should install requirements.txt packages
$ pip3 install -r requirements.txt # if not, install them explicitly
```

## Usage

This section will show some of the functionality of each module; however, it is by no means exhaustive.

### Getting data with the stock_reader module

```python
from stock_analysis import StockReader

reader = StockReader('2017-01-01', '2018-12-31')

# Get faang data
tickers = ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOG']
fb, aapl, amzn, nflx, goog = (
    reader.get_ticker_data(ticker) \
    for ticker in tickers
)

# Get S&P 500 data
sp = reader.get_index_data('S&P 500')

# Get bitcoin data in USD
bitcoin = reader.get_bitcoin_data('USD')

```

* Group stocks by name and display their summary statistics:

```python
from stock_analysis.utils import group_stocks, describe_group

faang = group_stocks(
    {
        'Facebook': fb,
        'Apple': aapl,
        'Amazon': amzn,
        'Netflix': nflx,
        'Google': goog
    }
)

describe_group(faang)
```

* Group stocks by date and sum their columns to build a portfolio:

```python
from stock_analysis.utils import make_portfolio

faang_portfolio = make_portfolio(faang)
```

### Visualizing data with the stock_visualizer module

#### For a single asset

* Evolution over time:

```python
import matplotlib.pyplot as plt
from stock_analysis import StockVisualizer

netflix_viz = StockVisualizer(nflx)

ax = netflix_viz.plot_evolution_over_time(
    'close',
    figsize=(10, 4),
    legend=False,
    title='Netflix closing price over time'
)
netflix_viz.plot_reference_line(
    ax,
    x=nflx.high.idxmax(),
    color='k',
    linestyle=':',
    label=f'highest value ({nflx.high.idxmax():%b %d})',
    alpha=0.5
)
ax.set_ylabel('price ($)')
plt.show()
```

<img src="images/netflix_line_plot.png?raw=true" align="center" width="600" alt="line plot with reference line">

* After hours trades:

```python
netflix_viz.plot_after_hours_trades()
plt.show()
```

<img src="images/netflix_after_hours_trades.png?raw=true" align="center" width="800" alt="after hours trades plot">

* Differential in closing price versus another asset:

```python
netflix_viz.plot_area_between_close_prices(fb)
plt.show()
```
<img src="images/nflx_vs_fb_closing_price.png?raw=true" align="center" width="600" alt="differential between NFLX and FB">

* Candlestick plots with resampling (uses the `mplfinance` library):

```python
netflix_viz.plot_candlestick(
    resample='2W', 
    volume=True, 
    xrotation=90, 
    datetime_format='%Y-%b -'
)
```

<img src="images/candlestick.png?raw=true" align="center" width="600" alt="resampled candlestick plot">

*Note: run `help()` on `StockVisualizer` for more visualization options*

#### For asset groups

* Correlation heatmap:

```python
from stock_analysis import AssetGroupVisualizer

faang_viz = AssetGroupVisualizer(faang)
faang_viz.plot_heatmap(True)
```

<img src="images/faang_heatmap.png?raw=true" align="center" width="450" alt="correlation heatmap">


### Analyzing data with the stock_analysis module

Below are some metrics you can calculate.

#### For a single asset

```python
from stock_analysis import StockAnalyzer

nflx_analyzer = stock_analysis.StockAnalyzer(nflx)

# Annualized volatility
nflx_analyzer.calc_annualized_volatility()

# Sharpe ratio
nflx_analyzer.calc_sharpe_ratio()
```

#### For asset groups

* Methods of the `StockAnalyzer` class can be accessed by name with the `AssetGroupAnalyzer` class's `analyze()` method.

```python
from stock_analysis import AssetGroupAnalyzer

faang_analyzer = AssetGroupAnalyzer(faang)

faang_analyzer.analyze('annualized_volatility')
faang_analyzer.analyze('calc_alpha')
faang_analyzer.analyze('calc_beta')
```

### Modeling data with the stock_modeler module

```python
from stock_analysis import StockModeler
```

#### Time series decomposition

* Build the model and plot the results:

```python
decomposition = StockModeler.create_decompose_object(nflx, 20)
fig = decomposition.plot()
plt.show()
```

<img src="images/nflx_ts_decomposition.png?raw=true" align="center" width="450" alt="time series decomposition">

#### ARIMA model

* Build the model:

```python
arima_model = StockModeler.create_arima_model(nflx, 10, 1, 5)
```

* Check the residuals:

```python
StockModeler.plot_residuals(arima_model)
plt.show()
```

<img src="images/arima_residuals.png?raw=true" align="center" width="650" alt="ARIMA residuals">

* Run the model and plot the predictions:

```python
arima_ax = StockModeler.calc_arima_predictions(
    arima_model, 
    start=start, 
    end=end,
    df=nflx, 
    ax=axes[0], 
    title='ARIMA'
)
plt.show()
```

<img src="images/arima_predictions.png?raw=true" align="center" width="450" alt="ARIMA predictions">

#### Linear regression model

* Build the model:

```python
X, Y, lm = StockModeler.create_linear_regression_model(nflx)
```

* Plot the residuals:

```python
StockModeler.plot_residuals(lm)
plt.show()
```

<img src="images/lm_residuals.png?raw=true" align="center" width="650" alt="linear regression residuals">

* Run the model and plot the predictions:

```python
linear_reg = StockModeler.calc_linear_regression_predictions(
    lm, start=start, 
    end=end,
    df=nflx, 
    ax=axes[1], 
    title='Linear Regression'
)
plt.show()
```

<img src="images/lm_predictions.png?raw=true" align="center" width="450" alt="linear regression predictions">
