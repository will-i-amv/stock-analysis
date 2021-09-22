from distutils.core import setup

setup(
    name='stock_analysis',
    version='0.2',
    description='Classes for technical analysis of stocks.',
    author='William Vera',
    license='MIT',
    url='https://github.com/will-i-amv/stock-analysis',
    packages=['stock_analysis'],
    install_requires=[
        'matplotlib>=3.0.2',
        'numpy>=1.15.2',
        'pandas>=0.23.4',
        'pandas-datareader>=0.7.0',
        'seaborn>=0.11.0',
        'statsmodels>=0.11.1',
        'mplfinance>=0.12.7a4'
    ],
)
