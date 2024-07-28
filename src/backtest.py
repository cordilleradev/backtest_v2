import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Callable, Dict, Any, Tuple
from enum import Enum
from src.normalize import DataMerger
from src.bound import StrategyType, OptionsStrategy

class Outcome(Enum):
    """
    Enum representing the possible outcomes of a backtest.
    """
    PROFITABLE = "Profitable"
    UNPROFITABLE = "Unprofitable"
    LIQUIDATED = "Liquidated"
    SKIPPED = "Skipped"

class BacktestResult:
    """
    Class to store and analyze the results of a backtest.
    """
    def __init__(self, outcomes: pd.DataFrame, merged_data: pd.DataFrame):
        """
        Initialize the BacktestResult with outcomes and merged data.

        Args:
            outcomes (pd.DataFrame): DataFrame containing the outcomes of the backtest.
            merged_data (pd.DataFrame): DataFrame containing the merged dataset used for backtesting.
        """
        self.merged_data = merged_data
        self.outcomes = outcomes

    def plot_results(self, strategy_name: str):
        """
        Plot the results of the backtest.

        Args:
            strategy_name (str): Name of the strategy being backtested.
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot price with a darker blue
        ax.plot(self.merged_data['date'], self.merged_data['close'], label='Price', color='#0000FF', linewidth=2)

        # Color background based on outcomes with lighter colors
        colors = {
            Outcome.PROFITABLE.value: '#90EE90',  # Light green
            Outcome.UNPROFITABLE.value: '#F7502C',  # Light yellow
            Outcome.LIQUIDATED.value: '#8A9BEE',  # Light purple
            Outcome.SKIPPED.value: '#FFFFFF'  # White
        }

        for _, row in self.outcomes.iterrows():
            color = colors[row['outcome']]
            ax.axvspan(row['start_date'], row['end_date'], alpha=0.3, color=color)

        # Plot other metrics with darker colors
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.1))

        dark_colors = ['#8B0000', '#006400', '#00008B', '#8B008B', '#A0522D']  # Dark red, green, blue, purple, brown

        for i, column in enumerate(self.merged_data.columns):
            if column not in ['date', 'open', 'high', 'low', 'close']:
                color = dark_colors[i % len(dark_colors)]
                if i % 2 == 0:
                    ax2.plot(self.merged_data['date'], self.merged_data[column], label=column, color=color, linewidth=1.5)
                else:
                    ax3.plot(self.merged_data['date'], self.merged_data[column], label=column, color=color, linewidth=1.5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{strategy_name} Backtest Results')

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

        plt.tight_layout()
        plt.show()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from the backtest results.

        Returns:
            Dict[str, Any]: Dictionary containing statistics such as total trades, win rate, and loss rate.
        """
        total_trades = len(self.outcomes[self.outcomes['outcome'] != Outcome.SKIPPED.value])
        profitable_trades = len(self.outcomes[self.outcomes['outcome'] == Outcome.PROFITABLE.value])
        unprofitable_trades = len(self.outcomes[self.outcomes['outcome'] == Outcome.UNPROFITABLE.value])
        liquidated_trades = len(self.outcomes[self.outcomes['outcome'] == Outcome.LIQUIDATED.value])

        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'unprofitable_trades': unprofitable_trades,
            'liquidated_trades': liquidated_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'loss_rate': (unprofitable_trades + liquidated_trades) / total_trades if total_trades > 0 else 0
        }

class OptionsBacktester:
    """
    Class to perform backtesting on options trading strategies.
    """
    def __init__(self, data_merger: DataMerger):
        """
        Initialize the OptionsBacktester with a DataMerger instance.

        Args:
            data_merger (DataMerger): Instance of DataMerger containing the merged dataset.
        """
        self.data_merger = data_merger
        self.merged_data = data_merger.get_merged_data()

    def backtest(self,
                 strategy_type: StrategyType,
                 period_days: int,
                 profit_lower_bound: float,
                 profit_upper_bound: float,
                 liquidation_lower_bound: float,
                 liquidation_upper_bound: float,
                 start_position: Callable[[pd.DataFrame], bool],
                 start_from: str = None) -> BacktestResult:
        """
        Perform backtesting on a specified strategy.

        Args:
            strategy_type (StrategyType): The type of strategy to backtest.
            period_days (int): The duration for each backtest period.
            profit_lower_bound (float): Lower bound for profit as a fraction.
            profit_upper_bound (float): Upper bound for profit as a fraction.
            liquidation_lower_bound (float): Lower bound for liquidation as a fraction.
            liquidation_upper_bound (float): Upper bound for liquidation as a fraction.
            start_position (Callable[[pd.DataFrame], bool]): Function to determine if a position should be started.
            start_from (str, optional): Date to start backtesting from. Defaults to None.

        Returns:
            BacktestResult: The result of the backtest.
        """
        results = []

        if start_from:
            data = self.merged_data[self.merged_data['date'] >= start_from]
        else:
            data = self.merged_data

        for i in range(len(data) - period_days + 1):
            df = data.iloc[i:i+period_days]

            start_date = df.iloc[0]['date']
            end_date = df.iloc[-1]['date']
            open_price = df.iloc[0]['open']

            # Check if we should start a position
            if not start_position(data.iloc[:i]):
                results.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'outcome': Outcome.SKIPPED.value,
                    'open_price': round(open_price, 2),
                    'close_price': round(df.iloc[-1]['close'], 2),
                    'liquidation_price': 0
                })
                continue

            # Create strategy
            strategy = self._create_strategy(strategy_type, open_price,
                                             profit_lower_bound, profit_upper_bound,
                                             liquidation_lower_bound, liquidation_upper_bound)

            # Determine outcome
            outcome, liquidation_price = self._get_outcome(df, strategy)

            results.append({
                'start_date': start_date,
                'end_date': end_date,
                'outcome': outcome.value,
                'open_price': round(open_price, 2),
                'close_price': round(df.iloc[-1]['close'], 2),
                'liquidation_price': round(liquidation_price, 2) if liquidation_price else 0
            })

        outcomes_df = pd.DataFrame(results)
        return BacktestResult(outcomes_df, data)

    def _create_strategy(self, strategy_type: StrategyType, open_price: float,
                         profit_lower: float, profit_upper: float,
                         liquidation_lower: float, liquidation_upper: float) -> OptionsStrategy:
        """
        Create an options strategy.

        Args:
            strategy_type (StrategyType): The type of strategy to create.
            open_price (float): The opening price.
            profit_lower (float): Lower bound for profit as a fraction.
            profit_upper (float): Upper bound for profit as a fraction.
            liquidation_lower (float): Lower bound for liquidation as a fraction.
            liquidation_upper (float): Upper bound for liquidation as a fraction.

        Returns:
            OptionsStrategy: The created options strategy.
        """
        from src.bound import create_strategy

        return create_strategy(
            strategy_type,
            open_price,
            open_price * (1 - liquidation_lower),
            open_price * (1 - profit_lower),
            open_price * (1 + profit_upper),
            open_price * (1 + liquidation_upper),
            0  # Net premium is not used in this simplified model
        )

    def _get_outcome(self, df: pd.DataFrame, strategy: OptionsStrategy) -> Tuple[Outcome, float]:
        """
        Determine the outcome of a strategy over a given period.

        Args:
            df (pd.DataFrame): DataFrame containing the price data for the period.
            strategy (OptionsStrategy): The options strategy being tested.

        Returns:
            Tuple[Outcome, float]: The outcome of the strategy and the liquidation price if applicable.
        """
        for _, row in df.iterrows():
            if strategy.should_liquidate(row['low']) or strategy.should_liquidate(row['high']):
                return Outcome.LIQUIDATED, row['low'] if strategy.should_liquidate(row['low']) else row['high']

        if strategy.is_profitable(df.iloc[-1]['close']):
            return Outcome.PROFITABLE, 0
        else:
            return Outcome.UNPROFITABLE, 0
