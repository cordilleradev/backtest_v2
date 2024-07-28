from enum import Enum
from typing import Optional, Tuple

class OptionType(Enum):
    """
    Enum representing the types of options.
    """
    CALL = "Call"
    PUT = "Put"

class StrategyType(Enum):
    """
    Enum representing different options trading strategies.
    """
    CALL = "Call"
    PUT = "Put"
    BEAR_PUT_SPREAD = "Bear Put Spread"
    BEAR_CALL_SPREAD = "Bear Call Spread"
    BULL_PUT_SPREAD = "Bull Put Spread"
    BULL_CALL_SPREAD = "Bull Call Spread"
    STRAP = "Strap"
    STRIP = "Strip"
    STRADDLE = "Straddle"
    STRANGLE = "Strangle"
    LONG_BUTTERFLY = "Long Butterfly"
    LONG_CONDOR = "Long Condor"

class VolatilityType(Enum):
    """
    Enum representing the types of market volatility.
    """
    LOW = "Low Volatility"
    HIGH = "High Volatility"

class Direction(Enum):
    """
    Enum representing the market direction.
    """
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"

class Bounds:
    """
    Class representing the bounds for profit and liquidation conditions.
    """
    def __init__(self, lower: Optional[float] = None, upper: Optional[float] = None, inside: bool = True):
        self.lower = lower
        self.upper = upper
        self.inside = inside

    def contains(self, value: float) -> bool:
        """
        Check if a value is within the bounds.

        Args:
            value (float): The value to check.

        Returns:
            bool: True if the value is within the bounds, False otherwise.
        """
        if self.inside:
            return (self.lower is None or value >= self.lower) and (self.upper is None or value <= self.upper)
        else:
            return (self.lower is not None and value < self.lower) or (self.upper is not None and value > self.upper)

class OptionsStrategy:
    """
    Base class for options trading strategies.
    """
    def __init__(self, open_price: float, strike_price: float, premium: float):
        self.open_price = open_price
        self.strike_price = strike_price
        self.premium = premium
        self.profit_bound = Bounds()
        self.liquidation_bound = Bounds()

    def is_profitable(self, current_price: float) -> bool:
        """
        Check if the strategy is profitable at the current price.

        Args:
            current_price (float): The current price of the underlying asset.

        Returns:
            bool: True if the strategy is profitable, False otherwise.
        """
        return self.profit_bound.contains(current_price)

    def should_liquidate(self, current_price: float) -> bool:
        """
        Check if the strategy should be liquidated at the current price.

        Args:
            current_price (float): The current price of the underlying asset.

        Returns:
            bool: True if the strategy should be liquidated, False otherwise.
        """
        return self.liquidation_bound.contains(current_price)

    @property
    def volatility_type(self) -> VolatilityType:
        """
        Get the type of market volatility for the strategy.

        Returns:
            VolatilityType: The volatility type.
        """
        raise NotImplementedError

    @property
    def direction(self) -> Direction:
        """
        Get the market direction for the strategy.

        Returns:
            Direction: The market direction.
        """
        raise NotImplementedError

class Call(OptionsStrategy):
    """
    Class representing a Call option strategy.
    """
    def __init__(self, open_price: float, strike_price: float, premium: float):
        super().__init__(open_price, strike_price, premium)
        self.profit_bound = Bounds(lower=self.strike_price + self.premium)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.HIGH

    @property
    def direction(self) -> Direction:
        return Direction.BULLISH

class Put(OptionsStrategy):
    """
    Class representing a Put option strategy.
    """
    def __init__(self, open_price: float, strike_price: float, premium: float):
        super().__init__(open_price, strike_price, premium)
        self.profit_bound = Bounds(upper=self.strike_price - self.premium)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.HIGH

    @property
    def direction(self) -> Direction:
        return Direction.BEARISH

class BearPutSpread(OptionsStrategy):
    """
    Class representing a Bear Put Spread strategy.
    """
    def __init__(self, open_price: float, long_strike: float, short_strike: float, net_premium: float):
        super().__init__(open_price, long_strike, net_premium)
        self.short_strike = short_strike
        self.profit_bound = Bounds(lower=self.short_strike - self.premium, upper=self.strike_price)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.LOW

    @property
    def direction(self) -> Direction:
        return Direction.BEARISH

class BearCallSpread(OptionsStrategy):
    """
    Class representing a Bear Call Spread strategy.
    """
    def __init__(self, open_price: float, long_strike: float, short_strike: float, net_premium: float):
        super().__init__(open_price, long_strike, net_premium)
        self.short_strike = short_strike
        self.profit_bound = Bounds(upper=self.short_strike + self.premium)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.LOW

    @property
    def direction(self) -> Direction:
        return Direction.BEARISH

class BullPutSpread(OptionsStrategy):
    """
    Class representing a Bull Put Spread strategy.
    """
    def __init__(self, open_price: float, long_strike: float, short_strike: float, net_premium: float):
        super().__init__(open_price, long_strike, net_premium)
        self.short_strike = short_strike
        self.profit_bound = Bounds(lower=self.short_strike - self.premium)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.LOW

    @property
    def direction(self) -> Direction:
        return Direction.BULLISH

class BullCallSpread(OptionsStrategy):
    """
    Class representing a Bull Call Spread strategy.
    """
    def __init__(self, open_price: float, long_strike: float, short_strike: float, net_premium: float):
        super().__init__(open_price, long_strike, net_premium)
        self.short_strike = short_strike
        self.profit_bound = Bounds(lower=self.strike_price + self.premium, upper=self.short_strike)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.LOW

    @property
    def direction(self) -> Direction:
        return Direction.BULLISH

class Strap(OptionsStrategy):
    """
    Class representing a Strap strategy.
    """
    def __init__(self, open_price: float, strike_price: float, call_premium: float, put_premium: float):
        super().__init__(open_price, strike_price, call_premium + put_premium)
        self.call_premium = call_premium
        self.put_premium = put_premium
        self.profit_bound = Bounds(lower=self.strike_price + self.premium / 2, upper=self.strike_price - self.premium)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.HIGH

    @property
    def direction(self) -> Direction:
        return Direction.BULLISH

class Strip(OptionsStrategy):
    """
    Class representing a Strip strategy.
    """
    def __init__(self, open_price: float, strike_price: float, call_premium: float, put_premium: float):
        super().__init__(open_price, strike_price, call_premium + put_premium)
        self.call_premium = call_premium
        self.put_premium = put_premium
        self.profit_bound = Bounds(lower=self.strike_price + self.premium, upper=self.strike_price - self.premium / 2)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.HIGH

    @property
    def direction(self) -> Direction:
        return Direction.BEARISH

class Straddle(OptionsStrategy):
    """
    Class representing a Straddle strategy.
    """
    def __init__(self, open_price: float, strike_price: float, call_premium: float, put_premium: float):
        super().__init__(open_price, strike_price, call_premium + put_premium)
        self.profit_bound = Bounds(lower=self.strike_price + self.premium, upper=self.strike_price - self.premium)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.HIGH

    @property
    def direction(self) -> Direction:
        return Direction.NEUTRAL

class Strangle(OptionsStrategy):
    """
    Class representing a Strangle strategy.
    """
    def __init__(self, open_price: float, call_strike: float, put_strike: float, call_premium: float, put_premium: float):
        super().__init__(open_price, (call_strike + put_strike) / 2, call_premium + put_premium)
        self.call_strike = call_strike
        self.put_strike = put_strike
        self.profit_bound = Bounds(lower=self.call_strike + self.premium, upper=self.put_strike - self.premium)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.HIGH

    @property
    def direction(self) -> Direction:
        return Direction.NEUTRAL

class LongButterfly(OptionsStrategy):
    """
    Class representing a Long Butterfly strategy.
    """
    def __init__(self, open_price: float, lower_strike: float, middle_strike: float, upper_strike: float, net_premium: float):
        super().__init__(open_price, middle_strike, net_premium)
        self.lower_strike = lower_strike
        self.upper_strike = upper_strike
        self.profit_bound = Bounds(lower=self.lower_strike + self.premium, upper=self.upper_strike - self.premium)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.LOW

    @property
    def direction(self) -> Direction:
        return Direction.NEUTRAL

class LongCondor(OptionsStrategy):
    """
    Class representing a Long Condor strategy.
    """
    def __init__(self, open_price: float, lower_strike: float, lower_middle_strike: float, upper_middle_strike: float, upper_strike: float, net_premium: float):
        super().__init__(open_price, (lower_middle_strike + upper_middle_strike) / 2, net_premium)
        self.lower_strike = lower_strike
        self.lower_middle_strike = lower_middle_strike
        self.upper_middle_strike = upper_middle_strike
        self.upper_strike = upper_strike
        self.profit_bound = Bounds(lower=self.lower_middle_strike, upper=self.upper_middle_strike)
        self.liquidation_bound = Bounds(lower=self.lower_strike, upper=self.upper_strike, inside=False)

    def is_profitable(self, current_price: float) -> bool:
        return self.profit_bound.contains(current_price)

    def should_liquidate(self, current_price: float) -> bool:
        return self.liquidation_bound.contains(current_price)

    @property
    def volatility_type(self) -> VolatilityType:
        return VolatilityType.LOW

    @property
    def direction(self) -> Direction:
        return Direction.NEUTRAL

def create_strategy(strategy_type: StrategyType, open_price: float, *args) -> OptionsStrategy:
    """
    Factory function to create an options strategy instance based on the strategy type.

    Args:
        strategy_type (StrategyType): The type of strategy to create.
        open_price (float): The opening price of the underlying asset.
        *args: Additional arguments required for the specific strategy.

    Returns:
        OptionsStrategy: The created options strategy instance.
    """
    if strategy_type == StrategyType.CALL:
        return Call(open_price, *args)
    elif strategy_type == StrategyType.PUT:
        return Put(open_price, *args)
    elif strategy_type == StrategyType.BEAR_PUT_SPREAD:
        return BearPutSpread(open_price, *args)
    elif strategy_type == StrategyType.BEAR_CALL_SPREAD:
        return BearCallSpread(open_price, *args)
    elif strategy_type == StrategyType.BULL_PUT_SPREAD:
        return BullPutSpread(open_price, *args)
    elif strategy_type == StrategyType.BULL_CALL_SPREAD:
        return BullCallSpread(open_price, *args)
    elif strategy_type == StrategyType.STRAP:
        return Strap(open_price, *args)
    elif strategy_type == StrategyType.STRIP:
        return Strip(open_price, *args)
    elif strategy_type == StrategyType.STRADDLE:
        return Straddle(open_price, *args)
    elif strategy_type == StrategyType.STRANGLE:
        return Strangle(open_price, *args)
    elif strategy_type == StrategyType.LONG_BUTTERFLY:
        return LongButterfly(open_price, *args)
    if strategy_type == StrategyType.LONG_CONDOR:
        return LongCondor(open_price, *args)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
