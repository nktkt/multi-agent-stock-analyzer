import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.technical_analyst import (
    sma,
    ema,
    ema_series,
    rsi,
    macd,
    bollinger_bands,
)


class TestSMA(unittest.TestCase):
    def test_sma_constant_series(self):
        """Constant input should yield a constant output equal to that value."""
        values = [5.0] * 30
        self.assertAlmostEqual(sma(values, window=10), 5.0, places=4)
        self.assertAlmostEqual(sma(values, window=20), 5.0, places=4)

    def test_sma_known_short_sequence(self):
        """SMA of [1,2,3,4,5] with window=3 should equal (3+4+5)/3 = 4.0."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(sma(values, window=3), 4.0, places=4)
        self.assertAlmostEqual(sma(values, window=5), 3.0, places=4)

    def test_sma_insufficient_data_returns_none(self):
        """Returns None when fewer values than the window are provided."""
        self.assertIsNone(sma([1.0, 2.0], window=5))


class TestEMA(unittest.TestCase):
    def test_ema_first_value_is_sma_seed(self):
        """ema_series first value equals SMA of the first `window` values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        series = ema_series(values, window=3)
        self.assertAlmostEqual(series[0], 2.0, places=4)

    def test_ema_converges_toward_steady_input(self):
        """EMA of a long constant tail should converge toward that constant."""
        values = [1.0, 2.0, 3.0] + [10.0] * 200
        result = ema(values, window=5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 10.0, places=4)

    def test_ema_constant_input_is_constant(self):
        """Constant input means EMA equals that constant everywhere."""
        values = [7.0] * 50
        series = ema_series(values, window=10)
        for v in series:
            self.assertAlmostEqual(v, 7.0, places=4)


class TestRSI(unittest.TestCase):
    def test_rsi_all_up_series_near_100(self):
        """Strictly increasing series should produce RSI near 100."""
        values = [float(i) for i in range(1, 40)]
        result = rsi(values, window=14)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 100.0, places=4)

    def test_rsi_all_down_series_near_0(self):
        """Strictly decreasing series should produce RSI near 0."""
        values = [float(i) for i in range(40, 0, -1)]
        result = rsi(values, window=14)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=4)

    def test_rsi_insufficient_length_returns_none(self):
        """RSI requires at least window+1 values."""
        self.assertIsNone(rsi([1.0, 2.0, 3.0], window=14))


class TestMACD(unittest.TestCase):
    def test_macd_returns_three_aligned_values(self):
        """MACD returns a 3-tuple of (macd_line, signal_line, histogram)."""
        values = [100.0 + i * 0.5 for i in range(60)]
        result = macd(values, fast=12, slow=26, signal=9)
        self.assertEqual(len(result), 3)
        macd_line, signal_line, histogram = result
        self.assertIsNotNone(macd_line)
        self.assertIsNotNone(signal_line)
        self.assertIsNotNone(histogram)
        self.assertAlmostEqual(histogram, macd_line - signal_line, places=4)

    def test_macd_insufficient_data(self):
        """Very short input should not produce a full MACD triple."""
        result = macd([1.0, 2.0, 3.0], fast=12, slow=26, signal=9)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, (None, None, None))


class TestBollingerBands(unittest.TestCase):
    def test_bb_mid_equals_sma(self):
        """Middle band equals SMA over the same window."""
        values = [float(i) for i in range(1, 31)]
        upper, middle, lower = bollinger_bands(values, window=20, num_std=2.0)
        self.assertAlmostEqual(middle, sma(values, window=20), places=4)

    def test_bb_ordering_with_variance(self):
        """With variance present, upper > middle > lower."""
        values = [float(i % 7) + 10.0 for i in range(40)]
        upper, middle, lower = bollinger_bands(values, window=20, num_std=2.0)
        self.assertGreater(upper, middle)
        self.assertGreater(middle, lower)

    def test_bb_constant_series_collapses_bands(self):
        """Constant input yields zero variance so upper == middle == lower."""
        values = [5.0] * 30
        upper, middle, lower = bollinger_bands(values, window=20, num_std=2.0)
        self.assertAlmostEqual(upper, middle, places=4)
        self.assertAlmostEqual(middle, lower, places=4)

    def test_bb_insufficient_data_returns_none_tuple(self):
        """Returns (None, None, None) when fewer values than window."""
        result = bollinger_bands([1.0, 2.0], window=20, num_std=2.0)
        self.assertEqual(result, (None, None, None))


if __name__ == "__main__":
    unittest.main()
