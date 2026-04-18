import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.pattern_recognizer import (
    find_local_extrema,
    detect_double_top,
    detect_trend,
)


class TestFindLocalExtrema(unittest.TestCase):
    def test_hand_crafted_peak_and_trough(self):
        """A series with one clear peak and one clear trough is detected."""
        series = [
            1.0, 2.0, 3.0, 4.0, 5.0,
            10.0,
            5.0, 4.0, 3.0, 2.0, 1.0,
            0.5,
            1.0, 2.0, 3.0, 4.0, 5.0,
        ]
        peaks, troughs = find_local_extrema(series, window=5)
        peak_indices = [idx for idx, _ in peaks]
        trough_indices = [idx for idx, _ in troughs]
        self.assertIn(5, peak_indices)
        self.assertIn(11, trough_indices)
        peak_values = dict(peaks)
        trough_values = dict(troughs)
        self.assertAlmostEqual(peak_values[5], 10.0, places=4)
        self.assertAlmostEqual(trough_values[11], 0.5, places=4)

    def test_too_short_series_returns_empty(self):
        """Series shorter than 2*window+1 returns empty peaks and troughs."""
        peaks, troughs = find_local_extrema([1.0, 2.0, 3.0], window=5)
        self.assertEqual(peaks, [])
        self.assertEqual(troughs, [])

    def test_monotonic_series_has_no_interior_extrema(self):
        """A strictly monotonic series has no interior peaks or troughs."""
        series = [float(i) for i in range(20)]
        peaks, troughs = find_local_extrema(series, window=3)
        self.assertEqual(peaks, [])
        self.assertEqual(troughs, [])


class TestDetectDoubleTop(unittest.TestCase):
    def test_obvious_double_top(self):
        """Two near-equal peaks separated by enough distance are detected."""
        peaks = [(10, 100.0), (25, 100.5)]
        results = detect_double_top(peaks)
        self.assertEqual(len(results), 1)
        pattern = results[0]
        self.assertEqual(pattern["pattern"], "double_top")
        self.assertEqual(pattern["indices"], [10, 25])
        self.assertAlmostEqual(pattern["prices"][0], 100.0, places=4)
        self.assertAlmostEqual(pattern["prices"][1], 100.5, places=4)

    def test_peaks_too_close_rejected(self):
        """Peaks with index distance < 5 do not form a double top."""
        peaks = [(10, 100.0), (12, 100.2)]
        self.assertEqual(detect_double_top(peaks), [])

    def test_peaks_too_different_rejected(self):
        """Peaks differing by more than 3% are not a double top."""
        peaks = [(10, 100.0), (25, 150.0)]
        self.assertEqual(detect_double_top(peaks), [])

    def test_empty_peaks_returns_empty(self):
        """No peaks means no double tops."""
        self.assertEqual(detect_double_top([]), [])


class TestDetectTrend(unittest.TestCase):
    def test_monotonic_increase_is_uptrend(self):
        """A strictly increasing series is classified as an uptrend."""
        closes = [100.0 + i * 1.0 for i in range(30)]
        self.assertEqual(detect_trend(closes), "uptrend")

    def test_monotonic_decrease_is_downtrend(self):
        """A strictly decreasing series is classified as a downtrend."""
        closes = [200.0 - i * 1.0 for i in range(30)]
        self.assertEqual(detect_trend(closes), "downtrend")

    def test_flat_series_is_sideways(self):
        """A flat/constant series is classified as sideways."""
        closes = [100.0] * 30
        self.assertEqual(detect_trend(closes), "sideways")

    def test_near_flat_noise_is_sideways(self):
        """Small oscillations around a flat mean are classified as sideways."""
        closes = [100.0 + (0.1 if i % 2 == 0 else -0.1) for i in range(30)]
        self.assertEqual(detect_trend(closes), "sideways")

    def test_short_series_defaults_sideways(self):
        """Fewer than 6 points falls back to sideways."""
        self.assertEqual(detect_trend([1.0, 2.0, 3.0]), "sideways")


if __name__ == "__main__":
    unittest.main()
