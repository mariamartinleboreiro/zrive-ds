"""
Unit tests for module_1_meteo_api.
Tests cover: call_api, validate_response, get_data_meteo_api,
resample_monthly and fetch_all_cities using mocks so no real HTTP
requests are made.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.module_1.module_1_meteo_api import (
    VARIABLES,
    call_api,
    fetch_all_cities,
    get_data_meteo_api,
    resample_monthly,
    validate_response,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

MOCK_DAILY_RESPONSE = {
    "latitude": 40.416775,
    "longitude": -3.703790,
    "elevation": 667.0,
    "generationtime_ms": 1.5,
    "utc_offset_seconds": 0,
    "timezone": "UTC",
    "timezone_abbreviation": "UTC",
    "daily_units": {
        "time": "iso8601",
        "temperature_2m_mean": "°C",
        "precipitation_sum": "mm",
        "wind_speed_10m_max": "km/h",
    },
    "daily": {
        "time": ["2010-01-01", "2010-01-02", "2010-01-03"],
        "temperature_2m_mean": [5.0, 6.0, 7.0],
        "precipitation_sum": [0.0, 2.5, 1.0],
        "wind_speed_10m_max": [10.0, 15.0, 12.0],
    },
}


def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.ok = status_code < 400
    mock_resp.json.return_value = json_data
    mock_resp.content = True
    return mock_resp


# ---------------------------------------------------------------------------
# Tests for call_api
# ---------------------------------------------------------------------------


class TestCallApi:
    def test_returns_json_on_success(self):
        """call_api should return parsed JSON on a 200 response."""
        mock_resp = _make_mock_response({"result": "ok"}, 200)
        with patch("requests.get", return_value=mock_resp):
            result = call_api("http://fake-url", {})
        assert result == {"result": "ok"}

    def test_raises_on_client_error(self):
        """call_api should raise RuntimeError immediately on 400 errors."""
        mock_resp = _make_mock_response({"error": True, "reason": "Bad parameter"}, 400)
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="HTTP 400"):
                call_api("http://fake-url", {})

    def test_retries_on_rate_limit_then_succeeds(self):
        """call_api should retry on 429 and succeed on a subsequent 200."""
        mock_429 = _make_mock_response({}, 429)
        mock_200 = _make_mock_response({"result": "ok"}, 200)

        with patch("requests.get", side_effect=[mock_429, mock_200]):
            with patch("time.sleep"):  # avoid actual waiting
                result = call_api("http://fake-url", {}, max_retries=3, backoff=0.0)

        assert result == {"result": "ok"}

    def test_raises_after_max_retries(self):
        """call_api should raise RuntimeError after exhausting all retries."""
        mock_503 = _make_mock_response({}, 503)

        with patch("requests.get", return_value=mock_503):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="failed after"):
                    call_api("http://fake-url", {}, max_retries=2, backoff=0.0)

    def test_retries_on_network_exception(self):
        """call_api should retry on RequestException."""
        import requests as req_lib

        mock_200 = _make_mock_response({"result": "ok"}, 200)

        with patch(
            "requests.get",
            side_effect=[req_lib.exceptions.ConnectionError("timeout"), mock_200],
        ):
            with patch("time.sleep"):
                result = call_api("http://fake-url", {}, max_retries=3, backoff=0.0)

        assert result == {"result": "ok"}


# ---------------------------------------------------------------------------
# Tests for validate_response
# ---------------------------------------------------------------------------


class TestValidateResponse:
    def test_valid_response_passes(self):
        """validate_response should not raise on a well-formed response."""
        validate_response(MOCK_DAILY_RESPONSE)  # should not raise

    def test_missing_top_level_key_raises(self):
        """validate_response should raise ValueError for missing top-level keys."""
        bad = {k: v for k, v in MOCK_DAILY_RESPONSE.items() if k != "daily"}
        with pytest.raises(ValueError, match="top-level keys"):
            validate_response(bad)

    def test_missing_daily_variable_raises(self):
        """validate_response should raise ValueError for missing daily variable."""
        bad = {**MOCK_DAILY_RESPONSE}
        bad["daily"] = {"time": ["2010-01-01"], "temperature_2m_mean": [5.0]}
        with pytest.raises(ValueError, match="daily.*missing"):
            validate_response(bad)


# ---------------------------------------------------------------------------
# Tests for get_data_meteo_api
# ---------------------------------------------------------------------------


class TestGetDataMeteoApi:
    def test_returns_dataframe_for_valid_city(self):
        """get_data_meteo_api should return a DataFrame with expected columns."""
        with patch(
            "src.module_1.module_1_meteo_api.call_api",
            return_value=MOCK_DAILY_RESPONSE,
        ):
            df = get_data_meteo_api("Madrid")

        assert isinstance(df, pd.DataFrame)
        assert "city" in df.columns
        for var in VARIABLES:
            assert var in df.columns

    def test_dataframe_has_datetime_index(self):
        """get_data_meteo_api should set a DatetimeIndex."""
        with patch(
            "src.module_1.module_1_meteo_api.call_api",
            return_value=MOCK_DAILY_RESPONSE,
        ):
            df = get_data_meteo_api("Madrid")

        assert isinstance(df.index, pd.DatetimeIndex)

    def test_city_column_value_is_correct(self):
        """The 'city' column must contain the requested city name."""
        with patch(
            "src.module_1.module_1_meteo_api.call_api",
            return_value=MOCK_DAILY_RESPONSE,
        ):
            df = get_data_meteo_api("London")

        assert (df["city"] == "London").all()

    def test_raises_for_unknown_city(self):
        """get_data_meteo_api should raise ValueError for an unknown city."""
        with pytest.raises(ValueError, match="not found"):
            get_data_meteo_api("Atlantis")

    def test_row_count_matches_response(self):
        """The number of rows must equal the number of time entries returned."""
        with patch(
            "src.module_1.module_1_meteo_api.call_api",
            return_value=MOCK_DAILY_RESPONSE,
        ):
            df = get_data_meteo_api("Rio")

        assert len(df) == len(MOCK_DAILY_RESPONSE["daily"]["time"])


# ---------------------------------------------------------------------------
# Tests for resample_monthly
# ---------------------------------------------------------------------------


class TestResampleMonthly:
    def _build_daily_df(self, city: str = "Madrid") -> pd.DataFrame:
        """Create a small daily DataFrame for testing."""
        dates = pd.date_range("2010-01-01", periods=60, freq="D")
        df = pd.DataFrame(
            {
                "temperature_2m_mean": [5.0] * 60,
                "precipitation_sum": [1.0] * 60,
                "wind_speed_10m_max": [10.0] * 60,
                "city": city,
            },
            index=dates,
        )
        return df

    def test_returns_dataframe(self):
        """resample_monthly should return a DataFrame."""
        df = self._build_daily_df()
        result = resample_monthly(df)
        assert isinstance(result, pd.DataFrame)

    def test_monthly_temperature_is_mean(self):
        """Monthly temperature should be the mean of daily values."""
        df = self._build_daily_df()
        result = resample_monthly(df)
        assert (result["temperature_2m_mean"] == 5.0).all()

    def test_monthly_precipitation_is_sum(self):
        """Monthly precipitation should be the sum of daily values."""
        df = self._build_daily_df()
        result = resample_monthly(df)
        # January has 31 days, February has 28 days (2010 is not a leap year)
        jan_total = result.loc[result.index.month == 1, "precipitation_sum"].values[0]
        assert jan_total == pytest.approx(31.0)

    def test_fewer_rows_than_daily(self):
        """Monthly data must have fewer (or equal) rows than daily data."""
        df = self._build_daily_df()
        result = resample_monthly(df)
        assert len(result) < len(df)

    def test_multiple_cities(self):
        """resample_monthly should handle multiple cities correctly."""
        df_mad = self._build_daily_df("Madrid")
        df_lon = self._build_daily_df("London")
        combined = pd.concat([df_mad, df_lon])
        result = resample_monthly(combined)
        assert set(result["city"].unique()) == {"Madrid", "London"}


# ---------------------------------------------------------------------------
# Tests for fetch_all_cities
# ---------------------------------------------------------------------------


class TestFetchAllCities:
    def test_combines_all_cities(self):
        """fetch_all_cities should return data for all three cities."""
        with patch(
            "src.module_1.module_1_meteo_api.call_api",
            return_value=MOCK_DAILY_RESPONSE,
        ):
            df = fetch_all_cities()

        assert set(df["city"].unique()) == {"Madrid", "London", "Rio"}

    def test_accepts_custom_city_list(self):
        """fetch_all_cities should respect a custom city list."""
        with patch(
            "src.module_1.module_1_meteo_api.call_api",
            return_value=MOCK_DAILY_RESPONSE,
        ):
            df = fetch_all_cities(cities=["Madrid", "London"])

        assert set(df["city"].unique()) == {"Madrid", "London"}
