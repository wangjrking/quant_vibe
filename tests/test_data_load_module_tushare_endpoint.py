import os
import unittest
from unittest import mock

from tushare.pro.client import DataApi

import data_load_module


class TushareEndpointConfigTests(unittest.TestCase):
    def setUp(self):
        self.original_endpoint = DataApi._DataApi__http_url

    def tearDown(self):
        DataApi._DataApi__http_url = self.original_endpoint
        os.environ.pop("TUSHARE_PRO_API_URL", None)
        os.environ.pop("TUSHARE_API_URL", None)

    def test_configure_endpoint_normalizes_base_host(self):
        endpoint = data_load_module.configure_tushare_pro_endpoint("https://api.tushare.pro")

        self.assertEqual(endpoint, "https://api.tushare.pro/dataapi")
        self.assertEqual(DataApi._DataApi__http_url, "https://api.tushare.pro/dataapi")

    def test_configure_endpoint_keeps_dataapi_suffix(self):
        endpoint = data_load_module.configure_tushare_pro_endpoint("http://api.tushare.pro/dataapi/")

        self.assertEqual(endpoint, "http://api.tushare.pro/dataapi")
        self.assertEqual(DataApi._DataApi__http_url, "http://api.tushare.pro/dataapi")

    def test_get_pro_uses_official_sdk_with_configured_endpoint(self):
        with mock.patch.object(data_load_module.ts, "pro_api", return_value=object()) as pro_api:
            result = data_load_module.get_pro("fake-token", endpoint="https://api.tushare.pro")

        self.assertIsNotNone(result)
        pro_api.assert_called_once_with("fake-token")
        self.assertEqual(DataApi._DataApi__http_url, "https://api.tushare.pro/dataapi")

    def test_env_endpoint_is_used_without_explicit_endpoint(self):
        os.environ["TUSHARE_PRO_API_URL"] = "https://api.tushare.pro"

        with mock.patch.object(data_load_module.ts, "pro_api", return_value=object()):
            data_load_module.get_pro("fake-token")

        self.assertEqual(DataApi._DataApi__http_url, "https://api.tushare.pro/dataapi")


if __name__ == "__main__":
    unittest.main()
