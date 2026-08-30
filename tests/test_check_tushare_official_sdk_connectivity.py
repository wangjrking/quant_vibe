import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import check_tushare_official_sdk_connectivity as tool


class CheckTushareOfficialSdkConnectivityTests(unittest.TestCase):
    def test_probe_uses_get_pro_and_reports_shape(self):
        fake_frame = mock.Mock()
        fake_frame.shape = (3, 2)
        fake_pro = mock.Mock()
        fake_pro.query.return_value = fake_frame

        with mock.patch.object(tool, "_project_root", return_value=Path.cwd()):
            with mock.patch.dict("sys.modules", {"tushare": mock.Mock(__version__="x")}):
                with mock.patch("data_load_module.get_pro", return_value=fake_pro) as get_pro:
                    result = tool._probe("https://api.tushare.pro", "token", "20260710")

        self.assertTrue(result["success"])
        self.assertEqual(result["shape"], [3, 2])
        get_pro.assert_called_once_with("token", endpoint="https://api.tushare.pro")
        fake_pro.query.assert_called_once_with("daily", trade_date="20260710")

    def test_main_writes_json_and_returns_failure_without_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "probe.json"
            with mock.patch.object(tool, "_load_token", side_effect=RuntimeError("missing")):
                with mock.patch("sys.argv", ["tool", "--trade-date", "20260710", "--output", str(output)]):
                    code = tool.main()

            self.assertEqual(code, 2)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["success"])
            self.assertFalse(payload["business_data_written"])
            self.assertEqual(payload["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
