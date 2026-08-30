import unittest

import run_juejin_signal_backtest as runner


class RunJuejinSignalBacktestArgsTest(unittest.TestCase):
    def test_explicit_database_disable_flags_are_supported(self):
        args = runner.parse_args(
            [
                "--strategy-dir",
                "strategy",
                "--signal-file",
                "signals.csv",
                "--log-file",
                "run.log",
                "--disable-score-db",
                "--disable-market-db",
            ]
        )

        self.assertTrue(args.disable_score_db)
        self.assertTrue(args.disable_market_db)


if __name__ == "__main__":
    unittest.main()
