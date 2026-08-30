# Execution Gateway

This module is the isolated execution-side skeleton for the project's future
`shadow -> paper -> live` promotion path.

It is intentionally narrower than the research and production signal chain:

- It only accepts already-produced formal latest batches from `L5/L6/L7`.
- It does not train models, change strategy parameters, or modify upstream
  production assets.
- It defaults to non-executable behavior even when signals are present.
- It must not bypass `pending_buy_day_hard_gate`,
  `ready_for_human_confirmation_execution`, audit, owner approval, or external
  kill switches.

## Current Modules

- `contracts.py`
  - `ExecutionAuthorization`, `OrderIntent`, account snapshots, market
    snapshots, and risk decisions.
- `policy.py`
  - Loads execution authorization JSON and verifies `policy_hash`.
- `signal_loader.py`
  - Converts formal latest CSV/status assets into auditable order intents.
- `pretrade_risk.py`
  - Performs authorization, expiry, ST, quote freshness, cash, position, and
    kill-switch checks.
- `order_state.py`
  - Defines the allowed deterministic order state machine.
- `kill_switch.py`
  - AI-process-external kill switch contract backed by a simple JSON file.
- `live_arm.py`
  - Short-lived live execution arm contract. Live mode requires this file in
    addition to the execution policy.
- `ledger.py`
  - Append-only JSONL event ledger for prepared/submitted/shadowed/acknowledged
    events.
- `state_store.py`
  - SQLite state store for batches, prepared orders, and broker
    acknowledgements.
- `broker_adapter.py`
  - `ShadowBrokerAdapter`, `PaperBrokerAdapter`, and `MiniQmtBrokerAdapter`.
  - The MiniQMT adapter is wired to the local `xtquant` API shape but remains
    fail-closed by default.
- `executor.py`
  - Converts approved risk decisions into A-share order requests with board-lot
    sizing and sellable-share defaults.
- `reconciliation.py`
  - Lightweight reconciliation between expected orders, local ledgers, and
    broker-visible orders.
- `service.py`
  - Evaluates a latest batch into structured execution decisions.
- `../tools/run_execution_gateway_shadow.py`
  - One-shot dry-run over latest CSV/status, holdings, and market snapshot.
- `../tools/run_execution_gateway_batch.py`
  - End-to-end local runner for `latest -> risk -> prepared order -> ledger ->
    shadow/paper/live adapter -> reconciliation`.

## Hard Boundaries

The current implementation still does **not**:

- auto-submit real trades by default
- replace the broker's authoritative order/trade ledger
- replace full end-of-day reconciliation
- store live secrets in the repo
- self-enable live execution from inside the AI workflow

`live` mode now requires all of the following at the same time:

1. an execution policy with `stage=live`
2. `--allow-live-submit`
3. a valid short-lived `live arm` file
4. kill switch checks passing
5. valid formal latest assets from the audited production chain

If any of these are missing, the code should fail closed.

## Recommended Local Rollout

1. Copy `quant/main/config/live_trading_policy.example.json` into a private
   local policy file and keep `enabled=false`.
2. Copy `quant/main/config/live_arm_state.example.json` into a private local arm
   file and keep `enabled=false`.
3. Run `shadow` mode against formal latest assets, read-only holdings, and
   realtime or simulated market snapshots.
4. Run `paper` mode to validate:
   - JSONL event ledger
   - SQLite batch/order/ack persistence
   - idempotency behavior
   - reconciliation outputs
   - kill switch and live arm failure paths
5. Only after audit and explicit owner approval should `live` be armed for a
   short window on an isolated execution host.

## Design Intent

This gateway is here to make future real execution safer and more inspectable,
not to make it easier to bypass governance.

If the system cannot prove legality, auditability, reproducibility, and bounded
execution, it should stop before submission rather than force a trade through.
