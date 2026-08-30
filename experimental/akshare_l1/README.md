# AKShare experimental L1 gap fill

This adapter is limited to experimental/test L1 assets. It rejects any output
outside `quant/data_file/experimental_assets` and never changes the production
route, registry, or downstream layers.

The pinned environment is intentionally separate from the project virtual
environment because current AKShare requires pandas 2.x.

```powershell
python akshare_l1_experimental.py --as-of 20260713
```

Generated assets are samples for source and schema validation. They are marked
`not_approved_for_production` and require user approval plus audit before any
production promotion.
