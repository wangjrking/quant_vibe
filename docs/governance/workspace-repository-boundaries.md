# Workspace and repository boundaries

`D:/work/quant/quant_mcp` is a workspace shell, not a Git repository. Do not
create an empty `.git` directory at the workspace root.

The current repository boundaries are:

| Path | Role |
| --- | --- |
| `quant/main` | Quant data, factor, model, strategy, execution, and governance code |
| `site` | Web application and site runtime |
| `wjr-quant-strategy-mcp` | Investment platform documentation |

`mcp_server` currently lives in the workspace shell without an independent Git
repository. Changes there must not be reported as part of `quant/main` or
`site`; establish an explicit repository boundary before treating it as a
versioned release unit.

## Rules

- Run Git commands from the owning repository, never from the workspace shell.
- Do not use one repository to ignore or track files owned by another.
- Keep production data, reports, runtime evidence, and rollback assets outside
  source-control cleanup decisions.
- Generated caches and logs may be recycled only under the approved project
  cleanup policy.
- Repository consolidation or splitting requires an explicit migration plan;
  an empty root `.git` is not a migration plan.
