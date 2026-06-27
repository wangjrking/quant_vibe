# Windows Host Assessment for Cloud Center Deployment

Date: 2026-06-27

This note records the current local machine state for adapting the
`deploy/cloud-center/` package to run on this Windows computer as the target
server.

## Summary

The current machine is a Windows desktop intended to act as the quant cloud
center host. The existing deployment package is Linux-oriented and assumes
`/opt/quant_mcp`, Docker Compose, and Linux shell scripts. The machine is not
yet ready to start the services directly because Docker and a usable WSL Linux
distribution are not installed or not available in the current shell.

Recommended adaptation path:

1. Prefer WSL2 Ubuntu plus Docker Desktop with WSL integration.
2. Keep the canonical Linux deployment path `/opt/quant_mcp` inside WSL.
3. Avoid rewriting the production compose package for native Windows unless
   WSL2 cannot be used.
4. Do not start ClickHouse, PostgreSQL, MinIO, or MCP services until the
   environment variables and runtime layout are reviewed.

## Machine Profile

Observed from PowerShell:

```text
OS: Windows 10 Pro
Windows version: 2009
OS architecture: 64-bit
System type: x64-based PC
CPU: AMD Ryzen 5 9600X 6-Core Processor
Memory: 33,460,539,392 bytes, about 31.2 GiB
```

Disk state:

```text
C:\ used about 109.7 GB, free about 103.4 GB
D:\ used about 195.5 GB, free about 589.5 GB
```

The D drive has enough free space for an initial local server deployment, but
the production design still recommends a larger dedicated data disk if the
warehouse grows to full L1-L7 history.

## Current Deployment Workspace

The Windows-side preparation directory has been created:

```text
D:\opt\quant_mcp\
  backups\
  env\
    warehouse.env
  repo\
  runtime\
  volumes\
    clickhouse\
    mcp_logs\
    minio\
    postgres\
```

Repository state in `D:\opt\quant_mcp\repo`:

```text
remote: https://github.com/wangjrking/quant_vibe.git
branch: cloud-center-mcp
head: abae9e62f2302d964cd7fa20c34e9fb4ecb549b9
status: clean before this assessment file
```

`D:\opt\quant_mcp\env\warehouse.env` was copied from
`deploy/cloud-center/env.example`. It is outside the git repo and should stay
outside git. It still contains placeholder credentials unless the operator has
edited it locally.

## Tooling State

Git:

```text
The ordinary git command is not available in PATH.
Codex bundled git is available:
C:\Users\25747\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe
git version 2.53.0.windows.3
```

Python:

```text
python resolves to Microsoft Store alias:
C:\Users\25747\AppData\Local\Microsoft\WindowsApps\python.exe
```

No project Python runtime has been confirmed for this deployment workspace.

Docker:

```text
docker command: not found
docker-compose command: not found
Docker services: no matching Docker service observed
```

WSL:

```text
wsl.exe exists at C:\WINDOWS\system32\wsl.exe
wsl -l -v reports that no Linux distribution is installed/available.
```

Windows optional feature state could not be read from the current shell because
`Get-WindowsOptionalFeature -Online` requires elevation.

Package installer:

```text
winget exists:
C:\Users\25747\AppData\Local\Microsoft\WindowsApps\winget.exe
```

## Port Availability

The ports required by the current compose package are free in the current
Windows session:

```text
5432  PostgreSQL        FREE
8123  ClickHouse HTTP   FREE
9000  ClickHouse native FREE
9100  MinIO API         FREE
9101  MinIO console     FREE
```

## Current Compose Compatibility Issue

The current compose file uses Linux absolute paths:

```yaml
volumes:
  - /opt/quant_mcp/volumes/clickhouse:/var/lib/clickhouse
  - /opt/quant_mcp/volumes/postgres:/var/lib/postgresql/data
  - /opt/quant_mcp/volumes/minio:/data
```

This is correct for Ubuntu/Linux or WSL2. It is not directly suitable for native
Windows PowerShell execution unless the compose file is adapted to Windows
paths or parameterized volume roots.

The health check shell script is also Linux-oriented:

```text
deploy/cloud-center/scripts/health_check.sh
```

A PowerShell script exists:

```text
deploy/cloud-center/scripts/health_check.ps1
```

but the core service startup still depends on Docker availability.

## Recommended Adaptation Options

### Option A: WSL2 Ubuntu plus Docker Desktop

This is the recommended path.

Expected final layout inside WSL:

```text
/opt/quant_mcp/
  repo/
  env/warehouse.env
  volumes/clickhouse/
  volumes/postgres/
  volumes/minio/
  volumes/mcp_logs/
  backups/
  runtime/
```

Benefits:

- Matches the existing deployment documentation.
- Keeps Linux paths and shell scripts valid.
- Avoids fragile Windows path mounting changes in compose.
- Makes future migration to a real Linux server straightforward.

Required preparation:

1. Install or enable WSL2.
2. Install Ubuntu for WSL.
3. Install Docker Desktop.
4. Enable Docker Desktop WSL integration for the Ubuntu distribution.
5. Recreate or copy the repo into `/opt/quant_mcp/repo` inside WSL.
6. Fill `/opt/quant_mcp/env/warehouse.env` locally.

### Option B: Native Windows Docker Desktop

This is possible but less preferred.

Required changes:

- Parameterize host volume root in `docker-compose.yml`, for example with
  `QUANT_MCP_ROOT`.
- Use Windows-compatible paths such as `D:/opt/quant_mcp/volumes/clickhouse`.
- Confirm ClickHouse, PostgreSQL, and MinIO file permissions on Windows bind
  mounts.
- Use PowerShell health checks by default.

This route should be used only if WSL2 is blocked.

## Suggested Next Design Task

Produce an adaptation plan for this host that answers:

1. Whether to require WSL2 or support native Windows Docker.
2. Whether `deploy/cloud-center/docker-compose.yml` should stay Linux-only or
   gain a parameterized volume root.
3. Exact install commands for WSL2, Ubuntu, and Docker Desktop on Windows 10
   Pro.
4. Exact migration commands from the current Windows-side
   `D:\opt\quant_mcp` preparation directory into WSL `/opt/quant_mcp`, if WSL2
   is chosen.
5. A preflight script that checks Docker, WSL, ports, required directories, and
   placeholder passwords before any service startup.

## Safety Boundaries Already Observed

- Services were not started.
- `docker compose up -d` was not run.
- No production data was migrated.
- No mainline data entrypoint was changed.
- No secrets were committed.
