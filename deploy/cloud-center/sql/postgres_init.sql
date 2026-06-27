CREATE SCHEMA IF NOT EXISTS registry;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS registry.assets (
    asset_id text PRIMARY KEY,
    asset_name text NOT NULL,
    layer text NOT NULL,
    track text NOT NULL CHECK (track IN ('production', 'experimental', 'legacy')),
    owner_agent text NOT NULL,
    storage_type text NOT NULL CHECK (storage_type IN ('clickhouse', 'object_storage', 'postgres', 'external')),
    storage_uri text NOT NULL,
    manifest_uri text,
    database_name text,
    table_name text,
    version text NOT NULL,
    trade_date text,
    status text NOT NULL,
    audit_record_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS registry.manifests (
    manifest_id text PRIMARY KEY,
    layer text NOT NULL,
    track text NOT NULL CHECK (track IN ('production', 'experimental', 'legacy')),
    owner_agent text NOT NULL,
    manifest_uri text NOT NULL,
    approved_for_mainline boolean NOT NULL DEFAULT false,
    approved_for_l5 boolean NOT NULL DEFAULT false,
    audit_record_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS registry.lineage (
    lineage_id bigserial PRIMARY KEY,
    output_asset_id text NOT NULL,
    input_asset_id text NOT NULL,
    relation_type text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit.records (
    audit_record_id text PRIMARY KEY,
    request_agent text NOT NULL,
    audit_agent text NOT NULL DEFAULT 'audit-agent',
    layer text NOT NULL,
    asset_id text,
    conclusion text NOT NULL,
    risk_level text NOT NULL,
    evidence_uri text NOT NULL,
    approved_for_production boolean NOT NULL DEFAULT false,
    approved_for_downstream boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ops.workflow_tasks (
    task_id text PRIMARY KEY,
    workflow_name text NOT NULL,
    owner_agent text NOT NULL,
    status text NOT NULL,
    started_at timestamptz,
    eta_minutes integer,
    completed_at timestamptz,
    blocker text,
    audit_required boolean NOT NULL DEFAULT false,
    audit_closed boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_assets_layer_track ON registry.assets(layer, track);
CREATE INDEX IF NOT EXISTS idx_assets_owner ON registry.assets(owner_agent);
CREATE INDEX IF NOT EXISTS idx_audit_records_asset ON audit.records(asset_id);
CREATE INDEX IF NOT EXISTS idx_workflow_tasks_status ON ops.workflow_tasks(status);
