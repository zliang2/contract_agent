# Deploying to Cloud Run + Cloud SQL

One Cloud Run service runs the API and serves the built UI. It uses PostgreSQL 16 with pgvector on Cloud SQL. The BM25 index lives in Cloud Storage, and API keys and the DB URL are kept in Secret Manager. `.github/workflows/deploy.yml` deploys every merge to `main` once the repository variables are set. Until then that job is skipped.

> **Cost:** Cloud SQL has no free tier. The instance bills for every hour it exists, even with no traffic, so delete it when you're done (see "Tear down"). With min instances = 0, Cloud Run bills only while handling requests. Artifact Registry and Cloud Storage bill for the image and the index stored.

## 1. One-time setup

Requires `gcloud` logged in as a project owner, plus a project with billing enabled.

```bash
PROJECT=my-project REGION=us-central1 GITHUB_REPO=zliang2/contract_agent deploy/setup_gcp.sh
```

The script creates:
- the Artifact Registry repo, the bucket and the Cloud SQL instance;
- the `clausewitz-database-url` and `clausewitz-gemini-api-key` secrets (it prompts for the key);
- a runtime and a deploy service account;
- Workload Identity Federation limited to this repository, so GitHub holds no service account key;
- log-based metrics on the trace line.

It then prints the GitHub repository variables to set.

## 2. Load the data

The BM25 index maps to `chunks.id`, so the database and the index must come from the same ingestion. Restoring a dump of the local database keeps the ids identical.

```bash
# Dump the local, already-ingested database.
docker compose exec -T db pg_dump -U clausewitz -Fc clausewitz > data/db.dump

# Restore into Cloud SQL through the Cloud SQL Auth Proxy (in another terminal:
#   cloud-sql-proxy --port 5433 $PROJECT:$REGION:clausewitz-db).
pg_restore -h localhost -p 5433 -U clausewitz -d clausewitz --no-owner data/db.dump

# Upload the matching BM25 index.
gcloud storage cp -r indexes/bm25 gs://$PROJECT-clausewitz/
```

## 3. Deploy

Push to `main`, or run the `deploy` workflow by hand. The workflow:
- builds the image;
- pushes it to Artifact Registry;
- deploys it with `--min-instances 0`, which is a deliberate deviation from the spec (see `docs/decisions.md`);
- then curls `/healthz` and `/contracts`.

## 4. Check tracing

Every `POST /query` writes one JSON line. Cloud Logging stores it as `jsonPayload`:

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND jsonPayload.message="query"' \
  --project $PROJECT --limit 1 --format json
```

The line carries:
- `trace_id`;
- `duration_ms` per stage (bm25, vector, fusion, rerank, generate, verify) plus the total;
- `llm_calls`, with tokens per call;
- `generation_tokens_per_s` and `cost_usd`;
- `chunks_retrieved`, `claims_generated`, `claims_dropped` and `refused`.

The log-based metrics are:
- `clausewitz_latency_ms`;
- `clausewitz_cost_usd`;
- `clausewitz_refusals`;
- `clausewitz_queries`.

For the dashboard, chart p50/p95 of the latency metric, the mean of the cost metric, and refusals ÷ queries.

## Tear down

```bash
gcloud run services delete clausewitz --region $REGION --project $PROJECT
gcloud sql instances delete clausewitz-db --project $PROJECT
```
