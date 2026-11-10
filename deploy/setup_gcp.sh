#!/usr/bin/env bash
# One-time GCP setup for Clausewitz (docs/deploy.md). Creates billable resources:
# Cloud SQL bills for every hour the instance exists, even when idle. Review before
# running; nothing in CI runs this.
#
#   PROJECT=my-project REGION=us-central1 GITHUB_REPO=owner/repo deploy/setup_gcp.sh
set -euo pipefail
: "${PROJECT:?}" "${REGION:?}" "${GITHUB_REPO:?}"
INSTANCE=${INSTANCE:-clausewitz-db}
BUCKET=${BUCKET:-$PROJECT-clausewitz}
DEPLOY_SA=clausewitz-deploy@$PROJECT.iam.gserviceaccount.com
RUNTIME_SA=clausewitz-run@$PROJECT.iam.gserviceaccount.com
gc() { gcloud --project "$PROJECT" "$@"; }

gc services enable run.googleapis.com sqladmin.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iamcredentials.googleapis.com storage.googleapis.com

# Artifact Registry repository, bucket for the BM25 index.
gc artifacts repositories create clausewitz --repository-format docker --location "$REGION" || true
gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" \
  --uniform-bucket-level-access || true

# Cloud SQL: PostgreSQL 16, smallest shared-core tier. pgvector is created by schema.sql.
gc sql instances create "$INSTANCE" --database-version POSTGRES_16 --region "$REGION" \
  --tier db-f1-micro --edition ENTERPRISE --storage-size 10 || true
gc sql databases create clausewitz --instance "$INSTANCE" || true
DB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=')
gc sql users create clausewitz --instance "$INSTANCE" --password "$DB_PASSWORD" || \
  gc sql users set-password clausewitz --instance "$INSTANCE" --password "$DB_PASSWORD"
CONN="$PROJECT:$REGION:$INSTANCE"

# Secrets (values never echoed). Cloud Run reaches Cloud SQL over its built-in unix socket.
printf '%s' "postgresql://clausewitz:$DB_PASSWORD@/clausewitz?host=/cloudsql/$CONN" |
  gc secrets create clausewitz-database-url --data-file - 2>/dev/null ||
  printf '%s' "postgresql://clausewitz:$DB_PASSWORD@/clausewitz?host=/cloudsql/$CONN" |
  gc secrets versions add clausewitz-database-url --data-file -
if ! gc secrets describe clausewitz-gemini-api-key >/dev/null 2>&1; then
  read -rsp "Gemini API key: " KEY; echo
  printf '%s' "$KEY" | gc secrets create clausewitz-gemini-api-key --data-file -
fi

# Service accounts: runtime (reads secrets, index, Cloud SQL) and deploy (used by Actions).
gc iam service-accounts create clausewitz-run || true
gc iam service-accounts create clausewitz-deploy || true
for role in roles/cloudsql.client roles/secretmanager.secretAccessor; do
  gc projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$RUNTIME_SA" --role $role >/dev/null
done
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "serviceAccount:$RUNTIME_SA" --role roles/storage.objectViewer >/dev/null
for role in roles/run.admin roles/artifactregistry.writer; do
  gc projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$DEPLOY_SA" --role $role >/dev/null
done
gc iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member "serviceAccount:$DEPLOY_SA" --role roles/iam.serviceAccountUser >/dev/null

# Workload Identity Federation: only this GitHub repository's workflows can deploy.
gc iam workload-identity-pools create github --location global || true
gc iam workload-identity-pools providers create-oidc github --location global \
  --workload-identity-pool github --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping google.subject=assertion.sub,attribute.repository=assertion.repository \
  --attribute-condition "assertion.repository == '$GITHUB_REPO'" || true
POOL=$(gc iam workload-identity-pools describe github --location global --format 'value(name)')
gc iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/$POOL/attribute.repository/$GITHUB_REPO" >/dev/null

# Log-based metrics on the per-request trace line (clausewitz/tracing.py) for the dashboard.
FILTER='resource.type="cloud_run_revision" AND jsonPayload.message="query"'
gc logging metrics create clausewitz_latency_ms --log-filter "$FILTER" \
  --value-extractor 'EXTRACT(jsonPayload.duration_ms.total)' --metric-kind DELTA --value-type DISTRIBUTION \
  --bucket-exponential-num-buckets 40 --bucket-exponential-growth-factor 1.25 --bucket-exponential-scale 10 \
  2>/dev/null || true
gc logging metrics create clausewitz_cost_usd --log-filter "$FILTER AND jsonPayload.cost_usd>=0" \
  --value-extractor 'EXTRACT(jsonPayload.cost_usd)' --metric-kind DELTA --value-type DISTRIBUTION \
  --bucket-exponential-num-buckets 40 --bucket-exponential-growth-factor 1.5 --bucket-exponential-scale 0.00001 \
  2>/dev/null || true
gc logging metrics create clausewitz_refusals --log-filter "$FILTER AND jsonPayload.refused=true" 2>/dev/null || true
gc logging metrics create clausewitz_queries --log-filter "$FILTER" 2>/dev/null || true

PROVIDER=$(gc iam workload-identity-pools providers describe github --location global \
  --workload-identity-pool github --format 'value(name)')
cat <<OUT

Done. Set these GitHub repository variables (Settings > Secrets and variables > Variables):
  GCP_PROJECT_ID=$PROJECT
  GCP_REGION=$REGION
  GCP_WIF_PROVIDER=$PROVIDER
  GCP_DEPLOY_SA=$DEPLOY_SA
  GCP_RUNTIME_SA=$RUNTIME_SA
  CLOUD_SQL_INSTANCE=$CONN
  BM25_INDEX_URI=gs://$BUCKET/bm25
  GENERATE_MODEL=<model id>   VERIFY_MODEL=<model id>
Then load the data (docs/deploy.md, "Load the data").
OUT
