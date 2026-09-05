# Helm charts

PROJECT_PLAN.md section 10: "one Dockerfile and one Helm chart each" for the nine
Phase 4 services.

## Status

Every chart (`helm lint`) and every rendered template (`helm template`) has been
verified for real against Helm 3 installed on this machine — not just hand-written
and assumed correct. What has **not** been verified: an actual `helm install`
against a live Kubernetes cluster. This environment has `kubectl` but no `kind` or
`minikube` (no local cluster to install into) — Phase 8 is where these charts
actually get deployed and load-tested (PROJECT_PLAN.md section 14).

```bash
helm lint infra/helm/risk-engine
helm template infra/helm/risk-engine   # renders real Deployment + Service YAML
```

## What's in each chart

`Chart.yaml`, `values.yaml`, and `templates/{deployment,service,_helpers}.yaml` --
one Deployment (with liveness/readiness probes against the service's own `/health`
endpoint, which every one of the nine services genuinely implements) and one
ClusterIP-shaped Service.

## Deliberately not here yet (Phase 8 scope, PROJECT_PLAN.md section 14)

- **HorizontalPodAutoscaler** on `stream-processor` and `risk-engine`
- **PodDisruptionBudget**
- **NetworkPolicy** (deny-by-default)
- TLS/mTLS, OIDC/SMART scopes at the ingress level (services.common.auth already
  does the SMART scope *verification*; Keycloak as the actual issuer is Phase 8)
- A top-level umbrella chart wiring all nine together with real inter-service
  `env` values (each chart's `values.yaml` has an `env: {}` placeholder for exactly
  this — e.g. `clinician-api`'s `RISK_ENGINE_URL`)
