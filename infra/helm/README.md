# Helm charts

PROJECT_PLAN.md section 10: "one Dockerfile and one Helm chart each" for the nine
Phase 4 services.

## Status

Every chart (`helm lint`) and every rendered template (`helm template`) has been
verified for real against Helm 3 installed on this machine. Beyond that,
**Phase 8 actually deployed `risk-engine`'s chart to a live `kind` cluster** and
ran the HPA/PodDisruptionBudget/NetworkPolicy/chaos-kill verification
PROJECT_PLAN.md section 14 and deliverables 9/11 call for -- see
`infra/k8s/README.md` for the real, observed results (HPA scaling 1->5 replicas
under real load, a killed pod transparently replaced, an unlabeled pod blocked
by NetworkPolicy and a labelled one allowed). All 9 charts carry the identical
HPA(where enabled)/PDB/NetworkPolicy wiring verified there; only `risk-engine`
was the one actually installed into a live cluster in this pass (see
`infra/k8s/README.md`'s "What's NOT verified" for the honest boundary).

```bash
helm lint infra/helm/risk-engine
helm template infra/helm/risk-engine   # renders real Deployment + Service + HPA + PDB + NetworkPolicy YAML
```

## What's in each chart

- `Chart.yaml`, `values.yaml`
- `templates/deployment.yaml` -- liveness/readiness probes against the
  service's own `/health` endpoint (every one of the nine genuinely
  implements it), and a `project: capstone-rpm` pod label every
  `templates/networkpolicy.yaml` and `infra/k8s/base/*.yaml` selector matches on.
- `templates/service.yaml` -- a ClusterIP Service.
- `templates/pdb.yaml` -- a PodDisruptionBudget (`podDisruptionBudget.enabled`,
  default `true`; `minAvailable: 0` at `replicaCount: 1`, since requiring 1-of-1
  available would block voluntary disruptions entirely on a single replica).
- `templates/networkpolicy.yaml` -- ingress-only allow rule for this service's
  own port, from other `project: capstone-rpm` pods (`networkPolicy.enabled`,
  default `true`). Only meaningful alongside `infra/k8s/base/default-deny.yaml`,
  which every one of these charts assumes is already applied.
- `risk-engine` and `stream-processor` only: `templates/hpa.yaml`, a
  HorizontalPodAutoscaler (`autoscaling.enabled`, default `true`, 1-5 replicas
  targeting 50% CPU) -- PROJECT_PLAN.md section 14's named HPA targets.

## Deliberately not here yet

- TLS/mTLS at the service level -- `infra/k8s/tls/cluster-issuer.yaml` verified
  a real cert-manager `Certificate` issuance for `risk-engine`, but wiring TLS
  termination into every chart's own Service/Ingress is future work. mTLS
  between services is explicitly descoped per PROJECT_PLAN.md section 16's own
  descope order -- see `infra/k8s/README.md`'s "mTLS" section for why.
- **A top-level umbrella chart** wiring all nine together with real
  inter-service `env` values -- each chart's `values.yaml` still has an
  `env: {}` placeholder for this (e.g. `clinician-api`'s `RISK_ENGINE_URL`).
  `infra/compose/docker-compose.yml` already has the correct, complete set of
  these values per service; an umbrella chart would mirror it for the
  Kubernetes path.
