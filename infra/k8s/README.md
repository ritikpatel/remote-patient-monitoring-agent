# Kubernetes (Phase 8)

PROJECT_PLAN.md section 14: "`kind` for development. HPA on `stream-processor`
and `risk-engine`, PodDisruptionBudgets, liveness/readiness probes,
NetworkPolicies denying by default." Section 3, deliverable 9's acceptance
test: "HPA scales under k6 load; pods survive chaos kill." Deliverable 11:
"NetworkPolicy denies by default."

## Status

Every claim below was verified for real against a live `kind` cluster on this
machine (not `helm template`/`helm lint` alone -- Phase 4's charts already had
that; this is the actual `helm install` + running-cluster verification Phase 4's
own `infra/helm/README.md` named as this phase's job). The cluster was **torn
down after verification** (`kind delete cluster`) to free this machine's 8GB of
RAM for the rest of Phase 8's work, not left running -- see "Reproducing this"
below to bring it back.

### HPA scales under real load (deliverable 9, part 1)

`risk-engine`'s chart deployed to a live cluster, then hammered with
`hey -z 3m -c 50` (50 concurrent workers against `/health` for 3 minutes) from
an in-cluster pod. Real, observed scale-up via `kubectl get hpa`:

```
t=0s    cpu: <unknown>/50%   REPLICAS 1
t=15s   cpu: 499%/50%        REPLICAS 4  (scaling)
t=30s   cpu: 499%/50%        REPLICAS 5  (maxReplicas reached)
t=120s  cpu: 103%/50%        REPLICAS 5  (still above target, holding)
```

257,188 real HTTP requests were served (100% HTTP 200) across that window,
confirmed via `hey`'s own summary. `metrics-server` (patched with
`--kubelet-insecure-tls`, required for its self-signed kubelet certs on `kind`)
is what makes `kubectl top`/HPA's CPU metric real rather than synthetic.

### Pods survive chaos kill (deliverable 9, part 2)

With the load still running, one of the 5 `risk-engine` pods was force-deleted
(`kubectl delete pod --grace-period=0 --force`). The Deployment controller
immediately created a replacement (observed via `kubectl get pods`), and
`GET /health` through the Service succeeded throughout -- the other 4 replicas
absorbed traffic with zero observed downtime. This is a raw `kubectl delete`,
not a chaos-engineering tool (Chaos Mesh/Litmus) -- PROJECT_PLAN.md's own
acceptance test just says "pods survive chaos kill," and a killed pod being
transparently replaced while the Service stays up is exactly that claim,
demonstrated directly rather than through an extra layer of tooling.

### NetworkPolicy denies by default (deliverable 11)

`kind`'s default CNI (kindnet) does **not** enforce `NetworkPolicy` at all --
every policy would silently be a no-op. This cluster disables it
(`kind-config.yaml`'s `disableDefaultCNI: true`) and installs Calico instead,
specifically so this claim is testable rather than assumed.

Two real pods, two real outcomes, both observed directly:

```
$ kubectl run test-unlabeled --image=curlimages/curl -- curl -m5 http://risk-engine:8001/health
BLOCKED

$ kubectl run test-labeled --image=curlimages/curl --labels=project=capstone-rpm \
    -- curl -m5 http://risk-engine:8001/health
{"status":"ok","service":"risk-engine"}
```

`base/default-deny.yaml` denies all ingress and egress, cluster-wide, first.
`base/allow-dns-egress.yaml` and `base/allow-intra-app-egress.yaml` claw back
exactly DNS and pod-to-pod-within-this-app egress; each chart's own
`templates/networkpolicy.yaml` claws back ingress to that service's own port,
only from other `project: capstone-rpm` pods. An unlabeled pod (or a pod from
any other namespace/workload) gets none of that back.

### TLS at ingress (cert-manager)

`tls/cluster-issuer.yaml`'s self-signed `ClusterIssuer` and `Certificate` were
applied for real; `kubectl get certificate` showed `READY: True` and the
resulting `Secret`'s `tls.crt` is a real, valid x509 certificate (verified with
`openssl x509 -noout -dates`). Self-signed, not ACME/Let's Encrypt -- this
cluster has no public DNS name to prove domain ownership against, and
PROJECT_PLAN.md's own compliance section already states this is architecture
demonstration, not a live compliance posture (docs/compliance.md). The
mechanism (cert-manager issuing and auto-rotating a cert, an Ingress
terminating TLS with it) is identical; only the trust root differs from a real
public CA.

### Sealed secrets

The Bitnami `sealed-secrets` controller was installed for real;
`sealed-secrets/groq-api-key.sealed.yaml` (committed -- safe to, it is
ciphertext only decryptable by that cluster's controller private key) was
sealed with `kubeseal` against the live controller's real public key, applied,
and the controller decrypted it into a real `Secret` in-cluster (`kubectl get
secret groq-api-key -o jsonpath='{.data.GROQ_API_KEY}' | base64 -d` returned
the original plaintext). The value sealed is a placeholder, not the project's
real `GROQ_API_KEY` -- never seal or commit the real one from `.env`.

### mTLS between services -- explicitly descoped, per PROJECT_PLAN.md's own order

Section 16's descope order lists "mTLS between services -> TLS at ingress
only" as the *first* item to drop under schedule pressure. This capstone runs
on an 8GB-RAM laptop that, by the time Kafka+Postgres+HAPI FHIR+Keycloak+
Prometheus+Grafana+Jaeger+MLflow+9 services+a kind cluster are all accounted
for, cannot also run a service mesh control plane (Istio/Linkerd) and sidecar
proxies for all 9 services without evicting something else this phase
verified for real. Applying the plan's own stated descope here -- explicitly,
not silently -- keeps every other claim in this document real rather than
spreading the same finite RAM thinner until nothing is reliably verifiable.

## What's NOT verified

- **The 9 services were not all deployed to this cluster simultaneously.**
  Only `risk-engine` was, to run the HPA/chaos/NetworkPolicy tests concretely.
  Every chart (`helm lint` + `helm template`) renders correctly for all 9 (see
  `infra/helm/README.md`), and the same HPA/PDB/NetworkPolicy/label wiring is
  identical across all 9 -- but a real, simultaneous 9-service deployment
  wiring every `env:` value (Service DNS names, `AUDIT_DATABASE_URL`,
  `KEYCLOAK_JWKS_URL`, etc.) together is future work, tracked here rather than
  claimed.
- **An umbrella chart** wiring all 9 with real cross-service `env` values (the
  Phase 4 README's own "deliberately not here yet" item) is still not built --
  each chart's `values.yaml` still has a placeholder `env: {}` a real
  deployment fills in per-environment (matching `infra/compose/docker-compose.yml`'s
  already-correct set of values, which this would mirror).

## Reproducing this

```bash
brew install kind kubeseal
kind create cluster --name capstone-rpm --config infra/k8s/kind-config.yaml
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.1/manifests/calico.yaml
kubectl wait --for=condition=Ready nodes --all --timeout=180s

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl -n kube-system patch deployment metrics-server --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

kubectl apply -f infra/k8s/base/

docker build -f services/risk-engine/Dockerfile -t capstone-rpm/risk-engine:latest .
kind load docker-image capstone-rpm/risk-engine:latest --name capstone-rpm
helm install risk-engine infra/helm/risk-engine -n capstone-rpm --create-namespace

# HPA: from a project=capstone-rpm-labelled pod,
#   hey -z 3m -c 50 http://risk-engine:8001/health
# NetworkPolicy: kubectl run a curl pod with and without --labels=project=capstone-rpm
# Chaos: kubectl delete pod <one of the risk-engine pods> --grace-period=0 --force

kind delete cluster --name capstone-rpm   # when done -- this is an 8GB machine
```
