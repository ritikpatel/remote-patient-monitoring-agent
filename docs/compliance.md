# Compliance mapping

> This platform is validated on a 100-patient demo subset of MIMIC-IV. Clinical
> narrative is LLM-generated from structured data. Wearable deterioration
> signals are synthetically morphed from healthy-volunteer recordings. The
> engineering is real and the methodology is rigorous; the clinical
> performance figures demonstrate pipeline validity and do not transfer to
> clinical practice.

PROJECT_PLAN.md section 14: "`docs/compliance.md` mapping each technical
control to its HIPAA/GDPR safeguard. State plainly that MIMIC is already
de-identified, so this is architecture demonstration, not a live compliance
posture."

## The one fact that governs everything below

**MIMIC-IV Demo is already de-identified** under HIPAA's Safe Harbor method
(dates shifted, ages >89 aggregated, direct identifiers removed) before
PhysioNet ever published it -- see `docs/DATA_USE.md`. Nothing this project
does operates on PHI in the regulatory sense. Every control below is a real,
working piece of engineering (each one is independently verified -- see the
"Verified" column and its linked evidence), demonstrating the *architecture* a
genuine PHI-handling deployment would need, not a system that has undergone
the actual HIPAA/GDPR compliance process (a Business Associate Agreement, a
Data Protection Impact Assessment, a security risk analysis, breach-
notification procedures, and so on -- none of which apply to, or were sought
for, de-identified demo data).

## Control -> safeguard mapping

| Technical control | HIPAA safeguard | GDPR principle | Verified |
|---|---|---|---|
| Append-only, hash-chained audit log (`services/common/audit.py`, `audit_postgres.py`) recording every PHI read, agent decision, and alert acknowledgement | §164.312(b) Audit controls | Art. 5(2) Accountability; Art. 30 Records of processing | Yes -- `verify_chain()` detects tampering at the exact tampered row, tested against both SQLite and a real dockerized Postgres (`infra/compose/README.md`) |
| SMART-on-FHIR scope enforcement on every clinician-api route (`services/common/auth.py`) | §164.312(a)(1) Access control; §164.308(a)(4) Information access management | Art. 5(1)(c) Data minimisation | Yes -- unit-tested against local test tokens and a real Keycloak-issued RS256 token (`infra/compose/README.md`) |
| Keycloak as the real OIDC identity provider issuing those tokens | §164.312(d) Person or entity authentication | Art. 32(1)(b) Confidentiality of processing systems | Yes -- real realm import, real token issuance, real JWKS verification (`infra/compose/README.md`) |
| NetworkPolicy denying all ingress/egress by default, with explicit per-service allow rules (`infra/k8s/base/`, each chart's `templates/networkpolicy.yaml`) | §164.312(e)(1) Transmission security | Art. 32(1)(b) Confidentiality; Art. 25 Data protection by design | Yes -- an unlabeled pod blocked, a labelled one allowed, against a real Calico-enforced cluster (`infra/k8s/README.md`) |
| TLS via cert-manager (`infra/k8s/tls/`) | §164.312(e)(1) Transmission security (encryption in transit) | Art. 32(1)(a) Encryption | Partially -- a real certificate was issued and verified for one service; not wired into every chart's Ingress yet (`infra/k8s/README.md`) |
| mTLS between services | §164.312(e)(1) Transmission security | Art. 32(1)(a) Encryption | **Not implemented** -- explicitly descoped per PROJECT_PLAN.md section 16's own descope order; see `infra/k8s/README.md`'s "mTLS" section for the resource-constraint reasoning |
| Sealed Secrets for credential material committed to git (`infra/k8s/sealed-secrets/`) | §164.312(a)(2)(iv) Encryption and decryption (of secrets/config, not PHI itself, but the same principle) | Art. 32(1)(a) Encryption | Yes -- a real secret sealed with `kubeseal` against a live controller's public key, applied, and decrypted correctly in-cluster (`infra/k8s/README.md`) |
| Encryption at rest on the Postgres PVC | §164.312(a)(2)(iv) Encryption and decryption | Art. 32(1)(a) Encryption | **Not independently verifiable in a local `kind` cluster** -- `kind`'s default `local-path` StorageClass has no encryption-at-rest support; this is a cloud StorageClass/CSI-driver concern (e.g. an encrypted EBS-backed `StorageClass`) that only a real cloud deployment can demonstrate. Column-level `pgcrypto` on the most sensitive audit fields would be the local-equivalent control but was not built in this pass -- documented as a gap, not silently assumed |
| API-key auth on ingest-gateway's REST path (`DEFAULT_API_KEY`) | §164.312(d) Person or entity authentication | Art. 32(1)(b) Confidentiality | Partially -- real enforcement, but the key is a module constant, not sourced from a secret store; `app.py`'s own docstring flags this |
| Quality flags on every Observation (`imputed`, `device_fault`, `synthetic`, ...) distinguishing measured from derived/synthetic data | N/A (not a HIPAA safeguard) | Art. 5(1)(d) Accuracy | Yes -- schema-level, tested since Phase 2 |
| Honest-reporting statement on every artefact (PROJECT_PLAN.md section 17) | N/A | Art. 5(1)(a) Lawfulness, fairness and transparency (in spirit -- this is a research/demo transparency practice, not a GDPR legal basis) | Yes -- present in this document's own opening, `eval/report.py`'s `HONEST_REPORTING_NOTICE`, and every phase's README |

## What "architecture demonstration, not a live compliance posture" means concretely

A real deployment handling actual PHI would additionally need, none of which
this project has or claims:

- A signed Business Associate Agreement with every subprocessor (cloud
  provider, LLM API vendor)
- A formal HIPAA Security Risk Analysis and a GDPR Data Protection Impact
  Assessment
- Breach notification procedures and a named Privacy/Security Officer
- BAA-covered LLM inference (this project's own `notes_synth/README.md` and
  `eval/README.md` already note that Groq, the LLM backend actually used, is
  not what PROJECT_PLAN.md's target backend -- claude-sonnet-5 -- would be in
  a real deployment, and that this substitution has real cost/behaviour
  consequences beyond compliance)
- A BAA with an SMS carrier for real paging. `services/common/sms.py` can
  place a real call to Twilio's API, attempted by `notification-gateway` on
  every high-severity alert notification -- it is a demo shortcut stated as
  one in its own module docstring, guarded to fail closed (dry run unless
  `SMS_MODE=live`, and every message body forced to carry
  `[SYNTHETIC DRILL]` so nothing that could read as a real clinical alert can
  leave this module). `notification-gateway` is the *only* caller: it used
  to also be triggered a second, independent time by `stream-processor`'s
  `EscalationLoop`, which meant every real alert silently double-notified
  and would have double-paged a phone once SMS was wired in -- fixed by
  making alert-service the one place a new alert calls notification-gateway
  (`services/alert-service/app.py`'s `_notify_dashboard`), and having
  `EscalationLoop` read that result back rather than requesting a second one
- Real ACME-issued TLS certificates from a publicly trusted CA (this project's
  are self-signed -- see above)
- Retention and deletion policies, and a real access-request/right-to-erasure
  process (GDPR Art. 15/17) -- meaningless against already-de-identified demo
  data, but mandatory before this architecture could touch real patient data
