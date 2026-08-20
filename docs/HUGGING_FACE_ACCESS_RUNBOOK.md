# AutoDL Hugging Face access runbook

This runbook is mandatory before downloading the gated LTX-2.5 BF16 pack from
AutoDL. It prevents confusing a network outage with a repository-gate denial.

## Established result

On the current AutoDL instance, direct access timed out with `LARA-MODEL-013`.
Applying AutoDL's official academic-resource accelerator with
`source /etc/network_turbo` reached the official Hugging Face endpoint. The
first authenticated request returned `LARA-MODEL-001` / HTTP 403 because the
account had not completed the browser-side model gate. After the account owner
completed **Agree and Access**, the same pinned gated-file request followed the
official redirect and returned HTTP 200 from the model CDN.

The account gate is therefore resolved. The AutoDL data disk was subsequently
expanded to 300 GB, so it satisfies the configured 100 GiB post-download
reserve for the complete 80.02 GB DEV pack. Do not re-diagnose a later
transfer failure as a gate failure until the metadata probe is rerun with the
scoped token and accelerated route.

## Required sequence

1. In the AutoDL shell, apply the provider's documented accelerator for that
   shell only: `source /etc/network_turbo`.
2. Keep Hugging Face cache and model files on the AutoDL data disk, not the
   root filesystem. Set `HF_HOME` to a data-disk cache path before a real
   download.
3. Run `tools/models/probe_access.py` with the project config and upstream
   manifest. The probe performs only a metadata request and never downloads a
   tensor payload.
4. Interpret the result before provisioning storage or transferring files:

   | Result | Meaning | Next action |
   |---|---|---|
   | `ok: true` | Transport and gated-file authorization work. | Provision enough data-disk space, then run the verified downloader. |
   | `LARA-MODEL-001`, HTTP 401/403 | Transport works but the token/account lacks access. | The account owner must review and accept the official model terms in Hugging Face, then use a scoped read token. Re-run the probe. |
   | `LARA-MODEL-013` | The provider route cannot reach the service within the configured timeout. | Reapply `network_turbo`, verify provider egress/DNS, then re-run the probe. |

5. Only after a successful probe, provision the full capacity required by
   `configs/project.toml`: the BF16 pack plus its configured post-download
   reserve. The active instance has a verified 300 GB data disk; do not use
   the root filesystem for either weights or cache.

## Large-file transfer fallback

The provider accelerator is suitable for metadata access but may stall while
the Hugging Face CDN starts a very large transfer. Hugging Face's current
client uses `hf-xet`; set `HF_XET_HIGH_PERFORMANCE=1` before a retry. If the
CDN still does not make byte progress, AutoDL's official network guide lists
`hf-mirror.com` as a transport mirror. A resumable `aria2c` transfer can use
that endpoint only with all of the following controls:

1. Preserve the official repository ID, pinned revision and exact filename in
   the URL path; the mirror must never select a different model revision.
2. Read the existing scoped Hugging Face token only from the local credential
   store and keep it in a mode-600 temporary aria2 configuration file. Never
   put it in shell history, process output, source code or the command line.
3. Use range-resume mode and retain partial files; do not discard a verified
   completed component merely because another component stalled.
4. Compare the final byte count against `golden/manifests/upstream.json` and
   compute SHA-256 before accepting the component. The mirror is a transport
   route, not a source of authority.

## Safety rules

- Never accept model terms, create tokens, or place tokens in source code,
  manifests, shell history, logs, or Git commits on behalf of the user.
- Use the official Hugging Face repository, pinned revision and manifest as
  the source of authority. A provider-documented transport mirror is allowed
  only under the exact-path and final hash-verification controls above.
- `network_turbo` is provider-documented acceleration and can slow unrelated
  network traffic; apply it only to the shell running Hugging Face operations.
- Retain the probe JSON artifacts as the release audit trail.
