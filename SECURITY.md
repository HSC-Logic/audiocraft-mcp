# Security

Internal non-commercial testing only. Not a hardened public service. No formal security audit or production support commitment.

Keep the Compose loopback binding. Bearer tokens over HTTP are appropriate only for this local testing boundary. Remote access requires a separately reviewed TLS/authentication design; do not simply bind all interfaces or add a public tunnel.

MCP requests and artifact downloads require authentication. The minimal `/healthz` probe is intentionally unauthenticated. Worker authentication uses a distinct secret; the worker publishes no host port. Do not expose tokens in URLs, prompts, logs, issue reports, shell tracing, or committed configuration. Protect `.secrets/` host permissions; Linux setup creates a private parent directory with container-readable files. Compose secrets are mounted files, not encrypted secret storage.

Only trusted local operators may change model allowlists or cache contents. AudioCraft's legacy checkpoint loading is not a safe boundary for untrusted serialized files. Do not accept caller-provided checkpoints, URLs, or paths. Container network isolation does not replace worker authentication.

Prompts and generated audio may be sensitive. Output volumes retain job metadata and artifacts; local administrators and Docker operators can access them. Retention is not secure deletion. Do not submit personal or confidential data during testing. Review generated content before further use; no guaranteed rights or safety filtering are implied.

Report vulnerabilities privately to HSC-Logic through the repository owner's contact channel. Use GitHub private vulnerability reporting if enabled. Do not open public issues containing secrets, sensitive prompts, or exploit details before coordinated disclosure.
