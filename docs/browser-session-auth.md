# Browser Sessions and Internal Service Keys

Browser login uses `/api/auth/login`, which issues an HttpOnly session cookie.
The cookie is sufficient for session-protected routes; a browser must never need
the shared `ISPEC_API_KEY`. Existing roles, forced password changes, disabled
accounts, and explicit project grants still determine access.

## Authentication Boundaries

| Caller | Credential | Scope |
| --- | --- | --- |
| Browser | User session cookie | That user's role and project grants |
| Slack/local assistant service | Explicit API key | Existing assistant service flags |
| Internal agent/receipt API | Explicit API key | Existing internal route controls |
| First-admin HTTP bootstrap | Configured API key | Only while there are no users |

A valid cookie takes precedence over an accompanying service key on user routes.
An invalid/expired/disabled cookie fails rather than falling back to a service
identity. Service callers should not carry browser cookies. A key alone cannot
bypass required login on core CRUD routes. Use the auth CLI for initial account
setup; the login page no longer offers anonymous admin bootstrap.

`ISPEC_REQUIRE_LOGIN=0` retains legacy key-gated anonymous development access
only when no cookie is supplied. Never use that configuration in production.
`ISPEC_API_REQUIRE_KEY=1` fails closed for service routes if the key is missing;
cookie login continues to work independently.

## Proxy and Production Requirements

- Set `ISPEC_REQUIRE_LOGIN=1`, `ISPEC_API_REQUIRE_KEY=1`, and
  `ISPEC_SESSION_COOKIE_SECURE=1`; use `ISPEC_SESSION_COOKIE_SAMESITE=lax`.
- Keep bootstrap/reset disabled for ordinary restarts. Preserve the password
  pepper when moving the authoritative database to amms06.
- Serve the frontend and `/api` on the same public origin. The Vite development
  proxy preserves Host, forwards cookies, and strips `X-API-Key`/`Authorization`.
  It rejects `VITE_API_KEY` in frontend configuration. Vite is not the production
  server; build static assets for nginx.
- Configure nginx equivalently: no shared key injection, clear incoming service
  credential headers on browser routes, preserve Host, and pass the verified
  external HTTPS scheme. Do not trust arbitrary client-supplied forwarded headers.
  Uvicorn must trust forwarded headers only from the local proxy.
- Keep `/api/agents/` and `/api/auth/bootstrap` private at the public proxy.
  Internal callers connect directly to the private backend, with their key.
  Model endpoints and operational databases are never public routes.
- Keep backend/model ports private. Add login rate limiting and complete the
  public-route/access review before public exposure. This change is not a full
  production security audit or deployment.

Unsafe browser requests with a foreign/malformed Origin or cross-site Fetch
Metadata are rejected. CORS allowlisting does not override this same-origin
write rule. Non-browser clients that omit those headers still need their normal
authentication. Cross-origin cookie deployments are not supported by this slice.

## Activation and Verification

Deploy/reload the backend change and frontend proxy together. An old backend
still requires the key that the new frontend no longer injects. No existing
passwords, project grants, sessions, service keys, or databases need migration.
Remove any legacy `VITE_API_KEY` from frontend configuration before building.

Local regression tests use temporary databases, stubbed model replies, and
loopback-only HTTP servers. They neither send Slack messages nor contact legacy.

```sh
.venv/bin/python -m pytest -q --run-testclient tests/unit/api/test_security.py tests/unit/api/test_auth_routes.py tests/integration/test_browser_session_auth.py
```

In the frontend `app/` directory, run `npm run test:proxy` and `npm run build`.
The deploy repository also tests its frontend Make target with synthetic env
files so no local credential files are loaded.
