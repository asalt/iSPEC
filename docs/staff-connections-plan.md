# Next slice: staff connection and mount inventory

Requested after relay deduplication: track other computers, connection details,
and expected mount points for internal staff viewing. Inventory is distinct
from credentials, current mount health, and permission to remount.

## Read-only starting observations (2026-09-30)

The current process mount table contains:

| Local mount point | Source | Transport |
| --- | --- | --- |
| `/home/alex/amms06/mnt/d` | `amms06@10.16.1.22:/mnt/d/` | SSHFS |
| `/home/alex/amms06/mnt/e` | `amms06@10.16.1.22:/mnt/e/` | SSHFS |

`/home/alex/windows` exists as a directory, but this process's mount table has
no mounts registered beneath it. This does not establish the intended share
list, host health, or whether another mount namespace has additional mounts.
No recursive network-directory traversal, host probes, or remounts were done.
Mount registration alone does not prove a remote filesystem is responsive.

## Recovered connection inventory (2026-10-01)

An initial local inventory now lives in the deployment repository at
`configs/staff-connections.local.json` (gitignored). The tracked
`configs/staff-connections.example.json` documents its small bootstrap format.
It contains six hosts and eight expected mount locations, plus dated observations
and the source of remembered usernames/options. No passwords are stored.
This is a manual bootstrap inventory for the future staff UI/database, not an
active config loader, mount executor, or a second permanent source of truth.

| Host | Address now observed | Mounts remembered | Observation |
| --- | --- | --- | --- |
| AMMS01 | `10.16.1.19` | `d` via GIO | DNS + NetBIOS identify host; TCP 445 open; old symlink still points at `.32` |
| PROTEOMICS_CORE | `10.16.1.31` | `dshare` via GIO, `G` via CIFS | NetBIOS identifies host; TCP 445 open; neither share mounted in this session |
| TIMS-ULTRA-0006 | `10.16.1.63` | `dshare` via GIO | Mounted; `gio info` returned a directory |
| THERMO-88FG206I / Eclipse | `10.16.1.3` | `d` via GIO | NetBIOS identifies host; TCP 445 open; saved link but no active mount |
| MSPC2_PD | `10.16.1.47` | `E` via CIFS | Name lookup + NetBIOS identify host; TCP 445 open; fish helper still uses `.15` |
| AMMS06 | `10.16.1.22` | `/mnt/d/`, `/mnt/e/` via SSHFS | Both registered; SSH port open; remote contents not traversed |

AMMS01's old `.32:445` timed out. Its saved symlink records account `shiro`,
domain `AMMS01`; the user's new `smb://amms01@...` attempt supplies account
`amms01`, not a host alias. The correct current account and the failed mount's
exact error remain to be confirmed. No login guessing, password reads, subnet
sweep, remount, symlink replacement, or helper modification was performed.
NetBIOS names/MACs are discovery hints, not authenticated host identities.

Alex subsequently reported `mount error(22): invalid argument`, citing
`mount.cifs`. The kernel journal at `2026-10-01T14:09:29-05:00` records
`CIFS: VFS: bad security option: ntlmsspq`. The installed `mount.cifs(8)` manual
lists `ntlmssp`, not `ntlmsspq`. This identifies a real invalid option in a
recent CIFS attempt; the exact attempted command/host is still needed to tie it
definitively to the reported failure. It is not evidence of an IP or password
failure, and does not by itself diagnose the separate GIO attempt.

Important distinctions for the implementation:

- A `~/windows/...` path may be a symlink into a per-user GVFS mount, not a
  kernel mount at that path. Combine GIO's mount listing, symlink targets and
  the kernel mount table. The earlier kernel-only observation was incomplete.
- A plain empty directory at a CIFS mount point is not an available remote
  share. Future file checks must confirm the expected mount first, rather
  than silently treating local fallback contents as remote data.
- GVFS observations belong to the Linux user/session. A service running under
  another user or without that desktop session may not see the same mounts.
- SMB login account, authentication domain, server name/address, remote share,
  and stable local path are separate fields. Keep GIO and CIFS recipes distinct.
- Use the invoking user's primary group (`id -g`) for the CIFS `gid` option,
  not their user ID (`id -u`). Remembered protocol options are not proof of
  successful current negotiation.
- For GIO mounts, discover the actual resulting GVFS path before updating a
  local symlink. Do not synthesize its attribute ordering or assume it stays
  identical when the address/account/domain changes.

## Small proposed implementation

1. Use the existing app database for a staff-managed host record and associated
   mount records, not another database or a legacy-imported scientific table.
   A host needs a display name and address/SSH alias. A mount needs host link,
   transport, remote path/share, local path, and whether it is expected mounted.
2. Keep passwords, tokens and private keys out of these records. Store an
   optional reference to externally configured credentials, not the secret.
3. Add a staff/admin-only Operations panel and equally protected backend API.
   Client accounts must not retrieve internal paths or host details. Reuse the
   role checks already used for staff schema operations in `api/routes/ops.py`.
4. Display configured expectations beside timestamped observations. Derive
   labels such as registered, not registered, and not checked; do not persist
   a separate lifecycle enum. Initial observation can read the local mount
   table plus GIO's session mount listing without traversing the remote filesystem.
5. Add optional on-demand bounded reachability checks later. Remounting remains
   a distinct explicit operation, not a reaction to an absent/stale observation.

Tests should cover staff/client authorization, host-to-mount relationships,
mount-table parsing with escaped paths, absent mounts, and the distinction
between registered and verified healthy. A first listing must not contact every
host, block on an unresponsive share, expose secrets, or mount/unmount anything.

Status: bootstrap inventory and diagnostics only. No inventory tables,
endpoints, automatic checks, or UI added yet.

## Local fish helpers (2026-10-01)

At Alex's request, local helpers under `~/.config/fish/functions/` were updated:

| Function | Behavior |
| --- | --- |
| `mount_mspc2_e` | CIFS `//10.16.1.47/E`; SMB 2.1, correct UID/GID; skips the existing expected mount |
| `mspc2_pd_mount` | Compatibility alias for `mount_mspc2_e` |
| `mount_proteomics_g` | CIFS `//10.16.1.31/G`; remembered username/domain; explicit `--username` and `--domain` overrides |
| `mount_amms01_d` | GIO `smb://shiro@10.16.1.19/d/`; optional `--username`; discovers the actual GVFS path after success before updating only the symlink |
| `amms06_mount` | Existing SSHFS helper, unchanged; `--status` reports both registered mounts |

The three new canonical helpers accept `--dry-run` and `--help`. CIFS helpers
refuse to mount over a different filesystem or a symlink. AMMS01 never replaces
a real file/directory and leaves the old symlink intact on authentication or
path-discovery failure. No passwords are stored in functions or this inventory.
These are local convenience helpers, not staff API actions or supervisor jobs.

The old `mspc2_pd_mount.fish` defined `mount_mspc2_e` under a nonmatching file
name. Canonical names now match fish autoload filenames. For an existing shell
that already loaded an older definition, source the new function once, or open
a new shell, before invoking it.

Alex successfully mounted MSPC2_PD at `.47`; kernel mount options confirm
SMB 2.1, `sec=ntlmssp`, UID 1000 and GID 100. The helper was run and returned
"Already mounted", with no extra mount or login attempt.

PROTEOMICS_CORE's later error 13 is confirmed by the kernel journal as
`STATUS_LOGON_FAILURE` during `SessSetup`. This is an authentication failure,
not evidence that the address changed or proof of share-level ACL denial.
Verify the actual Windows login account/domain (for example, `whoami` on that
computer), rather than repeatedly guessing credentials. The helper cannot fix
that by changing the SMB version or Linux ownership IDs.

Nine isolated fish checks passed: dry runs, account-option validation,
already-mounted behavior, refusing a conflicting filesystem, preserving an old
AMMS01 symlink after failed mock authentication, successful mock path discovery,
and refusing to replace a real directory. PROTEOMICS_CORE/AMMS01 helpers were
only dry-run; their live authentication remains a user-terminal step.
