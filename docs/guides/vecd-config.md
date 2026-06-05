# vecd.conf reference

`vecd` reads operator settings from `vecd.conf` in its config directory. Every
setting also has a `vecd serve` / `start` / `restart` flag; **a flag overrides
the file, which overrides the built-in default.** Settings an operator
shouldn't type on a command line every restart (TLS paths, secrets, rate
limits) belong in the file.

## You must configure vecd before running it

vecd refuses to run against an unconfigured base path: **every command except
`config` and `completions` requires a `vecd.conf`**, and errors with a pointer
to `vecd config auto` if none is found. (`vecd init --config-import <path>` is
the single-shot exception — it bootstraps the config and the DB together.)

## Where vecd.conf lives

Resolved in this order:

1. **`--conf <dir>`** — explicit config dir; overrides `$VECD_CONFIG` (and warns
   if both are set).
2. **`$VECD_CONFIG`** — a config dir. Relocates *all* server state (the DB and
   `local:` objects live under `data/` there).
3. **current dir vs. home** — `./vecd.conf` or `~/.config/vecd/vecd.conf`. If a
   config exists in exactly one, it's used; if in **both**, disambiguate with
   `--config-is-local` / `--config-is-home`, or `PREFER_CONFIG=local|home`
   (for scripts/CI).

The file is created `0600`. Format: one `key = value` per line; `#` begins a
comment; surrounding whitespace and a wrapping `"…"` are trimmed.

## Managing the config

Three commands — `auto` (bootstrap), `get` (read), `set` (write):

| Command | What it does |
|---------|--------------|
| `vecd config auto [--yes]` | Write safe defaults (local-only `bind` + `data_dir`) and confirm. |
| `vecd config get` | Print the whole config (native `key = value`) to stdout. |
| `vecd config get <key>` | Print one value (raw, for scripting). |
| `vecd config get --format json\|yaml` | Print the whole config as JSON/YAML to stdout. |
| `vecd config get --out <path>` | Write the whole config to a `.json`/`.yaml` file, a directory, or `-`. |
| `vecd config set <key> <value>` | Set one value (`--force` to change an already-set one). |
| `vecd config set --from <path>` | Replace the whole config from a `.json`/`.yaml`/`.conf` file, a directory, or `-` (stdin). `--force` to overwrite. |

Round-trip and convert freely, e.g. `vecd config get --format json` →
`vecd config set --from x.json --force`, or `vecd config get | vecd config set --from -`.

Changing an established config requires `--force`. Setting **`lock_config = on`**
freezes the whole config — every change is refused, *even with `--force`*, until
you unlock it (`vecd config set lock_config off --force`).

```ini
# ~/.config/vecd/vecd.conf
bind     = 0.0.0.0:8443
tls_cert = /etc/vecd/cert.pem
tls_key  = /etc/vecd/key.pem
db_backup = s3://vecd-private/backups/
ratelimit_client_download = 200MiB
```

## Keys

| Key | CLI flag | Default | Meaning |
|-----|----------|---------|---------|
| `data_dir` | `--data-dir` | `<config dir>/data` | DB, pidfile, `vecd.addr`, `vecd.log`, and `local:` backend objects. |
| `bind` | `--bind` | `127.0.0.1:8443` | Listen address. Loopback = local-only (safe default). A non-loopback address exposes vecd on the network — pair it with TLS. |
| `tls_cert` | `--tls-cert` | _(none)_ | PEM certificate. With `tls_key`, vecd serves HTTPS. |
| `tls_key` | `--tls-key` | _(none)_ | PEM private key. Must be given together with `tls_cert`. |
| `db_key` | _(env `VECD_DB_KEY`)_ | _(none)_ | SQLCipher key for an encrypted control-plane DB. Requires a build with `--features sqlcipher`. No CLI flag — secrets don't belong in argv. |
| `db_backup` | `--db-backup` | _(none)_ | Destination for scheduled control-plane DB snapshots: a filesystem path or `s3://bucket/prefix`. Unset = no scheduled backups. |
| `ratelimit_connection_download` | `--ratelimit-connection-download` | `0` (off) | Per-connection (`IP:port`) download cap, bytes/sec. |
| `ratelimit_connection_upload` | `--ratelimit-connection-upload` | `0` (off) | Per-connection upload cap. |
| `ratelimit_client_download` | `--ratelimit-client-download` | `0` (off) | Per-client (`IP`) download cap — shared across all of one host's connections. |
| `ratelimit_client_upload` | `--ratelimit-client-upload` | `0` (off) | Per-client upload cap. |
| `lock_config` | _(none)_ | `off` | When `on`, freezes the config against any change until set back to `off`. Config-management only — not a server setting. |

**Byte-rate values** (the `ratelimit_*` keys) accept a bare integer
(bytes/sec) or a suffix: binary `KiB`/`MiB`/`GiB` (×1024ⁿ) or decimal
`KB`/`MB`/`GB` and bare `K`/`M`/`G` (×1000ⁿ). `0` (any unit) means unlimited.
See the [rate-limits tutorial](../tutorials/vecd-rate-limits/) for what the
per-connection vs per-client distinction does.

**Backup scheduling** has two more knobs that are CLI-only (they don't read
from `vecd.conf`): `--backup-interval` (default `1h`, min `60s`) and
`--backup-retain` (default `24` snapshots). An `s3://` `db_backup` destination
requires the **`aws` CLI** on `PATH` (vecd shells out to it for the transfer);
a filesystem path has no external dependency. A missing `aws` is reported with
an actionable error rather than a cryptic spawn failure.

## Security notes

- The default `bind` is **loopback** — a bare `vecd start` is reachable only
  from the local host. Binding a non-loopback address **without** TLS serves
  bearer tokens in cleartext and prints a startup warning; configure
  `tls_cert`/`tls_key`, or terminate TLS at a reverse proxy in front of a
  loopback-bound vecd.
- `vecd.conf` is created `0600`; keep it that way if it holds `db_key`.

## See also

- [vecd intro & quickstart](./vecd-intro.md)
- [Deploying vecd](../../deploy/vecd/README.md) (Docker, systemd, TLS)
- [Daemon design](../design/vecd-daemon.md)
