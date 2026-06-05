# Deploying vecd

Three ways to run the [vecd](../../docs/guides/vecd-intro.md) gateway, from
quickest to most production-ready. All of them keep operator state in one
directory (`$VECD_CONFIG`, default `~/.config/vecd`): the control-plane DB,
the pidfile/addr/log, and — for a `local:` backend — the object store.

## 1. Binary

Install from the workspace, or grab a static binary from a tagged release
(`*-unknown-linux-musl` runs anywhere):

```bash
cargo install --path vecd          # from a checkout
# or: download vecd from the GitHub release for your target
```

Then follow the [2-minute quickstart](../../docs/guides/vecd-intro.md#2-minute-quickstart).

## 2. Docker

```bash
# Build the image from the workspace root.
docker build -f deploy/vecd/Dockerfile -t vecd .

# One-time: initialize the DB and mint a superuser token (printed once — save it).
docker run --rm -v vecd-data:/var/lib/vecd vecd init --superuser root

# Serve (data persists in the named volume).
docker run -d --name vecd -p 8443:8443 -v vecd-data:/var/lib/vecd vecd

# Admin commands run against the same volume:
docker exec vecd vecd status
docker run --rm -v vecd-data:/var/lib/vecd vecd backends add store \
  --kind local --endpoint local:/var/lib/vecd/objects --active
```

Inside a container, binding `0.0.0.0` is expected (the container network
namespace is the boundary). Terminate TLS at your ingress / load balancer, or
mount certs and set `tls_cert` / `tls_key` in `vecd.conf`.

## 3. systemd

For a long-lived host service supervised by systemd (uses `vecd serve`, not the
self-daemonizing `vecd start`):

```bash
sudo install -m0755 target/release/vecd /usr/local/bin/vecd
sudo useradd --system --home-dir /var/lib/vecd --create-home vecd
sudo install -m0644 deploy/vecd/vecd.service /etc/systemd/system/vecd.service

# One-time init as the service user.
sudo -u vecd VECD_CONFIG=/var/lib/vecd vecd init --superuser root

# Configure network exposure + TLS in /var/lib/vecd/vecd.conf (see below).
sudo systemctl enable --now vecd
sudo systemctl status vecd
journalctl -u vecd -f          # logs
```

The unit applies basic hardening (`NoNewPrivileges`, `ProtectSystem=strict`,
`ProtectHome`, `PrivateTmp`) and only grants write access to `/var/lib/vecd`.

## Exposing it on the network (TLS)

vecd binds **loopback** by default — safe out of the box. To serve other
hosts, set a non-loopback `bind` **and** TLS in `$VECD_CONFIG/vecd.conf`:

```ini
bind     = 0.0.0.0:8443
tls_cert = /etc/vecd/cert.pem
tls_key  = /etc/vecd/key.pem
```

If you bind a public address without TLS, vecd serves bearer tokens in
cleartext and prints a startup warning — fine only when something in front
(reverse proxy, ingress, container mesh) terminates TLS for you.

## Operate

- **Health / metrics:** `GET /healthz`, `GET /metrics` (Prometheus).
- **Status / logs:** `vecd status`, `vecd log [--tail N]`, `journalctl -u vecd`.
- **Backups:** set `db_backup = s3://… | /path` in `vecd.conf` for scheduled
  control-plane snapshots, or `vecd backup now <dest>` on demand; restore with
  `vecd restore <snapshot>` (daemon stopped).
- **Upgrades:** see the [config reference](../../docs/guides/vecd-intro.md) and
  the daemon [design doc](../../docs/design/vecd-daemon.md).
