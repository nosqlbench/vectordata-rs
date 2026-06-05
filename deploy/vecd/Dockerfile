# syntax=docker/dockerfile:1
#
# Container image for the vecd dataset gateway. Build from the workspace root:
#
#   docker build -f deploy/vecd/Dockerfile -t vecd .
#
# First run (one-time): initialize the DB + mint a superuser token (printed).
#   docker run --rm -v vecd-data:/var/lib/vecd vecd init --superuser root
# Then serve:
#   docker run -d --name vecd -p 8443:8443 -v vecd-data:/var/lib/vecd vecd
#
# See deploy/vecd/README.md for namespaces, tokens, and TLS.

# ---- build ----
FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
# Just the vecd binary. rusqlite is `bundled` (self-contained SQLite) and TLS
# is rustls, so there is no system libsqlite3 / OpenSSL to install or link.
RUN cargo build --release -p vecd --bin vecd

# ---- runtime ----
FROM debian:bookworm-slim
# ca-certificates lets outbound TLS (e.g. S3 DB backups) verify peers.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home-dir /var/lib/vecd vecd \
 && mkdir -p /var/lib/vecd \
 && chown vecd:vecd /var/lib/vecd
COPY --from=build /src/target/release/vecd /usr/local/bin/vecd
USER vecd
# vecd reads all operator state from $VECD_CONFIG (its DB + objects live under
# data/ there). The volume persists it across container restarts.
ENV VECD_CONFIG=/var/lib/vecd
VOLUME /var/lib/vecd
EXPOSE 8443
# `serve` runs in the foreground as PID 1, so the container's lifecycle is
# vecd's. Inside a container the network namespace is the trust boundary, so
# binding 0.0.0.0 is expected — terminate TLS at your ingress, or mount certs
# and set tls_cert/tls_key in vecd.conf. (vecd prints a cleartext heads-up if
# a public address is served without TLS.)
ENTRYPOINT ["vecd"]
CMD ["serve", "--bind", "0.0.0.0:8443"]
