# Agent discoverability check: vectordata

An optional pre-release check (SRD DX-31). Give a fresh agent session
only the packaged crate — `cargo package -p vectordata`, unpacked, or
the registry copy — and each task below, one per session. The check
passes for a task when the agent's answer uses the call listed, having
read no more of `src/` than `lib.rs`, module heads and `AGENTS.md`.

Record each run's result (pass/fail, what it read, what it wrote) in
the release notes. A failure is a documentation bug: fix the index, the
module head or the first doc line the agent followed instead, then run
the task again.

| # | Task given to the agent | Passes when it uses |
|---|---|---|
| 1 | "Fetch the base and query vectors of profile `default` of dataset `X` into the local cache, showing a progress meter like the CLI's." | `TestDataView::fetch` with `FetchRequest::facets` and `TextMeter` |
| 2 | "Before downloading, print how many bytes fetching all of profile `default` would cost." | `TestDataView::plan_fetch` → `FetchPlan::bytes_to_fetch` |
| 3 | "The user passes a dataset spec like `X:size=10m` on the command line. Open the profile it names." | `Catalog::open_spec` → `DatasetSelection::view` (not a split on `:`) |
| 4 | "Iterate over every base vector of a remote profile as fast as possible." | a `fetch` before the scan, then `VectorReader::get` (optionally `prefetch_in_background` per window) |
| 5 | "Read records 5,000,000..6,000,000 of the base vectors without downloading the rest." | `FetchRequest::window` (or `prefetch_in_background` with a window) |
| 6 | "Report a helpful message when the user names a dataset that does not exist." | `Catalog::lookup` → `Error::UnknownDataset { suggestions }` |
| 7 | "What does `vectordata datasets precache` call, so my program can do the same?" | the CLI ↔ library table in `AGENTS.md` |
