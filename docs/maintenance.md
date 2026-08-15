# Maintenance cycle

The project follows a 7/14/28-day review cycle. The calendar creates a review point, not a reason
to publish an empty release.

| Cadence | Review | Expected output |
|---|---|---|
| Every 7 days | Tests, packaging, installed CLI, dependencies, Actions pins, open defects | Passing health run or a scoped repair issue |
| Every 14 days | Issues, Discussions, repeated user workflows, documentation gaps | Ranked demand notes and accepted or declined proposals |
| Every 28 days | Adjacent projects, interface standards, roadmap fit, maintenance cost | Updated priorities and a release decision |
| Within 48 hours | Confirmed high-severity security issue | Triage, mitigation plan, and advisory workflow |

The scheduled GitHub workflow runs each Monday at 02:17 UTC. Dependabot checks Python packages
and Actions each Monday in the `Asia/Taipei` time zone. Maintainers still review and merge every
change through the protected default branch.

## How an update moves

1. Collect evidence from reproducible issues, Discussions, downstream usage, and active adjacent
   projects.
2. Rank the proposal by user value, fit with controlled tabular diagnostics, validation cost, and
   long-term maintenance cost.
3. Implement one focused change on a branch and document its behavior and limits.
4. Run lint, formatting, coverage, experiment, build, metadata, and installed-wheel checks.
5. Publish through a pull request; merge only after every required check passes.
6. Release only when the merged change gives users a meaningful new capability or fixes a defect.
7. Review adoption and failures at the next 14-day checkpoint.

## Demand gates

A large dependency or new problem domain normally needs at least one of these signals:

- three independent requests describing the same workflow within 28 days;
- two external projects using or linking the proposed interface; or
- one reproducible high-value use case with a clear evaluation plan and a committed maintainer.

Security, correctness, and compatibility repairs do not wait for a demand threshold.
