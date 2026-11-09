<!--
File    : docs/workflow.md
Owner   : shared (Amine + Ella)
What    : git and coding conventions
Input   : -
Output  : conventions everyone follows
-->
# Workflow
- Branch per task: `amine/als-model`, `ella/tfidf`. Never push directly to `main`.
- PR -> the co-owner of the file reviews it -> merge. `pytest -q` must pass.
- Keep the header block of every file up to date (owner, what it does, input, output).
- Comment the code: the code itself is graded ("code source commente").
- No data or large outputs in git (see `.gitignore`). Share data through the download script.
- Every hyper-parameter goes in `config/config.yaml`, never hard-coded.
- Fixed seed (`seed: 42`) for sampling, splitting, cold start and ALS.
- Commit messages: `area: short description` (e.g. `models: add gate alpha`).
- **Defense rule:** each of us must be able to explain ANY part of the code.
  Before the defense, each person reads and explains the other's files out loud (one session per folder).