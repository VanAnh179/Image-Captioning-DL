# Repository working conventions

- Create or reuse a task-specific branch before editing. Use `codex/lstm-attention` for the current LSTM, attention and training work.
- Commit and push task changes on that branch. Merge into `main` only when the user explicitly requests that merge. Preserve source branches unless the user explicitly requests deletion.
- Use clear English source filenames without spaces, following snake_case or camelCase. Place code in the existing project directories.
- Use descriptive Vietnamese filenames for reports in `reports/`. Keep reports concise and include actual checks, results and limitations. The user uploads reports to Media board.
- Write concise commit messages describing the concrete changes. Keep README focused on getting started and link detailed technical reports separately.
