# Repository working conventions

- Create or reuse a task-specific branch before editing. Use `lstm-attention` for the current LSTM, attention and training work, following its name on GitHub.
- Use descriptive branch names without the `codex/` prefix.
- Commit and push task changes on that branch. Merge into `main` only when the user explicitly requests that merge. Preserve source branches unless the user explicitly requests deletion.
- Use clear English source filenames without spaces, following snake_case or camelCase. Place code in the existing project directories.
- Use descriptive Vietnamese filenames for reports in `reports/`. Keep reports concise and include actual checks, results and limitations. The user uploads reports to Media board. Keep the entire reports directory, including paper sources and evaluation outputs, out of Git. Keep datasets, logs, checkpoints, and private handoff files out of Git as well.
- Write concise commit messages describing the concrete changes. Keep README focused on getting started. Link only documentation meant to be shared in the repository.
