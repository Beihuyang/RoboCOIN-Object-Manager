# Documentation maintenance

When changing any user-facing workflow, page control, model behavior, filtering rule, threshold, cache behavior, output, or recovery procedure:

1. Update `docs/DATA_COLLECTOR_GUIDE.md` in the same change.
2. Keep button names and visible status text consistent with the current UI.
3. Add a concise dated entry to the guide's “更新记录” table for material workflow changes.
4. Update the relevant technical explanation in `README.md` when implementation behavior changes.

Do not document a feature as available until its implementation has been verified.
