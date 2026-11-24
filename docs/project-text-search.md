# Finding a project from remembered words

Use `search_projects(query)` for staff or `my_projects(query)` for a scoped
account. Search for the project, not the person record: a PI or lab contact
can be named on a project without having an entry in the people directory.

An exact ID (including an MSPC display ID) looks up one project. Otherwise
the ranker counts distinct terms across title, PI, lab contact, description,
background, samples and project questions. A staff-wide search also includes
comments. Ties favor stronger fields, then lower project ID; repetitions don't
add votes. Each matched term has a field and a short excerpt. A partial match
is a lead, not evidence that every requested term occurs on the project.

The vocabulary is data in `src/ispec/assistant/project_terms.json`. Title
prefixes and single-character spelling errors in longer names can match with
less weight than exact words. No external service, search index, schema change
or backfill is required. Revisit indexing if catalog growth makes reads slow.

## Reproduce

Run `pytest tests/unit/assistant/test_project_search.py` for retrieval and ACLs.
For the local GPU, the command below runs the existing 32 cases and six new
synthetic lookups against one model, producing a transcript beside each wire
capture. Use a **new** output directory under the ignored local tree.

```sh
python scripts/run_olmo_application_gate.py tests/behavioral/local/lookup-rerun \
  --project-search --compare /path/to/baseline/scores.json
```

The seventh lookup is private and skips by default. To include it, pass
`--local-search-case /private/real.json` and optionally
`--local-search-backup /private/completed-snapshot/ispec-core.db`.
Use only a completed offline snapshot for the latter (opened immutable);
chat state is always separate. Keep the collected case, transcripts and real
ranked excerpts private and out of Git.
