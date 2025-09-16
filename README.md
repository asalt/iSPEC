# iSPEC

A Python toolkit for managing the iSPEC database, command-line utilities, and
FastAPI service. The project wraps a SQLite/SQLAlchemy data model with a REST
API, schema generation helpers, and import/export pipelines so research teams
can manage people, projects, and supporting documents from one place.

## Table of contents

<!-- TOC_START -->
- [Overview](#overview)
- [Key features](#key-features)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Database location](#database-location)
- [Command line interface](#command-line-interface)
  - [Database commands](#database-commands)
  - [Database migrations](#database-migrations)
  - [API commands](#api-commands)
  - [Logging commands](#logging-commands)
- [API service](#api-service)
- [Data import and export](#data-import-and-export)
- [Logging](#logging)
- [Running tests](#running-tests)
- [Documentation utilities](#documentation-utilities)
- [Contributing](#contributing)
<!-- TOC_END -->

## Overview

iSPEC centers on a SQLite database with SQLAlchemy models for core entities such
as `Person`, `Project`, `ProjectComment`, and `LetterOfSupport` and the
relationships between them.【F:src/ispec/db/models.py†L36-L183】【F:src/ispec/db/models.py†L184-L288】
The package exposes the models through:

* an installable CLI (`ispec`) for database administration, API control, and
  logging management【F:src/ispec/cli/main.py†L7-L33】;
* a FastAPI application that generates CRUD routers, form schemas, and async
  select option endpoints on top of the models and CRUD classes【F:src/ispec/api/main.py†L1-L15】【F:src/ispec/api/routes/routes.py†L1-L145】;
* data import/export helpers that convert CSV/TSV/Excel files into database
  rows and wire up cross-table relationships.【F:src/ispec/io/io_file.py†L1-L82】

## Key features

- **Dynamic REST API** – Automatic CRUD routers with JSON schema metadata and
  `/options` endpoints make it easy to build form-driven UIs on top of the
  database.【F:src/ispec/api/routes/routes.py†L17-L142】【F:docs/api-schema.md†L1-L43】
- **End-to-end CLI** – `ispec db` covers initialization, status checks, table
  listing, and import/export pipelines, while `ispec api` and `ispec logging`
  provide server and log controls.【F:src/ispec/cli/db.py†L11-L78】【F:src/ispec/cli/api.py†L11-L63】【F:src/ispec/cli/logging.py†L11-L55】
- **Database utilities** – Centralized session management, configurable storage
  paths, and helpers for creating or connecting to SQLite databases are built in
  to simplify deployment.【F:src/ispec/db/connect.py†L17-L94】
- **Structured logging** – A reusable logging factory writes to both console and
  file, with CLI controls to adjust levels or inspect the log path.【F:src/ispec/logging/logging.py†L1-L74】【F:src/ispec/cli/logging.py†L11-L55】
- **Extensive tests** – Integration tests exercise CLI commands and CRUD
  endpoints to ensure the toolkit works together end to end.【F:tests/integration/test_cli_db.py†L1-L129】【F:tests/integration/test_api_endpoints.py†L1-L121】

## Project layout

<!-- PROJECT_TREE_START -->
```text
iSPEC/
├── docs/                             # Generated and hand-written documentation assets
├── sql/                              # SQL initialization scripts
├── src/ispec/                        # Package source code
│   ├── api/                          # FastAPI app, routers, and schema builders
│   ├── cli/                          # argparse-powered CLI entry points
│   ├── db/                           # Database models, CRUD helpers, and session tooling
│   ├── io/                           # File import/export utilities
│   └── logging/                      # Logging configuration helpers
├── tests/                            # Unit and integration tests
├── pyproject.toml                    # Project metadata and dependency declarations
└── README.md                         # This guide
```
<!-- PROJECT_TREE_END -->

## Getting started

### Prerequisites

- Python 3.8 or newer.【F:pyproject.toml†L12-L18】
- SQLite 3 (bundled with Python, used through SQLAlchemy).

### Installation

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install iSPEC and its dependencies in editable mode:

   ```bash
   pip install -e .[test]
   ```

   Editable installs expose the `ispec` CLI entry point defined in the project
   metadata.【F:pyproject.toml†L20-L34】

### Database location

By default iSPEC stores its SQLite database in `~/ispec/ispec.db`. You can
control where the data lives with environment variables:

- `ISPEC_DB_DIR` – root directory for databases (defaults to `~/ispec`).
- `ISPEC_DB_PATH` – full SQLite path/URI used by session helpers and tests.

Both variables are respected by the connection utilities, which ensure the
folders exist and wire up SQLAlchemy session factories for you.【F:src/ispec/db/connect.py†L17-L94】

## Command line interface

iSPEC installs a single `ispec` entry point that exposes subcommands for the
database, API service, and logging configuration.【F:src/ispec/cli/main.py†L7-L33】

### Database commands

Initialize a new SQLite database (optionally providing a file path):

```bash
ispec db init --file ./ispec.db
```

Check the SQLite version and list tables:

```bash
ispec db status
ispec db show
```

Import CSV/TSV/Excel data into a table and export rows back out to CSV:

```bash
ispec db import --table-name person --file people.csv
ispec db export --table-name project --file project.csv
```

These commands delegate to the CRUD layer and IO helpers that batch insert
rows, update relationship tables, and log progress.【F:src/ispec/cli/db.py†L11-L78】【F:tests/integration/test_cli_db.py†L27-L128】【F:src/ispec/io/io_file.py†L1-L82】

### Database migrations

Alembic migrations live in the top-level ``alembic`` directory. The migration
environment imports the SQLAlchemy metadata from ``ispec.db.models`` so
autogeneration and revision scripts stay in sync with the ORM definitions.【F:alembic/env.py†L1-L66】
An initial revision provisions all tables defined by the models by calling
``Base.metadata.create_all`` inside Alembic's ``upgrade`` hook.【F:alembic/versions/0001_initial.py†L1-L26】

Run migrations through the CLI:

```bash
ispec db upgrade            # apply all migrations to the head revision
ispec db downgrade -1       # roll back the most recent revision
ispec db upgrade head --database ./custom.db
```

The CLI builds an Alembic configuration at runtime, resolves the project root,
and forwards the ``--database`` option so you can target specific SQLite files
or URLs when applying migrations.【F:src/ispec/cli/db.py†L36-L148】

### API commands

Start the FastAPI server with custom host/port options or check its status:

```bash
ispec api start --host 0.0.0.0 --port 9000
ispec api status
```

Behind the scenes the command loads `ispec.api.main:app` and hands it to
`uvicorn.run` for you.【F:src/ispec/cli/api.py†L11-L63】 Tests verify the command
passes through host/port settings as expected.【F:tests/integration/test_cli_api.py†L1-L38】

### Logging commands

Control logging output without editing code:

```bash
ispec logging set-level INFO
ispec logging show-path
ispec logging show-level
```

The logging CLI persists the selected level to a JSON config alongside the
logs, resets handlers when you change levels, reports the active log file
path resolved by the logging utility module, and prints the configured log
level for quick inspection.【F:src/ispec/cli/logging.py†L11-L55】【F:src/ispec/logging/config.py†L1-L90】【F:src/ispec/logging/logging.py†L1-L88】

## API service

The FastAPI application bundles multiple routers generated from SQLAlchemy
models and CRUD classes, yielding endpoints such as:

- `GET /status` health probe.【F:src/ispec/api/routes/routes.py†L205-L208】
- CRUD endpoints under `/people`, `/projects`, and `/project_comment` that
  accept and return dynamically built Pydantic models.【F:src/ispec/api/routes/routes.py†L90-L141】【F:tests/integration/test_api_endpoints.py†L27-L121】
- `GET /<resource>/schema` for JSON form metadata and `GET /<resource>/options`
  for populating select components.【F:src/ispec/api/routes/routes.py†L17-L89】

Routers use dependency-injected sessions from `ispec.db.connect.get_session`, so
tests can swap in temporary databases with ease.【F:src/ispec/db/connect.py†L95-L152】【F:tests/integration/test_api_endpoints.py†L1-L26】

## Data import and export

iSPEC ships with bulk import utilities that detect file formats (CSV/TSV/Excel),
normalize missing values, and invoke the appropriate CRUD class to insert
records. Post-import helpers maintain relationship tables (e.g., linking people
and projects).【F:src/ispec/io/io_file.py†L1-L82】 Exports rely on SQLAlchemy to
materialize whole tables to CSV files.【F:src/ispec/db/operations.py†L39-L54】

## Logging

Logging helpers resolve configurable directories (`ISPEC_LOG_DIR`), create log
files if needed, persist log level choices, and load saved levels when
``get_logger`` is called without an explicit override. Use the CLI to adjust
levels at runtime or inspect the log location.【F:src/ispec/logging/logging.py†L1-L94】【F:src/ispec/logging/config.py†L1-L90】【F:src/ispec/cli/logging.py†L11-L55】

## Running tests

The repository includes unit and integration test suites. After installing the
`test` extras, run all tests with:

```bash
pytest
```

Tests cover CLI flows, API endpoints, and IO utilities to ensure the core
workflows behave as expected.【F:tests/integration/test_cli_db.py†L1-L129】【F:tests/integration/test_api_endpoints.py†L1-L121】
Alembic migrations are validated by executing ``alembic upgrade head`` against a
temporary database as part of the unit test suite.【F:tests/unit/db/test_migrations.py†L1-L39】

## Documentation utilities

`docs/generate_api_schema_md.py` regenerates the API schema reference from the
live FastAPI routers, providing up-to-date examples of the JSON schema metadata
used by the frontend.【F:docs/generate_api_schema_md.py†L1-L136】 The generated
`docs/api-schema.md` explains how schema generation works and shows an example
response.【F:docs/api-schema.md†L1-L132】

`docs/update_readme.py` refreshes the table of contents and project layout
sections of this README so they always reflect the current repository state.

## Contributing

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies with `pip install -e .[test]`.
3. Run `pytest` before submitting pull requests.
4. Open a pull request describing your changes and referencing any relevant
   issues.

The project follows conventional Python formatting via Black (configured for an
88-character line length) and includes mypy/pyright settings in `pyproject.toml`.
【F:pyproject.toml†L36-L47】
