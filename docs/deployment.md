# Deployment guide

This guide walks through preparing a production-style environment for the iSPEC
CLI utilities and FastAPI service. It covers package installation, database and
logging configuration, process management, and upgrade routines so that a new
host can go from an empty virtual machine to a running API instance.

## 1. Prerequisites

* Linux host with Python 3.8+ available on the system PATH.
* Shell access with sudo privileges to create directories under `/var` and
  `/etc` (optional but recommended for multi-user deployments).
* `git` for fetching the repository and a supported database backend (SQLite is
  bundled with Python and used by default).

## 2. Create an installation environment

Production deployments typically isolate Python dependencies in a virtual
environment. The commands below clone the repository, create a virtual
environment, and install iSPEC with the runtime dependencies declared in
`pyproject.toml`:

```bash
sudo mkdir -p /opt/ispec
sudo chown "$USER" /opt/ispec
cd /opt/ispec
git clone https://github.com/<your-org>/iSPEC.git .
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[test]
```

The editable install (`pip install -e`) exposes the `ispec` console script so
administrators can run the CLI directly from the environment.

## 3. Configure environment variables

The toolkit reads a handful of environment variables to discover where databases
and log files live. Copy `.env.example` to a secure location (for example,
`/etc/ispec/.env`) and adjust the paths to match your system layout:

```bash
sudo mkdir -p /etc/ispec
cp .env.example /etc/ispec/.env
```

Populate the file with absolute paths. A typical production configuration might
look like this:

```dotenv
ISPEC_DB_DIR=/var/lib/ispec
ISPEC_LOG_DIR=/var/log/ispec
ISPEC_CONFIG_DIR=/etc/ispec
ISPEC_LOG_CONFIG=/etc/ispec/logging.json
ISPEC_STATE_DIR=/var/lib/ispec/state
```

Load the variables when using the CLI or service process. Two common approaches
are:

* Export the variables for the current shell session:
  ```bash
  set -a
  source /etc/ispec/.env
  set +a
  ```
* Reference the file from a process manager (systemd example shown below).

## 4. Provision the database

Run the CLI once the environment variables are in place. The commands below
create the database directory, initialize the SQLite file, and apply migrations:

```bash
source /opt/ispec/.venv/bin/activate
set -a && source /etc/ispec/.env && set +a
ispec db init
ispec db upgrade
```

To import existing records from CSV/TSV/Excel exports, use the high-level import
helpers:

```bash
ispec db import --table-name person --file people.csv
ispec db import --table-name project --file projects.xlsx
```

The import routine normalizes missing values and uses the CRUD layer to insert
rows so relationship tables (e.g. `project_person`) are updated automatically.

## 5. Launch the API service

For ad-hoc testing you can run the built-in CLI wrapper around Uvicorn:

```bash
ispec api start --host 0.0.0.0 --port 8080
```

The `status` subcommand verifies that the process is responding on its health
endpoint:

```bash
ispec api status
```

In production, delegate to a supervisor such as `systemd` or `supervisord` so
restarts happen automatically. A minimal `systemd` unit might look like this:

```ini
[Unit]
Description=iSPEC API service
After=network.target

[Service]
WorkingDirectory=/opt/ispec
EnvironmentFile=/etc/ispec/.env
ExecStart=/opt/ispec/.venv/bin/ispec api start --host 0.0.0.0 --port 8080
Restart=on-failure
User=ispec
Group=ispec

[Install]
WantedBy=multi-user.target
```

Place the file at `/etc/systemd/system/ispec.service`, reload the daemon, and
start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ispec.service
```

`ispec api status` continues to work because the CLI persists the host/port in a
state file defined by `ISPEC_STATE_DIR` or `ISPEC_API_STATE_FILE`.

## 6. Manage logs

The logging helpers write structured output to `ISPEC_LOG_DIR` and persist log
level choices in `ISPEC_LOG_CONFIG`. Review or rotate the files using your
standard tooling (e.g. `logrotate`). To inspect or change the level at runtime:

```bash
ispec logging show-path
ispec logging show-level
ispec logging set-level INFO
```

## 7. Upgrades and maintenance

1. Pull the latest code and reinstall the package:
   ```bash
   cd /opt/ispec
   source .venv/bin/activate
   git fetch origin
   git checkout <new-tag-or-branch>
   pip install -e .
   ```
2. Re-run migrations to apply schema changes:
   ```bash
   set -a && source /etc/ispec/.env && set +a
   ispec db upgrade
   ```
3. Restart the process manager (`systemctl restart ispec.service`).

Regularly back up the SQLite database file and configuration directory to guard
against data loss.

## 8. Troubleshooting checklist

* **Database path errors** – Confirm `ISPEC_DB_DIR` or `ISPEC_DB_PATH` point to a
  writable directory/file and that the service user owns the path.
* **API not responding** – Use `ispec api status` and check the service logs in
  `ISPEC_LOG_DIR`. Uvicorn errors are echoed there as well as to stdout.
* **Import failures** – Ensure `openpyxl` is installed when loading Excel files
  and verify the tabular data matches the expected schema. The `--file` argument
  must point to CSV/TSV/JSON/XLSX files depending on your data source.

With these steps in place the host is ready for day-to-day operation of the
iSPEC CLI and API service.
