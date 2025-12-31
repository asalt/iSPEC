from __future__ import annotations

"""
Helpers for seeding a sandbox SQLite database with realistic sample data.

The ``build_sandbox_db`` function is used by integration tests and can also be
invoked directly::

    $ python -m tests.integration.sandbox_builder --output sandbox/ispec.db

The resulting database includes linked ``Project``, ``Person``, ``Experiment``,
``ExperimentRun``, and ``E2G`` rows so that other tools (e.g. the frontend repo)
can point at a fully populated environment.
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence


SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from ispec.db.connect import get_session  # noqa: E402
from ispec.db.init import initialize_db  # noqa: E402
from ispec.db.models import (  # noqa: E402
    E2G,
    Experiment,
    ExperimentRun,
    Person,
    Project,
    ProjectPerson,
)


_PERSON_TEMPLATES: list[dict[str, str]] = [
    {
        "first": "Ada",
        "last": "Lovelace",
        "email": "ada.lovelace@example.org",
        "institution": "Analytical Sciences Lab",
    },
    {
        "first": "Katalin",
        "last": "Kariko",
        "email": "katalin.kariko@example.org",
        "institution": "mRNA Innovation Center",
    },
    {
        "first": "Sunita",
        "last": "Williams",
        "email": "sunita.williams@example.org",
        "institution": "Orbital Research Hub",
    },
]


_PROJECT_TEMPLATES: list[dict[str, str]] = [
    {
        "title": "Chromatin Response Atlas",
        "background": "Profiling histone modifications across treatment arms.",
        "description": "Longitudinal AP-MS series for oncology indications.",
    },
    {
        "title": "Signal Peptide Benchmark",
        "background": "Benchmarking signal peptide enrichments for secretome studies.",
        "description": "Includes replicate immunoprecipitations and TMT quant.",
    },
    {
        "title": "Mito Stress Map",
        "background": "Tracking mitochondrial perturbations post compound exposure.",
        "description": "Combines affinity capture with DIA validation runs.",
    },
]


def build_sandbox_db(
    db_path: str | Path,
    *,
    project_templates: Sequence[dict[str, str]] | None = None,
    person_templates: Sequence[dict[str, str]] | None = None,
    experiments_per_project: int = 2,
    runs_per_experiment: int = 2,
    genes_per_run: int = 3,
) -> dict[str, int]:
    """Create and populate a sandbox SQLite database.

    Parameters
    ----------
    db_path:
        Target SQLite file path.
    project_templates:
        Optional override for the default project seed data.
    person_templates:
        Optional override for the default person seed data.
    experiments_per_project, runs_per_experiment, genes_per_run:
        Controls how much experiment/run/E2G data is generated per project.

    Returns
    -------
    dict
        Count of inserted rows per table group.
    """

    if experiments_per_project < 1:
        raise ValueError("experiments_per_project must be >= 1")
    if runs_per_experiment < 1:
        raise ValueError("runs_per_experiment must be >= 1")
    if genes_per_run < 1:
        raise ValueError("genes_per_run must be >= 1")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    project_templates = list(project_templates or _PROJECT_TEMPLATES)
    person_templates = list(person_templates or _PERSON_TEMPLATES)
    if not project_templates:
        raise ValueError("project_templates must contain at least one entry")
    if not person_templates:
        raise ValueError("person_templates must contain at least one entry")

    initialize_db(file_path=str(db_path))

    summary = {"people": 0, "projects": 0, "experiments": 0, "runs": 0, "genes": 0}

    with get_session(file_path=str(db_path)) as session:
        person_rows = []
        for template in person_templates:
            person = Person(
                ppl_AddedBy="sandbox",
                ppl_Name_First=template["first"],
                ppl_Name_Last=template["last"],
                ppl_Email=template["email"],
                ppl_Institution=template["institution"],
            )
            session.add(person)
            person_rows.append(person)
        session.flush()
        summary["people"] = len(person_rows)

        project_rows = []
        for template in project_templates:
            project = Project(
                prj_AddedBy="sandbox",
                prj_ProjectTitle=template["title"],
                prj_ProjectBackground=template["background"],
                prj_ProjectDescription=template["description"],
            )
            session.add(project)
            project_rows.append(project)
        session.flush()
        summary["projects"] = len(project_rows)

        for idx, project in enumerate(project_rows):
            person = person_rows[idx % len(person_rows)]
            session.add(ProjectPerson(project_id=project.id, person_id=person.id))

        experiment_rows: list[Experiment] = []
        for project in project_rows:
            for exp_index in range(1, experiments_per_project + 1):
                record_no = f"{project.id:05d}-{exp_index:02d}"
                experiment = Experiment(project_id=project.id, record_no=record_no)
                session.add(experiment)
                experiment_rows.append(experiment)
        session.flush()
        summary["experiments"] = len(experiment_rows)

        run_rows: list[ExperimentRun] = []
        for experiment in experiment_rows:
            for run_index in range(1, runs_per_experiment + 1):
                run = ExperimentRun(
                    experiment_id=experiment.id,
                    run_no=run_index,
                    search_no=run_index,
                )
                session.add(run)
                run_rows.append(run)
        session.flush()
        summary["runs"] = len(run_rows)

        gene_rows: list[E2G] = []
        for run in run_rows:
            for gene_index in range(1, genes_per_run + 1):
                gene = E2G(
                    experiment_run_id=run.id,
                    gene=f"GENE{run.id:04d}{gene_index:02d}",
                    geneidtype="symbol",
                    label=f"label-{gene_index}",
                    iBAQ_dstrAdj=gene_index * 10.0,
                    peptideprint="_".join(
                        f"PEPTIDE{gene_index}{rep}" for rep in range(1, 3)
                    ),
                )
                session.add(gene)
                gene_rows.append(gene)
        summary["genes"] = len(gene_rows)

    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a sandbox SQLite database with linked demo data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sandbox") / "ispec_sandbox.db",
        help="Path to the SQLite database to create (default: sandbox/ispec_sandbox.db)",
    )
    parser.add_argument("--experiments-per-project", type=int, default=2)
    parser.add_argument("--runs-per-experiment", type=int, default=2)
    parser.add_argument("--genes-per-run", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_sandbox_db(
        args.output,
        experiments_per_project=args.experiments_per_project,
        runs_per_experiment=args.runs_per_experiment,
        genes_per_run=args.genes_per_run,
    )
    human_path = Path(args.output).resolve()
    print(
        f"Sandbox DB created at {human_path} "
        f"(projects={summary['projects']}, experiments={summary['experiments']}, genes={summary['genes']})"
    )


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
