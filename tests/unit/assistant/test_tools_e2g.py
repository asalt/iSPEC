from __future__ import annotations

from ispec.assistant.tools import run_tool
from ispec.db.models import E2G, Experiment, ExperimentRun, Project


def test_e2g_tools_can_search_and_fetch_hits(db_session):
    project = Project(id=1, prj_AddedBy="test", prj_ProjectTitle="Project 1")
    experiment = Experiment(id=100, project_id=1, record_no="100", exp_Name="Experiment 100")
    run = ExperimentRun(experiment_id=100, run_no=1, search_no=1, label="0")
    db_session.add_all([project, experiment, run])
    db_session.flush()

    e2g = E2G(
        experiment_run_id=run.id,
        gene="123",
        geneidtype="GeneID",
        label="0",
        gene_symbol="KRAS",
        psms_u2g=9,
        peptideprint="PEP_A__PEP_B",
    )
    db_session.add(e2g)
    db_session.commit()

    search_payload = run_tool(
        name="e2g_search_genes_in_project",
        args={"project_id": 1, "query": "kras", "limit": 5},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert search_payload["ok"] is True
    assert search_payload["result"]["count"] == 1
    assert search_payload["result"]["matches"][0]["gene_id"] == 123

    hits_payload = run_tool(
        name="e2g_gene_in_project",
        args={"project_id": 1, "gene_id": 123, "limit": 10},
        core_db=db_session,
        schedule_db=None,
        user=None,
        api_schema=None,
    )
    assert hits_payload["ok"] is True
    assert hits_payload["result"]["count"] == 1
    hit = hits_payload["result"]["hits"][0]
    assert hit["experiment_id"] == 100
    assert hit["experiment_run_id"] == run.id
    assert hit["gene_symbol"] == "KRAS"
    assert hit["peptideprint_preview"] == "PEP_A__PEP_B"
