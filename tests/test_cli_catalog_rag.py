from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from evenor.__main__ import app
from evenor.application.erpnext_catalog_sync_service import CatalogRagIndexPlan


runner = CliRunner()


def test_catalog_rag_rebuild_dry_run(test_settings, test_tenant, monkeypatch) -> None:
    tenant, _ = test_tenant
    catalog_dir = test_settings.data_root / "catalog" / tenant.slug
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "A.md").write_text("# A\n", encoding="utf-8")
    (catalog_dir / "B.md").write_text("# B\n", encoding="utf-8")

    plan = CatalogRagIndexPlan(
        needs_embed=[catalog_dir / "A.md"],
        already_indexed=[catalog_dir / "B.md"],
    )

    with patch("evenor.__main__.get_settings", return_value=test_settings), patch(
        "evenor.__main__.catalog_rag_index_plan",
        return_value=plan,
    ), patch("evenor.__main__.reconcile_catalog_rag") as mock_reconcile:
        result = runner.invoke(app, ["catalog-rag", "rebuild", tenant.slug, "--dry-run"])

    assert result.exit_code == 0
    assert "need embedding" in result.stdout
    assert "Dry run: would process 1 file(s)" in result.stdout
    mock_reconcile.assert_not_called()
