from chatbot.application.connector_service import ConnectorService
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType


class _FakeRepo:
    def __init__(self) -> None:
        self.rows: list = []
        self._next = 1

    def list_for_tenant(self, tenant_id: int):
        return [r for r in self.rows if r.tenant_id == tenant_id]

    def find_by_id(self, connector_id: int):
        for r in self.rows:
            if r.id == connector_id:
                return r
        return None

    def find_by_tenant_direction_type(self, tenant_id, *, direction, type):
        for r in self.rows:
            if r.tenant_id == tenant_id and r.direction == direction and r.type == type:
                return r
        return None

    def list_by_tenant_direction_type(self, tenant_id, *, direction, type):
        return [
            r
            for r in self.rows
            if r.tenant_id == tenant_id and r.direction == direction and r.type == type
        ]

    def find_active(self, tenant_id, *, direction, type):
        for r in self.list_by_tenant_direction_type(tenant_id, direction=direction, type=type):
            if r.active:
                return r
        return None

    def create(self, *, tenant_id, direction, type, mode, config, active=True):
        from chatbot.domain.models.connector import Connector
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        row = Connector(
            id=self._next,
            tenant_id=tenant_id,
            direction=direction,
            type=type,
            mode=mode,
            config=dict(config),
            active=active,
            created_at=now,
            updated_at=now,
        )
        self._next += 1
        self.rows.append(row)
        return row

    def update(self, connector_id, *, config=None, active=None, mode=None):
        from chatbot.domain.models.connector import Connector
        from datetime import datetime, timezone

        row = self.find_by_id(connector_id)
        if row is None:
            return None
        updated = Connector(
            id=row.id,
            tenant_id=row.tenant_id,
            direction=row.direction,
            type=row.type,
            mode=mode or row.mode,
            config=dict(config) if config is not None else dict(row.config),
            active=row.active if active is None else active,
            created_at=row.created_at,
            updated_at=datetime.now(timezone.utc),
        )
        self.rows = [updated if r.id == connector_id else r for r in self.rows]
        return updated

    def delete(self, connector_id):
        before = len(self.rows)
        self.rows = [r for r in self.rows if r.id != connector_id]
        return len(self.rows) < before


def test_create_multiple_ig_accounts() -> None:
    svc = ConnectorService(_FakeRepo())  # type: ignore[arg-type]
    a = svc.create_ig(tenant_id=1, config={"name": "Demo", "acc_type": "DEMO"}, active=True)
    b = svc.create_ig(tenant_id=1, config={"name": "Live", "acc_type": "LIVE"}, active=True)
    rows = svc.list_ig(1)
    assert len(rows) == 2
    assert {a.id, b.id} == {rows[0].id, rows[1].id}
    assert svc.find_ig(1).id == a.id  # first / lowest id


def test_upsert_ig_updates_by_id_without_deleting_others() -> None:
    svc = ConnectorService(_FakeRepo())  # type: ignore[arg-type]
    a = svc.create_ig(tenant_id=1, config={"name": "A"}, active=True)
    b = svc.create_ig(tenant_id=1, config={"name": "B"}, active=True)
    svc.upsert_ig(tenant_id=1, config={"name": "A2"}, active=True, connector_id=a.id)
    rows = svc.list_ig(1)
    assert len(rows) == 2
    assert svc.get_ig_by_id(1, a.id).config["name"] == "A2"
    assert svc.get_ig_by_id(1, b.id).config["name"] == "B"


def test_migrate_ig_does_not_delete_extra_both_rows() -> None:
    svc = ConnectorService(_FakeRepo())  # type: ignore[arg-type]
    both = svc.create_ig(tenant_id=1, config={"name": "Both"}, active=True)
    # Simulate legacy in row
    legacy = svc._repo.create(
        tenant_id=1,
        direction=ConnectorDirection.IN,
        type=ConnectorType.IG,
        mode=ConnectorMode.DIRECT,
        config={"name": "Legacy"},
        active=True,
    )
    assert svc.migrate_ig_to_both(1) is True
    assert svc.get_ig_by_id(1, both.id) is not None
    assert svc.get_ig_by_id(1, legacy.id) is None
    assert len(svc.list_ig(1)) == 1
