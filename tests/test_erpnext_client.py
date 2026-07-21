from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from chatbot.adapters.erpnext.client import ErpNextClient, _phone_variants


def _config() -> dict:
    return {
        "url": "https://erp.example.com",
        "api_key": "key",
        "api_secret": "secret",
        "identity_email_field": "email_id",
        "identity_phone_field": "mobile_no",
    }


def test_phone_variants() -> None:
    variants = _phone_variants("+33612345678")
    assert "+33612345678" in variants
    assert "33612345678" in variants


def test_create_customer_posts_customer_and_contact() -> None:
    client = ErpNextClient(_config())
    posts: list[tuple[str, dict]] = []
    puts: list[tuple[str, dict]] = []

    def fake_post(path: str, *, json_body: dict) -> dict:
        posts.append((path, json_body))
        if path.endswith("/Customer"):
            return {"data": {"name": "Alice Corp", "customer_name": "Alice Corp"}}
        if path.endswith("/Contact"):
            return {"data": {"name": "Contact-1"}}
        return {}

    def fake_put(path: str, *, json_body: dict) -> dict:
        puts.append((path, json_body))
        return {"data": {"name": "Alice Corp"}}

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Customer/Alice%20Corp") or path.endswith("/Customer/Alice Corp"):
            return {"data": {"name": "Alice Corp", "customer_name": "Alice Corp"}}
        return {}

    with patch.object(client, "_post_write", side_effect=fake_post), patch.object(
        client, "_put", side_effect=fake_put
    ), patch.object(client, "_get", side_effect=fake_get), patch.object(
        client, "_find_contact", return_value=None
    ):
        out = client.create_customer(
            "Alice Corp",
            email="alice@example.com",
            phone="33612345678",
            customer_group="Commercial",
        )

    assert out["customer_name"] == "Alice Corp"
    assert out["contact_name"] == "Contact-1"
    assert posts[0][0].endswith("/Customer")
    assert posts[0][1]["data"]["customer_group"] == "Commercial"
    assert posts[1][0].endswith("/Contact")
    assert posts[1][1]["data"]["links"][0]["link_name"] == "Alice Corp"
    assert puts[0][1]["data"]["customer_primary_contact"] == "Contact-1"


def test_create_customer_company_type() -> None:
    client = ErpNextClient(_config())
    posts: list[tuple[str, dict]] = []

    def fake_post(path: str, *, json_body: dict) -> dict:
        posts.append((path, json_body))
        if path.endswith("/Customer"):
            return {"data": {"name": "Acme Corp", "customer_name": "Acme Corp"}}
        if path.endswith("/Contact"):
            return {"data": {"name": "Contact-1"}}
        return {}

    def fake_put(path: str, *, json_body: dict) -> dict:
        return {"data": {"name": "Acme Corp"}}

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if "Customer" in path:
            return {"data": {"name": "Acme Corp", "customer_name": "Acme Corp"}}
        return {}

    with patch.object(client, "_post_write", side_effect=fake_post), patch.object(
        client, "_put", side_effect=fake_put
    ), patch.object(client, "_get", side_effect=fake_get), patch.object(
        client, "_find_contact", return_value=None
    ):
        out = client.create_customer(
            "Samuel MAMET",
            email="samuel@example.com",
            company_name="Acme Corp",
        )

    assert out["customer_name"] == "Acme Corp"
    assert out["company_name"] == "Acme Corp"
    assert posts[0][1]["data"]["customer_type"] == "Company"
    assert posts[0][1]["data"]["customer_name"] == "Acme Corp"
    assert posts[1][1]["data"]["first_name"] == "Samuel"
    assert posts[1][1]["data"]["last_name"] == "MAMET"
    assert posts[1][1]["data"]["company_name"] == "Acme Corp"
    assert posts[0][1]["data"]["customer_group"] == "Individual"


def test_create_customer_uses_default_customer_group() -> None:
    client = ErpNextClient(_config())
    posts: list[tuple[str, dict]] = []

    def fake_post(path: str, *, json_body: dict) -> dict:
        posts.append((path, json_body))
        if path.endswith("/Customer"):
            return {"data": {"name": "Bob", "customer_name": "Bob"}}
        if path.endswith("/Contact"):
            return {"data": {"name": "Contact-1"}}
        return {}

    with patch.object(client, "_post_write", side_effect=fake_post), patch.object(
        client, "_put", return_value={"data": {}}
    ), patch.object(client, "_get", return_value={"data": {"name": "Bob"}}), patch.object(
        client, "_find_contact", return_value=None
    ):
        client.create_customer("Bob", email="bob@example.com")

    assert posts[0][1]["data"]["customer_group"] == "Individual"


def test_get_item_by_code_returns_row() -> None:
    client = ErpNextClient(_config())
    with patch.object(
        client,
        "_list_resource",
        return_value=[{"item_code": "OKI Consumables", "item_name": "EP-M C650"}],
    ):
        row = client.get_item_by_code("OKI Consumables")
    assert row is not None
    assert row["item_code"] == "OKI Consumables"


def test_resolve_item_by_name() -> None:
    client = ErpNextClient(_config())
    with patch.object(client, "get_item_by_code", return_value=None), patch.object(
        client,
        "search_items",
        return_value=[
            {"item_code": "OKI Consumables", "item_name": "EP-M C650 (Image Drum)"},
        ],
    ):
        row = client.resolve_item("EP-M C650 (Image Drum)")
    assert row is not None
    assert row["item_code"] == "OKI Consumables"


def test_find_customer_by_email() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Contact/Contact-1"):
            return {
                "data": {
                    "name": "Contact-1",
                    "first_name": "Alice",
                    "last_name": "Smith",
                    "links": [{"link_doctype": "Customer", "link_name": "Alice Corp"}],
                }
            }
        assert "Contact" in path
        filters = json.loads(params["filters"])  # type: ignore[index]
        if filters == [["email_id", "=", "alice@example.com"]]:
            return {"data": [{"name": "Contact-1"}]}
        if filters == [["Contact Email", "email_id", "=", "alice@example.com"]]:
            return {"data": []}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(email="alice@example.com") == "Alice Corp"


def test_find_customer_fetches_full_contact_when_list_omits_links() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Contact/United%20Docks%20Business%20Park-United%20Docks%20Business%20Park"):
            return {
                "data": {
                    "name": "United Docks Business Park-United Docks Business Park",
                    "links": [
                        {
                            "link_doctype": "Customer",
                            "link_name": "United Docks Business Park",
                        }
                    ],
                }
            }
        filters = json.loads(params["filters"])  # type: ignore[index]
        if filters == [["email_id", "=", "agoburdhun@uniteddocks.com"]]:
            return {
                "data": [{"name": "United Docks Business Park-United Docks Business Park"}]
            }
        if filters == [["Contact Email", "email_id", "=", "agoburdhun@uniteddocks.com"]]:
            return {"data": []}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(email="agoburdhun@uniteddocks.com") == "United Docks Business Park"


def test_find_customer_by_phone() -> None:
    client = ErpNextClient(_config())
    calls: list[str] = []

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Contact/Contact-2"):
            return {
                "data": {
                    "name": "Contact-2",
                    "links": [{"link_doctype": "Customer", "link_name": "Bob SA"}],
                }
            }
        filters = json.loads(params["filters"])  # type: ignore[index]
        calls.append(filters[0][2])
        if filters[0][2] == "+33612345678":
            return {"data": [{"name": "Contact-2"}]}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(phone="33612345678") == "Bob SA"
    assert "+33612345678" in calls or "33612345678" in calls


def test_get_sales_invoices_and_quotations_with_line_items() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Sales Invoice/SINV-001"):
            return {
                "data": {
                    "name": "SINV-001",
                    "items": [
                        {
                            "item_code": "ITEM-1",
                            "item_name": "Widget",
                            "qty": 2,
                            "rate": 10,
                            "amount": 20,
                            "uom": "Nos",
                        }
                    ],
                }
            }
        if path.endswith("/Quotation/QT-001"):
            return {
                "data": {
                    "name": "QT-001",
                    "items": [
                        {
                            "item_code": "ITEM-2",
                            "item_name": "Gadget",
                            "qty": 1,
                            "rate": 5,
                            "amount": 5,
                            "uom": "Nos",
                        }
                    ],
                }
            }
        if "Sales Invoice" in path:
            return {
                "data": [
                    {
                        "name": "SINV-001",
                        "posting_date": "2026-01-01",
                        "status": "Paid",
                        "grand_total": 20,
                    }
                ]
            }
        if "Quotation" in path:
            return {
                "data": [
                    {
                        "name": "QT-001",
                        "transaction_date": "2026-02-01",
                        "status": "Open",
                        "grand_total": 5,
                    }
                ]
            }
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        orders = client.get_orders("Alice Corp", 5)
        quotes = client.get_quotations("Alice Corp", 5)
    assert orders[0]["name"] == "SINV-001"
    assert orders[0]["transaction_date"] == "2026-01-01"
    assert orders[0]["items"][0]["item_name"] == "Widget"
    assert quotes[0]["name"] == "QT-001"
    assert quotes[0]["items"][0]["item_name"] == "Gadget"


def test_get_sales_orders_list_only() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if "Sales Order" in path:
            return {"data": [{"name": "SO-001", "status": "To Deliver"}]}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        orders = client.get_sales_orders("Alice Corp", 5)
    assert orders[0]["name"] == "SO-001"


def test_get_customer_profile_and_contact() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Customer/Alice%20Corp"):
            return {
                "data": {
                    "name": "Alice Corp",
                    "customer_name": "Alice Corp",
                    "customer_type": "Company",
                    "customer_group": "Commercial",
                    "territory": "All Territories",
                    "email_id": "info@alice.example",
                    "mobile_no": "1111",
                    "customer_primary_contact": "Alice Smith",
                }
            }
        if path.endswith("/Contact/Alice%20Smith"):
            return {
                "data": {
                    "name": "Alice Smith",
                    "first_name": "Alice",
                    "last_name": "Smith",
                    "email_id": "alice@example.com",
                    "mobile_no": "2222",
                    "designation": "Buyer",
                }
            }
        if path.endswith("/Address/Alice%20Corp-Billing"):
            return {
                "data": {
                    "address_title": "Billing",
                    "address_line1": "1 Main St",
                    "city": "Port Louis",
                    "country": "Mauritius",
                }
            }
        if "Address" in path:
            return {"data": [{"name": "Alice Corp-Billing"}]}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        profile = client.get_customer_profile("Alice Corp")
        contact = client.get_matched_contact(
            email="alice@example.com",
            phone=None,
            customer="Alice Corp",
        )
    assert profile["name"] == "Alice Corp"
    assert profile["customer_type"] == "Company"
    assert profile["address"]["city"] == "Port Louis"
    assert contact is not None
    assert contact["full_name"] == "Alice Smith"
    assert contact["email"] == "alice@example.com"


def test_find_customer_prefers_person_contact_from_child_email() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Contact/Anju%20."):
            return {
                "data": {
                    "name": "Anju .",
                    "first_name": "Anju",
                    "last_name": ".",
                    "email_ids": [{"email_id": "agoburdhun@uniteddocks.com"}],
                    "links": [
                        {
                            "link_doctype": "Customer",
                            "link_name": "United Docks Business Park",
                        }
                    ],
                }
            }
        if path.endswith("/Contact/United%20Docks%20Business%20Park-United%20Docks%20Business%20Park"):
            return {
                "data": {
                    "name": "United Docks Business Park-United Docks Business Park",
                    "first_name": "United Docks Business Park",
                    "email_id": "agoburdhun@uniteddocks.com",
                    "links": [
                        {
                            "link_doctype": "Customer",
                            "link_name": "United Docks Business Park",
                        }
                    ],
                }
            }
        if params:
            filters = json.loads(params["filters"])
            if filters == [["email_id", "=", "agoburdhun@uniteddocks.com"]]:
                return {
                    "data": [{"name": "United Docks Business Park-United Docks Business Park"}]
                }
            if filters == [["Contact Email", "email_id", "=", "agoburdhun@uniteddocks.com"]]:
                return {"data": [{"name": "Anju ."}]}
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        assert client.find_customer(email="agoburdhun@uniteddocks.com") == "United Docks Business Park"
        contact = client.get_matched_contact(
            email="agoburdhun@uniteddocks.com",
            phone=None,
            customer="United Docks Business Park",
        )
    assert contact is not None
    assert contact["full_name"] == "Anju ."


def test_get_current_item_prices_uses_price_list_then_standard_rate() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Customer/Alice%20Corp"):
            return {"data": {"name": "Alice Corp", "default_price_list": "Standard Selling"}}
        if params and "Item Price" in path:
            filters = json.loads(params["filters"])
            assert filters[1] == ["price_list", "=", "Standard Selling"]
            return {
                "data": [
                    {
                        "item_code": "W-1",
                        "price_list_rate": 55,
                        "currency": "MUR",
                        "uom": "Nos",
                    }
                ]
            }
        if params and "Item" in path:
            return {
                "data": [
                    {
                        "item_code": "G-1",
                        "standard_rate": 12,
                        "stock_uom": "Nos",
                    }
                ]
            }
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        prices = client.get_current_item_prices("Alice Corp", ["W-1", "G-1"])
    assert prices["W-1"]["current_rate"] == 55
    assert prices["W-1"]["source"] == "price_list"
    assert prices["G-1"]["current_rate"] == 12
    assert prices["G-1"]["source"] == "standard_rate"


def test_get_current_item_prices_skips_zero_and_falls_back_to_standard_rate() -> None:
    client = ErpNextClient(_config())

    def fake_get(path: str, *, params: dict | None = None) -> dict:
        if path.endswith("/Customer/Alice%20Corp"):
            return {"data": {"name": "Alice Corp", "default_price_list": "Standard Selling"}}
        if params and "Item Price" in path:
            return {
                "data": [
                    {"item_code": "ZERO-PL", "price_list_rate": 0, "uom": "Nos"},
                    {"item_code": "MISSING", "price_list_rate": 0, "uom": "Nos"},
                ]
            }
        if params and "Item" in path:
            return {
                "data": [
                    {"item_code": "ZERO-PL", "standard_rate": 150, "stock_uom": "Nos"},
                    {"item_code": "MISSING", "standard_rate": 0, "stock_uom": "Nos"},
                ]
            }
        return {"data": []}

    with patch.object(client, "_get", side_effect=fake_get):
        prices = client.get_current_item_prices("Alice Corp", ["ZERO-PL", "MISSING", "UNKNOWN"])
    assert prices["ZERO-PL"]["current_rate"] == 150
    assert prices["ZERO-PL"]["source"] == "standard_rate"
    assert "MISSING" not in prices
    assert "UNKNOWN" not in prices


def test_find_customer_returns_none_on_http_error() -> None:
    client = ErpNextClient(_config())
    with patch.object(client, "_get", return_value={}):
        assert client.find_customer(email="missing@example.com") is None


def test_fetch_latest_invoice_rates_uses_newest_invoice_first() -> None:
    client = ErpNextClient(_config())

    def fake_list(
        doctype: str,
        *,
        filters: list[list[str]],
        fields: list[str],
        limit: int,
        order_by: str = "modified desc",
        limit_start: int = 0,
    ) -> list[dict]:
        assert doctype == "Sales Invoice"
        return [{"name": "SINV-NEW"}, {"name": "SINV-OLD"}]

    def fake_get_resource(doctype: str, name: str) -> dict:
        if name == "SINV-NEW":
            return {
                "currency": "MUR",
                "items": [
                    {"item_code": "A", "rate": 0, "uom": "Nos"},
                    {"item_code": "B", "rate": 500, "uom": "Nos"},
                ],
            }
        return {
            "currency": "MUR",
            "items": [{"item_code": "A", "rate": 100, "uom": "Nos"}],
        }

    with patch.object(client, "_list_resource", side_effect=fake_list), patch.object(
        client, "_get_resource", side_effect=fake_get_resource
    ):
        rates = client.fetch_latest_invoice_rates(item_codes={"A", "B"})

    assert rates["A"]["rate"] == 100.0
    assert rates["B"]["rate"] == 500.0
    assert rates["B"]["price_list"] == "last invoice"


def test_download_quotation_pdf_uses_print_template_endpoint() -> None:
    client = ErpNextClient(_config())
    calls: list[tuple[str, dict[str, str] | None]] = []

    def fake_fetch_bytes(path: str, *, params: dict[str, str] | None = None) -> tuple[bytes, str | None]:
        calls.append((path, params))
        if path.endswith("frappe.templates.pages.print.download_pdf") and params and params.get("format") == "Standard":
            return b"%PDF-1.4", None
        return b"", "Print format not found"

    with patch.object(client, "_fetch_bytes", side_effect=fake_fetch_bytes), patch.object(
        client, "_discover_quotation_print_formats", return_value=[]
    ), patch.object(client, "_default_quotation_print_format", return_value=None):
        data = client.download_quotation_pdf("QTN-0001")

    assert data == b"%PDF-1.4"
    assert calls[0][0].endswith("frappe.templates.pages.print.download_pdf")
    assert calls[0][1] == {"doctype": "Quotation", "name": "QTN-0001", "no_letterhead": "0"}
    assert any(call[1] and call[1].get("format") == "Standard" for call in calls)


def test_download_quotation_pdf_uses_configured_format_first() -> None:
    cfg = _config()
    cfg["quotation_print_format"] = "Custom Quote"
    client = ErpNextClient(cfg)

    def fake_fetch_bytes(path: str, *, params: dict[str, str] | None = None) -> tuple[bytes, str | None]:
        if params and params.get("format") == "Custom Quote":
            return b"%PDF-custom", None
        return b"", "not found"

    with patch.object(client, "_fetch_bytes", side_effect=fake_fetch_bytes), patch.object(
        client, "_discover_quotation_print_formats", return_value=["Other"]
    ), patch.object(client, "_default_quotation_print_format", return_value=None):
        data = client.download_quotation_pdf("QTN-0002")

    assert data == b"%PDF-custom"


def test_default_quotation_print_format_from_property_setter() -> None:
    client = ErpNextClient(_config())

    def fake_list(doctype: str, **kwargs) -> list[dict]:
        if doctype == "Property Setter":
            return [{"value": "Vdtec Quotation"}]
        return []

    with patch.object(client, "_list_resource", side_effect=fake_list):
        assert client._default_quotation_print_format() == "Vdtec Quotation"


def test_pdf_failure_hint_broken_images() -> None:
    hint = ErpNextClient._pdf_failure_hint("PDF generation failed because of broken image links")
    assert "host_name" in hint
    assert "broken image" not in hint.lower() or "load images" in hint.lower()


def test_submit_quotation_fetches_fresh_doc_before_submit() -> None:
    client = ErpNextClient(_config())
    submitted: list[dict] = []

    def fake_get_quotation(name: str) -> dict:
        if submitted:
            return {"name": name, "doctype": "Quotation", "docstatus": 1}
        return {
            "name": name,
            "doctype": "Quotation",
            "docstatus": 0,
            "modified": "2026-06-15 14:18:15.466254",
        }

    def fake_post_write(path: str, *, json_body: dict) -> dict:
        submitted.append(json_body["doc"])
        return {"message": "ok"}

    with patch.object(client, "get_quotation", side_effect=fake_get_quotation), patch.object(
        client, "_post_write", side_effect=fake_post_write
    ):
        assert client.submit_quotation("QTN-0001") is None

    assert submitted[0]["modified"] == "2026-06-15 14:18:15.466254"
    assert submitted[0]["docstatus"] == 0


def test_submit_quotation_skips_when_already_submitted() -> None:
    client = ErpNextClient(_config())

    with patch.object(
        client,
        "get_quotation",
        return_value={"name": "QTN-0002", "doctype": "Quotation", "docstatus": 1},
    ), patch.object(client, "_post_write") as post_write:
        assert client.submit_quotation("QTN-0002") is None

    post_write.assert_not_called()


def test_submit_quotation_retries_after_timestamp_mismatch() -> None:
    client = ErpNextClient(_config())
    get_calls = [
        {
            "name": "QTN-0003",
            "doctype": "Quotation",
            "docstatus": 0,
            "modified": "2026-06-15 14:17:39.476598",
        },
        {
            "name": "QTN-0003",
            "doctype": "Quotation",
            "docstatus": 0,
            "modified": "2026-06-15 14:18:15.466254",
        },
        {
            "name": "QTN-0003",
            "doctype": "Quotation",
            "docstatus": 1,
            "modified": "2026-06-15 14:18:15.466254",
        },
    ]

    with patch.object(client, "get_quotation", side_effect=get_calls), patch.object(
        client,
        "_post_write",
        side_effect=[
            {"_erpnext_error": "modified after you have opened it"},
            {"message": "ok"},
        ],
    ):
        assert client.submit_quotation("QTN-0003") is None


def test_probe_invoice_prices_reports_item_price_permission_error() -> None:
    client = ErpNextClient(_config())
    response = MagicMock()
    response.status_code = 403
    response.json.return_value = {"message": "Not permitted"}
    err = __import__("httpx").HTTPStatusError(
        "forbidden",
        request=MagicMock(),
        response=response,
    )

    def fake_get(url, *, headers, params=None):
        if "Item Price" in url:
            raise err
        response_ok = MagicMock()
        response_ok.raise_for_status = MagicMock()
        response_ok.json.return_value = {"data": [{"name": "SINV-0001"}]}
        return response_ok

    with patch("chatbot.adapters.erpnext.client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = fake_get
        with patch.object(client, "fetch_latest_invoice_rates", return_value={"A": {"rate": 1.0}}):
            result = client.probe_invoice_prices()
    assert result["ok"] is False
    assert result["item_price_http_status"] == 403
    assert result["item_price_access"] is False
    assert "Not permitted" in result["item_price_error"]
    assert "Item Price" in result["preview"]


def test_probe_invoice_prices_reports_invoice_permission_error() -> None:
    client = ErpNextClient(_config())
    response = MagicMock()
    response.status_code = 403
    response.json.return_value = {"message": "Not permitted"}
    err = __import__("httpx").HTTPStatusError(
        "forbidden",
        request=MagicMock(),
        response=response,
    )

    def fake_get(url, *, headers, params=None):
        if "Item Price" in url:
            response_ok = MagicMock()
            response_ok.raise_for_status = MagicMock()
            response_ok.json.return_value = {"data": [{"item_code": "X"}]}
            return response_ok
        raise err

    with patch("chatbot.adapters.erpnext.client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = fake_get
        result = client.probe_invoice_prices()
    assert result["ok"] is False
    assert result["item_price_access"] is True
    assert result["http_status"] == 403
    assert "Not permitted" in result["error"]


def test_probe_invoice_prices_success() -> None:
    client = ErpNextClient(_config())
    list_payload = {"data": [{"name": "SINV-0001", "posting_date": "2026-01-01"}]}
    detail_payload = {
        "data": {
            "name": "SINV-0001",
            "currency": "MUR",
            "items": [{"item_code": "ITEM-A", "rate": 100.0}],
        }
    }

    def fake_get(url, *, headers, params=None):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if "Item Price" in url:
            response.json.return_value = {"data": [{"item_code": "ITEM-A", "price_list_rate": 50}]}
        elif params is not None:
            response.json.return_value = list_payload
        else:
            response.json.return_value = detail_payload
        return response

    with patch("chatbot.adapters.erpnext.client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = fake_get
        with patch.object(
            client,
            "fetch_latest_invoice_rates",
            return_value={"ITEM-A": {"rate": 100.0, "currency": "MUR", "price_list": "last invoice"}},
        ):
            result = client.probe_invoice_prices(price_list="Standard Selling")
    assert result["ok"] is True
    assert result["item_price_access"] is True
    assert result["rates_found"] == 1
    assert result["sample_item_codes"] == ["ITEM-A"]
    assert "Item Price (Standard Selling): OK" in result["preview"]

