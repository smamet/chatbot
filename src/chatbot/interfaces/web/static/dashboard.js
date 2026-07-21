(function () {
  const SKIP_FORM_IDS = new Set(["chat-form", "chat-reset-form"]);

  function setPageLoading(on) {
    document.getElementById("page-loader")?.classList.toggle("active", on);
  }

  function setButtonLoading(btn, label) {
    if (!btn || btn.dataset.loadingActive === "1") return;
    btn.dataset.loadingActive = "1";
    btn.dataset.originalHtml = btn.innerHTML;
    btn.disabled = true;
    btn.classList.add("is-loading");
    btn.setAttribute("aria-busy", "true");
    const text = label || btn.dataset.loading || "Loading…";
    btn.innerHTML = `<span class="btn-spinner" aria-hidden="true"></span><span>${text}</span>`;
  }

  function guardForm(form) {
    form.classList.add("is-submitting");
    form.querySelectorAll('button[type="submit"], input[type="submit"]').forEach((btn) => {
      if (btn.disabled) return;
      setButtonLoading(btn, btn.dataset.loading);
    });
  }

  document.querySelectorAll("form").forEach((form) => {
    if (SKIP_FORM_IDS.has(form.id) || form.classList.contains("no-loader")) return;

    form.addEventListener("submit", (event) => {
      if (form.classList.contains("is-submitting")) {
        event.preventDefault();
        return;
      }
      const submitter =
        form.querySelector('button[type="submit"], input[type="submit"]') || null;
      if (!submitter || submitter.disabled) return;
      setPageLoading(true);
      guardForm(form);
    });
  });

  window.addEventListener("pageshow", (event) => {
    if (event.persisted) setPageLoading(false);
  });

  const integrationEditor = document.querySelector(".integration-editor");
  const integrationSlug =
    integrationEditor?.querySelector("[data-slug]")?.dataset.slug || "";
  const erpnextTestStorageKey = integrationSlug
    ? `chatbot:erpnext-test:${integrationSlug}`
    : "chatbot:erpnext-test";
  const erpnextTestDefaults = {
    email: "customer@example.com",
    phone: "+33612345678",
    company: "",
    customerName: "",
    item: "",
    qty: "1",
    notes: "",
  };
  const erpnextTestFieldGroups = [
    {
      key: "email",
      selectors: [
        ".integration-test-email",
        ".integration-customer-email",
        ".integration-quotation-email",
      ],
    },
    {
      key: "phone",
      selectors: [
        ".integration-test-phone",
        ".integration-customer-phone",
        ".integration-quotation-phone",
      ],
    },
    {
      key: "company",
      selectors: [".integration-customer-company", ".integration-quotation-company"],
    },
    { key: "customerName", selectors: [".integration-customer-name"] },
    { key: "item", selectors: [".integration-quotation-item"] },
    { key: "qty", selectors: [".integration-quotation-qty"] },
    { key: "notes", selectors: [".integration-quotation-notes"] },
  ];

  function loadErpnextTestValues() {
    try {
      const raw = localStorage.getItem(erpnextTestStorageKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        return { ...erpnextTestDefaults, ...parsed };
      }
    } catch (_) {
      /* ignore corrupt storage */
    }
    return { ...erpnextTestDefaults };
  }

  function saveErpnextTestValues(values) {
    try {
      localStorage.setItem(erpnextTestStorageKey, JSON.stringify(values));
    } catch (_) {
      /* ignore quota / private mode */
    }
  }

  function applyErpnextTestValues(values) {
    if (!integrationEditor) return;
    for (const group of erpnextTestFieldGroups) {
      const value = values[group.key] ?? erpnextTestDefaults[group.key];
      for (const selector of group.selectors) {
        integrationEditor.querySelectorAll(selector).forEach((el) => {
          el.value = value;
        });
      }
    }
  }

  function bindErpnextTestPersistence() {
    if (!integrationEditor) return;
    for (const group of erpnextTestFieldGroups) {
      for (const selector of group.selectors) {
        integrationEditor.querySelectorAll(selector).forEach((el) => {
          el.addEventListener("input", () => {
            const stored = loadErpnextTestValues();
            stored[group.key] = el.value;
            saveErpnextTestValues(stored);
            applyErpnextTestValues(stored);
          });
        });
      }
    }
  }

  if (integrationEditor) {
    applyErpnextTestValues(loadErpnextTestValues());
    bindErpnextTestPersistence();
  }

  document.querySelectorAll(".integration-test-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const form = document.getElementById("integration-form");
      const resultEl = document.querySelector(".integration-test-result");
      if (!form || !resultEl) return;
      const slug = btn.dataset.slug;
      const fd = new FormData(form);
      const card = form.closest(".integration-editor");
      const email = card?.querySelector(".integration-test-email")?.value?.trim() || "";
      const phone = card?.querySelector(".integration-test-phone")?.value?.trim() || "";
      fd.set("test_email", email);
      fd.set("test_phone", phone);
      btn.disabled = true;
      resultEl.hidden = false;
      resultEl.className = "integration-test-result";
      resultEl.textContent = "Testing…";
      try {
        const res = await fetch(`/dashboard/bots/${slug}/integrations/test`, {
          method: "POST",
          body: fd,
          credentials: "same-origin",
        });
        const data = await res.json();
        resultEl.classList.add(data.ok ? "ok" : "err");
        resultEl.textContent = data.preview || data.message || data.error || "Done";
        if (data.error && !data.ok) {
          resultEl.textContent = `${data.message}\n${data.error}`;
        }
      } catch (err) {
        resultEl.classList.add("err");
        resultEl.textContent = String(err);
      } finally {
        btn.disabled = false;
      }
    });
  });

  document.querySelectorAll(".integration-customer-create-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const form = document.getElementById("integration-form");
      const resultEl = document.querySelector(".integration-customer-create-result");
      if (!form || !resultEl) return;
      const slug = btn.dataset.slug;
      const fd = new FormData(form);
      const card = form.closest(".integration-editor");
      const customerName = card?.querySelector(".integration-customer-name")?.value?.trim() || "";
      const companyName = card?.querySelector(".integration-customer-company")?.value?.trim() || "";
      const email = card?.querySelector(".integration-customer-email")?.value?.trim() || "";
      const phone = card?.querySelector(".integration-customer-phone")?.value?.trim() || "";
      fd.set("customer_name", customerName);
      fd.set("company_name", companyName);
      fd.set("test_email", email);
      fd.set("test_phone", phone);
      btn.disabled = true;
      resultEl.hidden = false;
      resultEl.className = "integration-customer-create-result integration-test-result";
      resultEl.textContent = "Creating…";
      try {
        const res = await fetch(`/dashboard/bots/${slug}/integrations/erpnext/create-customer`, {
          method: "POST",
          body: fd,
          credentials: "same-origin",
        });
        const data = await res.json();
        resultEl.classList.add(data.ok ? "ok" : "err");
        resultEl.textContent = data.message || data.error || "Done";
        if (data.error && !data.ok) {
          resultEl.textContent = `${data.message}\n${data.error}`;
        }
      } catch (err) {
        resultEl.classList.add("err");
        resultEl.textContent = String(err);
      } finally {
        btn.disabled = false;
      }
    });
  });

  document.querySelectorAll(".integration-quotation-create-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const form = document.getElementById("integration-form");
      const resultEl = document.querySelector(".integration-quotation-create-result");
      const logEl = resultEl?.querySelector(".integration-quotation-create-log");
      const pdfBtn = resultEl?.querySelector(".integration-quotation-pdf-btn");
      if (!form || !resultEl || !logEl || !pdfBtn) return;
      const slug = btn.dataset.slug;
      const fd = new FormData(form);
      const card = form.closest(".integration-editor");
      const email = card?.querySelector(".integration-quotation-email")?.value?.trim() || "";
      const phone = card?.querySelector(".integration-quotation-phone")?.value?.trim() || "";
      const companyName = card?.querySelector(".integration-quotation-company")?.value?.trim() || "";
      const itemCode = card?.querySelector(".integration-quotation-item")?.value?.trim() || "";
      const qty = card?.querySelector(".integration-quotation-qty")?.value?.trim() || "1";
      const notes = card?.querySelector(".integration-quotation-notes")?.value?.trim() || "";
      fd.set("test_email", email);
      fd.set("test_phone", phone);
      fd.set("company_name", companyName);
      fd.set("item_code", itemCode);
      fd.set("qty", qty);
      fd.set("notes", notes);
      fd.set("stream", "1");
      btn.disabled = true;
      resultEl.hidden = false;
      resultEl.className = "integration-quotation-create-result integration-test-result";
      logEl.textContent = "";
      pdfBtn.hidden = true;
      pdfBtn.removeAttribute("href");

      const appendLog = (message) => {
        logEl.textContent = logEl.textContent ? `${logEl.textContent}\n${message}` : message;
        logEl.scrollTop = logEl.scrollHeight;
      };

      const applyDone = (data) => {
        resultEl.classList.add(data.ok ? "ok" : "err");
        if (data.ok && data.pdf_url) {
          appendLog(data.message || "Done");
          pdfBtn.href = data.pdf_url;
          if (data.pdf_filename) {
            pdfBtn.setAttribute("download", data.pdf_filename);
          }
          pdfBtn.hidden = false;
        } else if (data.ok && data.pdf_warning) {
          appendLog(`${data.message || "Done"}\n${data.pdf_warning}`);
        } else {
          appendLog(data.message || data.error || "Done");
          if (data.error && !data.ok) {
            appendLog(data.error);
          }
        }
      };

      try {
        const res = await fetch(`/dashboard/bots/${slug}/integrations/erpnext/create-quotation`, {
          method: "POST",
          body: fd,
          credentials: "same-origin",
        });
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("application/x-ndjson") && res.body) {
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const line of lines) {
              if (!line.trim()) continue;
              const event = JSON.parse(line);
              if (event.event === "log" && event.message) {
                appendLog(event.message);
              } else if (event.event === "done") {
                applyDone(event);
              }
            }
          }
          if (buffer.trim()) {
            const event = JSON.parse(buffer);
            if (event.event === "log" && event.message) {
              appendLog(event.message);
            } else if (event.event === "done") {
              applyDone(event);
            }
          }
        } else {
          const data = await res.json();
          applyDone(data);
        }
      } catch (err) {
        resultEl.classList.add("err");
        appendLog(String(err));
      } finally {
        btn.disabled = false;
      }
    });
  });

  document.querySelectorAll(".integration-catalog-sync-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const resultEl = document.querySelector(".integration-catalog-sync-result");
      if (!resultEl) return;
      const slug = btn.dataset.slug;
      btn.disabled = true;
      resultEl.hidden = false;
      resultEl.className = "integration-catalog-sync-result integration-test-result";
      resultEl.textContent = "Starting catalog sync…";
      try {
        const res = await fetch(`/dashboard/bots/${slug}/integrations/erpnext/sync-catalog`, {
          method: "POST",
          credentials: "same-origin",
        });
        const data = await res.json();
        resultEl.classList.add(data.ok ? "ok" : "err");
        resultEl.textContent = data.message || data.error || "Done";
      } catch (err) {
        resultEl.classList.add("err");
        resultEl.textContent = String(err);
      } finally {
        btn.disabled = false;
      }
    });
  });

  document.querySelectorAll(".integration-invoice-price-test-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const resultEl = document.querySelector(".integration-invoice-price-test-result");
      if (!resultEl) return;
      const slug = btn.dataset.slug;
      btn.disabled = true;
      resultEl.hidden = false;
      resultEl.className = "integration-invoice-price-test-result integration-test-result";
      resultEl.textContent = "Testing catalog price access…";
      try {
        const res = await fetch(
          `/dashboard/bots/${slug}/integrations/erpnext/test-invoice-prices`,
          { method: "POST", credentials: "same-origin" },
        );
        const data = await res.json().catch(() => ({}));
        resultEl.classList.add(data.ok ? "ok" : "err");
        const lines = [data.preview || data.message || ""];
        if (data.item_price_error && !data.item_price_access) {
          lines.push(String(data.item_price_error));
        }
        if (data.error && !data.ok) {
          lines.push(String(data.error));
        }
        if (data.item_price_http_status) {
          lines.push(`Item Price HTTP ${data.item_price_http_status}`);
        }
        if (data.http_status) {
          lines.push(`Sales Invoice HTTP ${data.http_status}`);
        }
        resultEl.textContent = lines.filter(Boolean).join("\n");
      } catch (err) {
        resultEl.classList.add("err");
        resultEl.textContent = String(err);
      } finally {
        btn.disabled = false;
      }
    });
  });

  document.querySelectorAll(".integration-catalog-purge-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!confirm("Delete all catalog markdown files and remove them from RAG?")) return;
      const resultEl = document.querySelector(".integration-catalog-sync-result");
      if (!resultEl) return;
      const slug = btn.dataset.slug;
      btn.disabled = true;
      resultEl.hidden = false;
      resultEl.className = "integration-catalog-sync-result integration-test-result";
      resultEl.textContent = "Purging catalog…";
      try {
        const res = await fetch(`/dashboard/bots/${slug}/integrations/erpnext/purge-catalog`, {
          method: "POST",
          credentials: "same-origin",
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          resultEl.classList.add("err");
          resultEl.textContent = data.message || data.detail || data.error || `HTTP ${res.status}`;
          return;
        }
        resultEl.classList.add(data.ok ? "ok" : "err");
        resultEl.textContent = data.message || data.error || "Done";
        if (data.ok) {
          const panel = document.getElementById("integration-catalog-sync-panel");
          panel?.querySelector(".integration-catalog-last-sync")?.replaceChildren(
            document.createTextNode("—"),
          );
          panel?.querySelector(".integration-catalog-item-count")?.replaceChildren(
            document.createTextNode("—"),
          );
          panel?.querySelector(".integration-catalog-last-error")?.replaceChildren(
            document.createTextNode("—"),
          );
        }
      } catch (err) {
        resultEl.classList.add("err");
        resultEl.textContent = String(err);
      } finally {
        btn.disabled = false;
      }
    });
  });

  document.querySelectorAll(".validation-inbox-row[data-href]").forEach((row) => {
    row.addEventListener("click", (event) => {
      if (event.target.closest("a, button, input, select, textarea, label")) {
        return;
      }
      window.location.assign(row.dataset.href);
    });
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      window.location.assign(row.dataset.href);
    });
  });

  const bulkForm = document.getElementById("validation-bulk-form");
  const bulkCountEl = document.getElementById("validation-bulk-count");
  const selectAllEl = document.getElementById("validation-select-all");
  const inboxSelectEls = () =>
    Array.from(document.querySelectorAll(".validation-inbox-select"));

  const selectedInboxIds = () =>
    inboxSelectEls()
      .filter((el) => el.checked)
      .map((el) => el.value);

  const syncValidationBulkBar = () => {
    if (!bulkForm || !bulkCountEl) return;
    const ids = selectedInboxIds();
    const count = ids.length;
    bulkCountEl.textContent =
      count === 1 ? "1 selected" : `${count} selected`;
    bulkForm.hidden = count === 0;
    if (selectAllEl) {
      const boxes = inboxSelectEls();
      selectAllEl.checked = boxes.length > 0 && boxes.every((el) => el.checked);
      selectAllEl.indeterminate =
        count > 0 && count < boxes.length;
    }
  };

  inboxSelectEls().forEach((el) => {
    el.addEventListener("change", syncValidationBulkBar);
  });

  if (selectAllEl) {
    selectAllEl.addEventListener("change", () => {
      const checked = selectAllEl.checked;
      inboxSelectEls().forEach((el) => {
        el.checked = checked;
      });
      syncValidationBulkBar();
    });
  }

  if (bulkForm) {
    bulkForm.addEventListener("submit", (event) => {
      const ids = selectedInboxIds();
      if (!ids.length) {
        event.preventDefault();
        return;
      }
      const template =
        bulkForm.dataset.confirmTemplate ||
        "Reject {count} selected email(s)? This cannot be undone.";
      const message = template.replace("{count}", String(ids.length));
      if (!window.confirm(message)) {
        event.preventDefault();
        return;
      }
      bulkForm
        .querySelectorAll('input[name="reply_ids"]')
        .forEach((el) => el.remove());
      ids.forEach((id) => {
        const input = document.createElement("input");
        input.type = "hidden";
        input.name = "reply_ids";
        input.value = id;
        bulkForm.appendChild(input);
      });
    });
  }

  const localInboxDateFormatter = new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
  const localInboxTimeFormatter = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  const localDateTimeFormatter = new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });

  function localDateKey(dt) {
    return `${dt.getFullYear()}-${dt.getMonth()}-${dt.getDate()}`;
  }

  function renderValidationInboxLocalTimes() {
    const tbody = document.querySelector(".validation-table--inbox tbody");
    if (!tbody) return;

    tbody.querySelectorAll(".validation-inbox-date-row").forEach((row) => row.remove());

    const colspan = Number.parseInt(tbody.dataset.inboxColspan || "7", 10);
    let prevDateKey = null;

    tbody.querySelectorAll(".validation-inbox-row").forEach((row) => {
      const utc = row.dataset.utc;
      if (!utc) return;
      const dt = new Date(utc);
      if (Number.isNaN(dt.getTime())) return;

      const dateKey = localDateKey(dt);
      if (dateKey !== prevDateKey) {
        const header = document.createElement("tr");
        header.className = "validation-inbox-date-row";
        const cell = document.createElement("td");
        cell.colSpan = colspan;
        cell.textContent = localInboxDateFormatter.format(dt);
        header.appendChild(cell);
        row.before(header);
        prevDateKey = dateKey;
      }

      const timeEl = row.querySelector(".js-local-time");
      if (timeEl) timeEl.textContent = localInboxTimeFormatter.format(dt);
    });
  }

  function renderLocalDateTimes(root = document) {
    root.querySelectorAll("time.js-local-datetime[datetime]").forEach((el) => {
      const dt = new Date(el.dateTime);
      if (!Number.isNaN(dt.getTime())) {
        el.textContent = localDateTimeFormatter.format(dt);
      }
    });
  }

  syncValidationBulkBar();
  renderValidationInboxLocalTimes();
  renderLocalDateTimes();

  document.querySelectorAll(".validation-row[data-panel]").forEach((row) => {
    const panel = document.getElementById(row.dataset.panel);
    if (!panel) return;
    const toggle = () => {
      const open = row.classList.toggle("is-open");
      panel.hidden = !open;
      row.setAttribute("aria-expanded", open ? "true" : "false");
    };
    row.addEventListener("click", (event) => {
      if (event.target.closest("a, button, input, form")) return;
      toggle();
    });
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggle();
    });
  });

  function initDocUploadDropzone() {
    const form = document.getElementById("doc-upload-form");
    if (!form) return;

    const dropzone = form.querySelector(".validation-attachments-dropzone");
    const input = form.querySelector(".validation-attachments-input");
    const browse = form.querySelector(".validation-attachments-browse");
    const errorEl = form.querySelector(".doc-upload-error");
    if (!dropzone || !input) return;

    function showError(message) {
      if (!errorEl) return;
      if (message) {
        errorEl.textContent = message;
        errorEl.hidden = false;
      } else {
        errorEl.textContent = "";
        errorEl.hidden = true;
      }
    }

    async function uploadFiles(fileList) {
      if (!fileList || !fileList.length) return;
      showError("");
      setPageLoading(true);
      const formData = new FormData();
      for (const file of fileList) {
        formData.append("files", file);
      }
      try {
        const response = await fetch(form.action, {
          method: "POST",
          body: formData,
          credentials: "same-origin",
          redirect: "manual",
        });
        if (response.status === 303) {
          const location = response.headers.get("Location");
          window.location.href = location || `${form.action}?tab=documents`;
          return;
        }
        const detail = await response.text();
        showError(detail.slice(0, 200) || "Upload failed");
      } catch (_err) {
        showError("Upload failed");
      } finally {
        setPageLoading(false);
      }
    }

    dropzone.addEventListener("click", (event) => {
      if (event.target.closest(".validation-attachments-browse")) return;
      input.click();
    });

    browse?.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      input.click();
    });

    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });

    input.addEventListener("change", () => {
      uploadFiles(input.files);
      input.value = "";
    });

    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("is-dragover");
    });

    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
      uploadFiles(event.dataTransfer?.files);
    });
  }

  initDocUploadDropzone();

  document.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-checkbox-set]");
    if (!(btn instanceof HTMLElement)) return;
    const panel = btn.closest(".checkbox-panel");
    if (!panel) return;
    const checked = btn.dataset.checkboxSet === "all";
    panel.querySelectorAll('input[type="checkbox"]').forEach((input) => {
      input.checked = checked;
    });
  });

  const connectorTestBtn = document.getElementById("connector-test-btn");
  if (connectorTestBtn) {
    connectorTestBtn.addEventListener("click", async () => {
      const form = document.getElementById("connector-form");
      const resultEl = document.querySelector("#connector-test-panel .connector-test-result");
      if (!form || !resultEl) return;
      const slug = connectorTestBtn.dataset.slug;
      const fd = new FormData(form);
      connectorTestBtn.disabled = true;
      resultEl.hidden = false;
      resultEl.className = "connector-test-result integration-test-result";
      resultEl.textContent = "Testing…";
      try {
        const res = await fetch(`/dashboard/bots/${slug}/connectors/test`, {
          method: "POST",
          body: fd,
          credentials: "same-origin",
        });
        const data = await res.json();
        resultEl.classList.add(data.ok ? "ok" : "err");
        if (data.error && !data.ok) {
          resultEl.textContent = `${data.message}\n${data.error}`;
        } else {
          resultEl.textContent = data.message || "Done";
        }
      } catch (err) {
        resultEl.classList.add("err");
        resultEl.textContent = String(err);
      } finally {
        connectorTestBtn.disabled = false;
      }
    });
  }

  const connectorOrdersBtn = document.getElementById("connector-test-orders-btn");
  const igOrderDialog = document.getElementById("ig-order-test-dialog");
  const igOrderConfirm = document.getElementById("ig-order-test-confirm");
  const igOrderCancel = document.getElementById("ig-order-test-cancel");

  function closeIgOrderDialog() {
    if (!igOrderDialog) return;
    if (typeof igOrderDialog.close === "function") {
      igOrderDialog.close();
    } else {
      igOrderDialog.removeAttribute("open");
    }
  }

  async function runIgWorkingOrderTest() {
    const form = document.getElementById("connector-form");
    const panel = document.getElementById("connector-test-panel");
    const resultEl = document.querySelector("#connector-test-panel .connector-test-result");
    if (!form || !resultEl || !connectorOrdersBtn) return;
    if (panel) panel.open = true;
    const slug = connectorOrdersBtn.dataset.slug;
    const fd = new FormData(form);
    connectorOrdersBtn.disabled = true;
    if (igOrderConfirm) igOrderConfirm.disabled = true;
    resultEl.hidden = false;
    resultEl.className = "connector-test-result integration-test-result";
    resultEl.textContent =
      "Placing working orders on IG DEMO… watch Working Orders for ~15s, then they cancel.";
    resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    try {
      const res = await fetch(`/dashboard/bots/${slug}/connectors/test-orders`, {
        method: "POST",
        body: fd,
        credentials: "same-origin",
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok && !data.message) {
        throw new Error(`HTTP ${res.status}`);
      }
      resultEl.classList.add(data.ok ? "ok" : "err");
      if (data.error && !data.ok) {
        resultEl.textContent = `${data.message || "Failed"}\n${data.error}`;
      } else {
        resultEl.textContent = data.message || "Done";
      }
    } catch (err) {
      resultEl.classList.add("err");
      resultEl.textContent = String(err);
    } finally {
      connectorOrdersBtn.disabled = false;
      if (igOrderConfirm) igOrderConfirm.disabled = false;
    }
  }

  if (connectorOrdersBtn && igOrderDialog) {
    connectorOrdersBtn.addEventListener("click", () => {
      if (typeof igOrderDialog.showModal === "function") {
        igOrderDialog.showModal();
      } else {
        igOrderDialog.setAttribute("open", "");
      }
    });
    igOrderCancel?.addEventListener("click", closeIgOrderDialog);
    igOrderConfirm?.addEventListener("click", async () => {
      closeIgOrderDialog();
      await runIgWorkingOrderTest();
    });
    igOrderDialog.addEventListener("click", (e) => {
      if (e.target === igOrderDialog) closeIgOrderDialog();
    });
  }
})();
