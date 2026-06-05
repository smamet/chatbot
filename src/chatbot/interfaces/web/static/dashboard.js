(function () {
  const SKIP_FORM_IDS = new Set(["chat-form"]);

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
})();
