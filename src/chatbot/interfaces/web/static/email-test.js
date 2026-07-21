(function () {
  const panel = document.getElementById("email-test-panel");
  if (!panel) return;

  const sendUrl = panel.dataset.sendUrl;
  const pollUrl = panel.dataset.pollUrl;
  const validationUrl = panel.dataset.validationUrl;
  const pollHint = panel.dataset.pollHint || "30";
  const form = document.getElementById("email-test-form");
  const statusEl = document.getElementById("email-test-status");
  const sendBtn = document.getElementById("email-test-send");
  const pollBtn = document.getElementById("email-test-poll");

  function showStatus(text, isError) {
    if (!statusEl) return;
    statusEl.hidden = false;
    statusEl.textContent = text;
    statusEl.classList.toggle("text-error", !!isError);
  }

  function setBusy(on) {
    sendBtn.disabled = on;
    pollBtn.disabled = on;
  }

  async function postForm(url, formData) {
    const res = await fetch(url, { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || res.statusText || "Request failed");
    }
    return data;
  }

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    setBusy(true);
    showStatus("Sending…", false);
    try {
      const fd = new FormData(form);
      const data = await postForm(sendUrl, fd);
      showStatus(
        `${data.message || "Sent."} Poll interval ~${data.poll_hint_seconds || pollHint}s. Open Validation after processing.`,
        false
      );
    } catch (err) {
      showStatus(err.message || String(err), true);
    } finally {
      setBusy(false);
    }
  });

  pollBtn?.addEventListener("click", async () => {
    setBusy(true);
    showStatus("Processing inbox…", false);
    try {
      const fd = new FormData();
      const data = await postForm(pollUrl, fd);
      const link = validationUrl ? ` <a href="${validationUrl}">Open Validation</a>` : "";
      showStatus(`${data.message || "Done."}${link}`, false);
      if (statusEl && validationUrl) {
        statusEl.innerHTML = `${data.message || "Done."} <a href="${validationUrl}">Open Validation</a>`;
      }
    } catch (err) {
      showStatus(err.message || String(err), true);
    } finally {
      setBusy(false);
    }
  });
})();
