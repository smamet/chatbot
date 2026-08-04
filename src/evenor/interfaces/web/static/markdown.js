(function () {
  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdown(text) {
    if (typeof marked === "undefined") {
      return `<pre>${escapeHtml(text)}</pre>`;
    }
    const raw = marked.parse(text, { breaks: true, gfm: true });
    if (typeof DOMPurify !== "undefined") {
      return DOMPurify.sanitize(raw);
    }
    return raw;
  }

  function applyMarkdown(root) {
    const scope = root || document;
    scope.querySelectorAll(".js-md").forEach((el) => {
      const source = el.textContent || "";
      el.innerHTML = renderMarkdown(source);
      el.classList.remove("js-md");
    });
  }

  window.ChatbotMarkdown = { escapeHtml, renderMarkdown, applyMarkdown };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => applyMarkdown());
  } else {
    applyMarkdown();
  }
})();
