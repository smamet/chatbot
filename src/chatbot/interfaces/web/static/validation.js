(function () {
  function scrollConversationToEnd(panel) {
    const thread = panel.querySelector(".validation-thread");
    if (!thread) return;
    const scroll = () => {
      thread.scrollTop = thread.scrollHeight;
    };
    requestAnimationFrame(() => {
      scroll();
      requestAnimationFrame(scroll);
    });
  }

  function toggleRow(row) {
    const panelId = row.dataset.panel;
    if (!panelId) return;
    const panel = document.getElementById(panelId);
    if (!panel) return;

    const willOpen = panel.hidden;
    panel.hidden = !willOpen;
    row.classList.toggle("is-open", willOpen);
    row.setAttribute("aria-expanded", willOpen ? "true" : "false");
    if (willOpen) {
      if (window.ChatbotMarkdown) {
        window.ChatbotMarkdown.applyMarkdown(panel);
      }
      scrollConversationToEnd(panel);
    }
  }

  document.querySelectorAll(".validation-row").forEach((row) => {
    row.addEventListener("click", (event) => {
      if (event.target.closest("a, button, input, select, textarea, form, label")) {
        return;
      }
      toggleRow(row);
    });

    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggleRow(row);
    });
  });

  function initQuillEditors() {
    if (typeof Quill === "undefined") return;

    document.querySelectorAll(".validation-editor").forEach((editorEl) => {
      const quillHost = editorEl.querySelector(".validation-quill");
      const source = editorEl.querySelector(".validation-draft-html-source");
      const saveForm = editorEl.querySelector(".validation-save-form");
      const hiddenInput = editorEl.querySelector(".validation-draft-html-input");
      if (!quillHost || !source || !saveForm || !hiddenInput) return;

      const quill = new Quill(quillHost, {
        theme: "snow",
        modules: {
          toolbar: [
            [{ header: [1, 2, 3, false] }],
            ["bold", "italic", "underline"],
            [{ color: [] }, { background: [] }],
            [{ list: "ordered" }, { list: "bullet" }],
            ["link"],
            ["clean"],
          ],
        },
      });

      let initialHtml = "";
      if (source.tagName === "SCRIPT") {
        try {
          initialHtml = JSON.parse(source.textContent || '""');
        } catch {
          initialHtml = "";
        }
      } else {
        initialHtml = source.value.trim();
      }
      if (initialHtml) {
        quill.clipboard.dangerouslyPasteHTML(initialHtml);
      }

      saveForm.addEventListener("submit", () => {
        hiddenInput.value =
          typeof quill.getSemanticHTML === "function"
            ? quill.getSemanticHTML()
            : quill.root.innerHTML;
      });
    });
  }

  initQuillEditors();
})();
