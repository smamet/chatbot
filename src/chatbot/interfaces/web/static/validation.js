(function () {
  function scrollConversationToEnd(container) {
    const thread = container.querySelector(".validation-thread");
    if (!thread) return;
    const scroll = () => {
      thread.scrollTop = thread.scrollHeight;
    };
    requestAnimationFrame(() => {
      scroll();
      requestAnimationFrame(scroll);
    });
  }

  const validationActionFormsSelector =
    ".validation-save-form, form[action*='/validation/'][action*='/approve'], form[action*='/validation/'][action*='/reject']:not(.validation-reject-blacklist-form)";

  const validationUnsavedMessage =
    "Vous avez des modifications non enregistrées dans le brouillon. Quitter cette page sans enregistrer ?";

  let validationDraftDirty = false;
  let validationDraftBaseline = { html: "", subject: "" };

  function captureValidationDraftBaseline(editorEl) {
    const quill = editorEl.querySelector(".validation-quill")?.__quill;
    const subjectEl = editorEl.querySelector(".validation-draft-subject");
    validationDraftBaseline = {
      html: quill ? getQuillHtml(quill) : "",
      subject: subjectEl?.value || "",
    };
    validationDraftDirty = false;
  }

  function updateValidationDraftDirty(editorEl) {
    const quill = editorEl.querySelector(".validation-quill")?.__quill;
    const subjectEl = editorEl.querySelector(".validation-draft-subject");
    if (!quill) return;
    const html = getQuillHtml(quill);
    const subject = subjectEl?.value || "";
    validationDraftDirty =
      html !== validationDraftBaseline.html ||
      subject !== validationDraftBaseline.subject;
  }

  function initValidationUnsavedGuard() {
    const editorEl = document.querySelector(".validation-editor");
    if (!editorEl) return;

    const quill = editorEl.querySelector(".validation-quill")?.__quill;
    if (!quill) return;

    captureValidationDraftBaseline(editorEl);

    quill.on("text-change", () => {
      updateValidationDraftDirty(editorEl);
    });

    const subjectEl = editorEl.querySelector(".validation-draft-subject");
    subjectEl?.addEventListener("input", () => {
      updateValidationDraftDirty(editorEl);
    });

    document.querySelectorAll(validationActionFormsSelector).forEach((form) => {
      form.addEventListener("submit", () => {
        syncValidationDraftToForms();
        captureValidationDraftBaseline(editorEl);
      });
    });

    document
      .querySelectorAll(".validation-regenerate-form, .validation-reject-blacklist-form")
      .forEach((form) => {
        form.addEventListener("submit", () => {
          validationDraftDirty = false;
        });
      });

    window.addEventListener("beforeunload", (event) => {
      if (!validationDraftDirty) return;
      event.preventDefault();
      event.returnValue = validationUnsavedMessage;
      return validationUnsavedMessage;
    });
  }

  function getQuillHtml(quill) {
    return typeof quill.getSemanticHTML === "function"
      ? quill.getSemanticHTML()
      : quill.root.innerHTML;
  }

  function syncValidationDraftHtmlToForms() {
    const editorEl = document.querySelector(".validation-editor");
    const quill = editorEl?.querySelector(".validation-quill")?.__quill;
    if (!quill) return;
    const html = getQuillHtml(quill);
    document.querySelectorAll(validationActionFormsSelector).forEach((form) => {
      let hidden = form.querySelector('input[name="draft_html"]');
      if (!hidden) {
        hidden = document.createElement("input");
        hidden.type = "hidden";
        hidden.name = "draft_html";
        form.appendChild(hidden);
      }
      hidden.value = html;
    });
  }

  function syncValidationSubjectToForms() {
    const subjectEl = document.querySelector(".validation-draft-subject");
    if (!subjectEl) return;
    const value = subjectEl.value;
    document
      .querySelectorAll(validationActionFormsSelector)
      .forEach((form) => {
        let hidden = form.querySelector('input[name="draft_subject"]');
        if (hidden && hidden !== subjectEl) {
          hidden.value = value;
          return;
        }
        if (!hidden) {
          hidden = document.createElement("input");
          hidden.type = "hidden";
          hidden.name = "draft_subject";
          form.appendChild(hidden);
        }
        hidden.value = value;
      });
  }

  function syncValidationDraftToForms() {
    syncValidationSubjectToForms();
    syncValidationDraftHtmlToForms();
  }

  function initValidationDraftSync() {
    const hasEditor = Boolean(document.querySelector(".validation-editor"));
    const hasSubject = Boolean(document.querySelector(".validation-draft-subject"));
    if (!hasEditor && !hasSubject) return;
    document.querySelectorAll(validationActionFormsSelector).forEach((form) => {
      form.addEventListener("submit", syncValidationDraftToForms);
    });
  }

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
      quillHost.__quill = quill;

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
        hiddenInput.value = getQuillHtml(quill);
      });
    });
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  function updateReplyAttachmentHint(attachments) {
    const hint = document.getElementById("validation-reply-attachments-hint");
    if (!hint) return;
    const items = attachments || [];
    hint.classList.toggle("is-empty", items.length === 0);

    const countEl = hint.querySelector(".validation-reply-attachments-count");
    const labelEl = hint.querySelector(".validation-reply-attachments-label");
    const namesEl = hint.querySelector(".validation-reply-attachments-names");
    if (countEl) countEl.textContent = String(items.length);
    if (labelEl) {
      if (items.length === 1) {
        labelEl.textContent = "1 attachment will be sent";
      } else if (items.length > 1) {
        labelEl.textContent = `${items.length} attachments will be sent`;
      } else {
        labelEl.textContent = "No attachments";
      }
    }
    if (namesEl) {
      namesEl.innerHTML = "";
      items.forEach((att, index) => {
        if (index > 0) {
          namesEl.appendChild(document.createTextNode(" · "));
        }
        if (att.view_url) {
          const link = document.createElement("a");
          link.href = att.view_url;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = att.filename;
          namesEl.appendChild(link);
        } else {
          namesEl.appendChild(document.createTextNode(att.filename));
        }
      });
    }
  }

  function appendAttachmentName(parent, att) {
    if (att.view_url) {
      const link = document.createElement("a");
      link.href = att.view_url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = att.filename;
      parent.appendChild(link);
    } else {
      parent.textContent = att.filename;
    }
  }

  function renderAttachmentList(listEl, attachments) {
    listEl.innerHTML = "";
    for (const att of attachments) {
      const li = document.createElement("li");
      li.className = "validation-attachments-item";
      if (att.is_quote_pdf) li.classList.add("validation-attachments-item--quote");
      li.dataset.path = att.path;

      const name = document.createElement("span");
      name.className = "validation-attachments-name";
      appendAttachmentName(name, att);

      const size = document.createElement("span");
      size.className = "validation-attachments-size text-muted";
      size.textContent = formatSize(att.size_bytes || 0);

      li.appendChild(name);
      li.appendChild(size);

      if (att.is_quote_pdf) {
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = "Quote PDF";
        li.appendChild(badge);
      }

      if (att.deletable) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "btn-ghost btn-sm validation-attachments-remove";
        btn.textContent = "Remove";
        li.appendChild(btn);
      }

      listEl.appendChild(li);
    }
    updateReplyAttachmentHint(attachments);
  }

  function initAttachmentDropzone(section) {
    const uploadUrl = section.dataset.uploadUrl;
    const dropzone = section.querySelector(".validation-attachments-dropzone");
    const input = section.querySelector(".validation-attachments-input");
    const listEl = section.querySelector(".validation-attachments-list");
    const errorEl = section.querySelector(".validation-attachments-error");
    const browse = section.querySelector(".validation-attachments-browse");
    if (!uploadUrl || !dropzone || !input || !listEl) return;

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
      const formData = new FormData();
      for (const file of fileList) {
        formData.append("files", file);
      }
      const response = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
        credentials: "same-origin",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        showError(payload.detail || "Upload failed");
        return;
      }
      renderAttachmentList(listEl, payload.attachments || []);
    }

    async function removeAttachment(path) {
      showError("");
      const url = `${uploadUrl}?path=${encodeURIComponent(path)}`;
      const response = await fetch(url, {
        method: "DELETE",
        credentials: "same-origin",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        showError(payload.detail || "Remove failed");
        return;
      }
      renderAttachmentList(listEl, payload.attachments || []);
    }

    dropzone.addEventListener("click", (event) => {
      if (event.target.closest(".validation-attachments-remove")) return;
      input.click();
    });

    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });

    if (browse) {
      browse.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        input.click();
      });
    }

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

    listEl.addEventListener("click", (event) => {
      if (event.target.closest("a")) return;
      const btn = event.target.closest(".validation-attachments-remove");
      if (!btn) return;
      const item = btn.closest(".validation-attachments-item");
      if (!item?.dataset.path) return;
      removeAttachment(item.dataset.path);
    });
  }

  function initValidationBodyToggle() {
    if (typeof window.initMessageBodyToggle === "function") {
      window.initMessageBodyToggle(document.querySelector(".validation-detail-page"));
    }
  }

  function initValidationRegenerateConfirm() {
    document.querySelectorAll(".validation-regenerate-form").forEach((form) => {
      form.addEventListener("submit", (event) => {
        const message = form.dataset.confirm || "Continue?";
        if (!window.confirm(message)) {
          event.preventDefault();
        }
      });
    });
  }

  function initValidationRejectBlacklistConfirm() {
    document.querySelectorAll(".validation-reject-blacklist-form").forEach((form) => {
      form.addEventListener("submit", (event) => {
        const message = form.dataset.confirm || "Continue?";
        if (!window.confirm(message)) {
          event.preventDefault();
        }
      });
    });
  }

  function initValidationTranslate() {
    document.querySelectorAll(".validation-editor").forEach((editorEl) => {
      const translateUrl = editorEl.dataset.translateUrl;
      if (!translateUrl) return;

      const errorEl = editorEl.querySelector(".validation-translate-error");
      const subjectEl = editorEl.querySelector(".validation-draft-subject");
      const buttons = editorEl.querySelectorAll(".validation-translate-btn");
      if (!buttons.length) return;

      const showError = (message) => {
        if (!errorEl) return;
        if (message) {
          errorEl.textContent = message;
          errorEl.hidden = false;
        } else {
          errorEl.textContent = "";
          errorEl.hidden = true;
        }
      };

      buttons.forEach((btn) => {
        btn.addEventListener("click", async () => {
          const quillHost = editorEl.querySelector(".validation-quill");
          const quill = quillHost?.__quill;
          if (!quill) {
            showError("Editor not ready.");
            return;
          }

          const targetLang = btn.dataset.target;
          if (targetLang !== "en" && targetLang !== "fr") return;

          showError("");
          const loadingLabel = btn.textContent;
          buttons.forEach((b) => {
            b.disabled = true;
          });
          btn.textContent = "Translating…";

          try {
            const response = await fetch(translateUrl, {
              method: "POST",
              credentials: "same-origin",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                target_lang: targetLang,
                draft_html: getQuillHtml(quill),
                draft_subject: subjectEl?.value || "",
              }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
              showError(payload.detail || "Translation failed.");
              return;
            }
            quill.setContents([]);
            if (payload.draft_html) {
              quill.clipboard.dangerouslyPasteHTML(payload.draft_html);
            }
            if (subjectEl && typeof payload.draft_subject === "string") {
              subjectEl.value = payload.draft_subject;
            }
            syncValidationDraftToForms();
            updateValidationDraftDirty(editorEl);
          } catch {
            showError("Translation failed.");
          } finally {
            buttons.forEach((b) => {
              b.disabled = false;
              if (b === btn) {
                b.textContent = loadingLabel;
              }
            });
          }
        });
      });
    });
  }

  function initDetailPage() {
    const page = document.querySelector(".validation-detail-page");
    if (!page) return;
    initQuillEditors();
    initValidationUnsavedGuard();
    initValidationDraftSync();
    initValidationBodyToggle();
    initValidationRegenerateConfirm();
    initValidationRejectBlacklistConfirm();
    initValidationTranslate();
    if (window.ChatbotMarkdown) {
      window.ChatbotMarkdown.applyMarkdown(page);
    }
    scrollConversationToEnd(page);
    document.querySelectorAll(".validation-attachments").forEach(initAttachmentDropzone);
  }

  initDetailPage();
})();
