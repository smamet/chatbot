(function () {
  const panel = document.getElementById("chat-panel");
  if (!panel) return;

  const sendUrl = panel.dataset.sendUrl;
  const validationUrl = panel.dataset.validationUrl || "";
  const requireIdentity = panel.dataset.requireIdentity === "true";
  const thread = document.getElementById("chat-thread");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const stopBtn = document.getElementById("chat-stop");
  const initialEl = document.getElementById("chat-initial");
  const hookStatusEl = document.getElementById("chat-hook-status");
  const devMode = panel.dataset.devMode === "true";
  const sessionLabelEl = document.getElementById("chat-session-label");
  const testEmailInput = document.getElementById("chat-test-email");
  const testPhoneInput = document.getElementById("chat-test-phone");
  const resetEmailInput = document.getElementById("chat-reset-email");
  const resetPhoneInput = document.getElementById("chat-reset-phone");
  const resetTestSessionInput = document.getElementById("chat-reset-test-session");
  const resetForm = document.getElementById("chat-reset-form");
  const botSlug = panel.dataset.botSlug || "";
  const identityStorageKey = botSlug ? `chatbot:test-chat:${botSlug}` : "chatbot:test-chat";
  const testSessionStorageKey = botSlug ? `chatbot:test-session:${botSlug}` : "chatbot:test-session";

  let abortController = null;
  let loading = false;

  const renderMarkdown = (text) =>
    window.ChatbotMarkdown?.renderMarkdown(text) ?? text;

  function identityValues() {
    return {
      email: testEmailInput?.value?.trim() || "",
      phone: testPhoneInput?.value?.trim() || "",
    };
  }

  function saveAnonymousSessionId(id) {
    if (!id?.startsWith("test:")) return;
    try {
      localStorage.setItem(testSessionStorageKey, id);
    } catch (_) {
      /* ignore quota / private mode */
    }
  }

  function getOrCreateAnonymousSessionId() {
    const urlSession = (panel.dataset.chatTestSession || "").trim();
    if (urlSession.startsWith("test:")) {
      saveAnonymousSessionId(urlSession);
      return urlSession;
    }
    try {
      const stored = localStorage.getItem(testSessionStorageKey);
      if (stored?.startsWith("test:")) return stored;
    } catch (_) {
      /* ignore corrupt storage */
    }
    const id = `test:${crypto.randomUUID()}`;
    saveAnonymousSessionId(id);
    return id;
  }

  function anonymousTestSessionForSend() {
    const { email, phone } = identityValues();
    if (email || phone) return "";
    return getOrCreateAnonymousSessionId();
  }

  function loadStoredIdentity() {
    try {
      const raw = localStorage.getItem(identityStorageKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        return {
          email: String(parsed.email ?? "").trim(),
          phone: String(parsed.phone ?? "").trim(),
        };
      }
    } catch (_) {
      /* ignore corrupt storage */
    }
    return { email: "", phone: "" };
  }

  function clearStoredIdentity() {
    try {
      localStorage.removeItem(identityStorageKey);
      localStorage.removeItem(testSessionStorageKey);
    } catch (_) {
      /* ignore quota / private mode */
    }
  }

  function saveStoredIdentity(email, phone) {
    try {
      localStorage.setItem(identityStorageKey, JSON.stringify({ email, phone }));
    } catch (_) {
      /* ignore quota / private mode */
    }
  }

  function initIdentityFields() {
    if (!testEmailInput && !testPhoneInput) return;
    const urlEmail = testEmailInput?.value?.trim() || "";
    const urlPhone = testPhoneInput?.value?.trim() || "";
    if (urlEmail || urlPhone) {
      saveStoredIdentity(urlEmail, urlPhone);
    } else {
      const { email, phone } = loadStoredIdentity();
      if (testEmailInput) testEmailInput.value = email;
      if (testPhoneInput) testPhoneInput.value = phone;
    }
    syncIdentityFields();
  }

  function syncIdentityFields() {
    const { email, phone } = identityValues();
    const testSession = email || phone ? "" : getOrCreateAnonymousSessionId();
    if (resetEmailInput) resetEmailInput.value = email;
    if (resetPhoneInput) resetPhoneInput.value = phone;
    if (resetTestSessionInput) resetTestSessionInput.value = testSession;
    if (sessionLabelEl) {
      if (email && phone) sessionLabelEl.textContent = `email:${email.toLowerCase()}|${phone}`;
      else if (email) sessionLabelEl.textContent = `email:${email.toLowerCase()}`;
      else if (phone) sessionLabelEl.textContent = `whatsapp:${phone}`;
      else if (testSession) sessionLabelEl.textContent = testSession;
      else if (requireIdentity) sessionLabelEl.textContent = "(set test email or phone)";
      else sessionLabelEl.textContent = sessionLabelEl.dataset.defaultSession || sessionLabelEl.textContent;
    }
  }

  function scrollThread() {
    if (thread) thread.scrollTop = thread.scrollHeight;
  }

  function formatCharCount(n) {
    const num = Number(n) || 0;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}k chars`;
    return `${num} chars`;
  }

  function formatContextSizeLabel(ctx) {
    if (!ctx) return "";
    const parts = [`RAG: ${ctx.rag_chunks} chunks, ${formatCharCount(ctx.rag_chars)}`];
    if (ctx.customer_chars) {
      parts.push(`Customer: ${formatCharCount(ctx.customer_chars)}`);
    }
    parts.push(`System: ${formatCharCount(ctx.system_chars)}`);
    return parts.join(" · ");
  }

  function appendMessage(role, content, { markdown = false, contextSize = null } = {}) {
    const turn = document.createElement("div");
    turn.className = `msg-turn msg-turn-${role}`;

    const div = document.createElement("div");
    div.className = `msg msg-${role}`;
    const label = document.createElement("strong");
    label.textContent = role;
    div.appendChild(label);
    const body = document.createElement("div");
    body.className = "msg-body";
    if (markdown && role === "assistant") {
      body.innerHTML = renderMarkdown(content);
    } else {
      body.textContent = content;
    }
    div.appendChild(body);
    turn.appendChild(div);

    if (devMode && contextSize && role === "assistant") {
      const debug = document.createElement("p");
      debug.className = "msg-context-debug";
      debug.textContent = formatContextSizeLabel(contextSize);
      turn.appendChild(debug);
    }

    thread.appendChild(turn);
    scrollThread();
    return turn;
  }

  function setPageLoading(on) {
    document.getElementById("page-loader")?.classList.toggle("active", on);
  }

  function setLoading(on) {
    loading = on;
    input.disabled = on;
    sendBtn.disabled = on;
    stopBtn.hidden = !on;
    sendBtn.hidden = on;
    setPageLoading(on);
  }

  function showTyping() {
    const turn = document.createElement("div");
    turn.className = "msg-turn msg-turn-assistant";
    turn.id = "chat-typing";
    turn.innerHTML =
      '<div class="msg msg-assistant msg-loading"><strong>assistant</strong><div class="msg-body"><span class="typing-dots"><span></span><span></span><span></span></span></div></div>';
    thread.appendChild(turn);
    scrollThread();
  }

  function hideTyping() {
    document.getElementById("chat-typing")?.remove();
  }

  function showHookStatus(data) {
    if (!hookStatusEl) return;
    if (!data || (!data.pdf_url && !data.queued && !data.hook_type && !data.message)) {
      hookStatusEl.hidden = true;
      hookStatusEl.textContent = "";
      return;
    }
    hookStatusEl.hidden = false;
    if (data.pdf_url) {
      const label = data.pdf_filename || "Download PDF";
      hookStatusEl.innerHTML = `${data.message || "Quotation available."} <a class="btn-secondary btn-sm" href="${data.pdf_url}" download>${label}</a>`;
      if (data.pdf_warning) {
        hookStatusEl.innerHTML += ` <span class="text-muted">${data.pdf_warning}</span>`;
      }
      return;
    }
    if (data.queued && validationUrl) {
      hookStatusEl.innerHTML = `${data.message || "Quote queued."} <a href="${validationUrl}">Open Validation</a>`;
      return;
    }
    if (data.hook_type || data.message) {
      hookStatusEl.textContent = data.message || `Hook detected: ${data.hook_type}`;
      return;
    }
    hookStatusEl.hidden = true;
    hookStatusEl.textContent = "";
  }

  function loadInitialQuotePdf() {
    const raw = panel.dataset.quotePdf;
    if (!raw || raw === "null") return;
    try {
      const data = JSON.parse(raw);
      if (data?.pdf_url) showHookStatus(data);
    } catch (_) {
      /* ignore malformed bootstrap JSON */
    }
  }

  function loadInitial() {
    if (!initialEl) return;
    try {
      const messages = JSON.parse(initialEl.textContent || "[]");
      messages.forEach((m) => {
        appendMessage(m.role, m.content, {
          markdown: m.role === "assistant",
          contextSize: m.context_size || null,
        });
      });
    } catch (_) {
      /* ignore malformed bootstrap JSON */
    }
    initialEl.remove();
  }

  async function sendMessage(text) {
    if (loading || !text.trim()) return;
    const { email, phone } = identityValues();
    syncIdentityFields();
    saveStoredIdentity(email, phone);
    abortController = new AbortController();
    setLoading(true);
    appendMessage("user", text.trim());
    input.value = "";
    showTyping();

    const body = new FormData();
    body.append("message", text.trim());
    body.append("test_email", email);
    body.append("test_phone", phone);
    body.append("test_session", anonymousTestSessionForSend());

    try {
      const res = await fetch(sendUrl, {
        method: "POST",
        body,
        signal: abortController.signal,
        credentials: "same-origin",
      });
      hideTyping();
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = err.detail || res.statusText || "Request failed";
        appendMessage("assistant", `Error: ${detail}`);
        return;
      }
      const data = await res.json();
      if (data.test_session) saveAnonymousSessionId(data.test_session);
      syncIdentityFields();
      appendMessage("assistant", data.reply || "", {
        markdown: true,
        contextSize: data.context_size || null,
      });
      showHookStatus(data);
    } catch (err) {
      hideTyping();
      if (err.name === "AbortError") return;
      appendMessage("assistant", `Error: ${err.message || "Network error"}`);
    } finally {
      abortController = null;
      setLoading(false);
      input.focus();
    }
  }

  form?.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(input.value);
  });

  input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  });

  stopBtn?.addEventListener("click", () => {
    abortController?.abort();
    hideTyping();
    setLoading(false);
    input.focus();
  });

  resetForm?.addEventListener("submit", () => {
    const { email, phone } = identityValues();
    syncIdentityFields();
    if (!email && !phone) {
      try {
        localStorage.removeItem(testSessionStorageKey);
      } catch (_) {
        /* ignore */
      }
    }
  });

  testEmailInput?.addEventListener("input", () => {
    const { email, phone } = identityValues();
    saveStoredIdentity(email, phone);
    syncIdentityFields();
  });
  testPhoneInput?.addEventListener("input", () => {
    const { email, phone } = identityValues();
    saveStoredIdentity(email, phone);
    syncIdentityFields();
  });

  document.getElementById("chat-clear-identity")?.addEventListener("click", () => {
    clearStoredIdentity();
    window.location.href = "?tab=chat";
  });

  if (sessionLabelEl && !sessionLabelEl.dataset.defaultSession) {
    sessionLabelEl.dataset.defaultSession = sessionLabelEl.textContent;
  }
  initIdentityFields();
  loadInitial();
  loadInitialQuotePdf();
  scrollThread();
  input?.focus();
})();
