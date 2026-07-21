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
  const sessionLabelEl = document.getElementById("chat-session-label");
  const testEmailInput = document.getElementById("chat-test-email");
  const testPhoneInput = document.getElementById("chat-test-phone");
  const testChannelSelect = document.getElementById("chat-test-channel");
  const resetEmailInput = document.getElementById("chat-reset-email");
  const resetPhoneInput = document.getElementById("chat-reset-phone");
  const resetTestSessionInput = document.getElementById("chat-reset-test-session");
  const resetForm = document.getElementById("chat-reset-form");
  const botSlug = panel.dataset.botSlug || "";
  const anonymousOnly = panel.dataset.anonymousOnly === "true";
  const identityStorageKey = botSlug ? `chatbot:test-chat:${botSlug}` : "chatbot:test-chat";
  const testSessionStorageKey = botSlug ? `chatbot:test-session:${botSlug}` : "chatbot:test-session";

  let abortController = null;
  let loading = false;

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
    if (anonymousOnly) {
      const fromServer = (panel.dataset.chatTestSession || "").trim();
      if (fromServer.startsWith("test:")) return fromServer;
      return getOrCreateAnonymousSessionId();
    }
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
          channel: String(parsed.channel ?? "").trim(),
        };
      }
    } catch (_) {
      /* ignore corrupt storage */
    }
    return { email: "", phone: "", channel: "" };
  }

  function clearStoredIdentity() {
    try {
      localStorage.removeItem(identityStorageKey);
      localStorage.removeItem(testSessionStorageKey);
    } catch (_) {
      /* ignore quota / private mode */
    }
  }

  function saveStoredIdentity(email, phone, channel = "") {
    try {
      localStorage.setItem(
        identityStorageKey,
        JSON.stringify({ email, phone, channel: channel || "" })
      );
    } catch (_) {
      /* ignore quota / private mode */
    }
  }

  function channelValue() {
    return testChannelSelect?.value?.trim() || "";
  }

  function initIdentityFields() {
    if (!testEmailInput && !testPhoneInput) return;
    const urlEmail = testEmailInput?.value?.trim() || "";
    const urlPhone = testPhoneInput?.value?.trim() || "";
    const stored = loadStoredIdentity();
    if (urlEmail || urlPhone) {
      saveStoredIdentity(urlEmail, urlPhone, stored.channel);
    } else {
      if (testEmailInput) testEmailInput.value = stored.email;
      if (testPhoneInput) testPhoneInput.value = stored.phone;
    }
    if (testChannelSelect && stored.channel) {
      const hasOption = Array.from(testChannelSelect.options).some(
        (opt) => opt.value === stored.channel
      );
      if (hasOption) testChannelSelect.value = stored.channel;
    }
    syncIdentityFields();
  }

  function syncIdentityFields() {
    const { email, phone } = identityValues();
    const testSession = anonymousOnly
      ? anonymousTestSessionForSend()
      : email || phone
        ? ""
        : getOrCreateAnonymousSessionId();
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

  function renderMessageBubble(turnData) {
    const role = turnData.role || "user";
    const turn = document.createElement("div");
    turn.className = `msg-turn msg-turn-${role}`;

    const div = document.createElement("div");
    div.className = `msg msg-${role}`;
    const label = document.createElement("strong");
    label.textContent = role;
    div.appendChild(label);

    const bubbleBody = document.createElement("div");
    bubbleBody.className = "validation-bubble-body";

    const useMarkdown = turnData.markdown && role === "assistant";
    const clean = document.createElement(useMarkdown ? "div" : "pre");
    clean.className = `validation-bubble-text msg-body validation-bubble-clean${
      role === "user" ? " msg-body--plain" : useMarkdown ? " js-md" : ""
    }`;
    const cleanText = turnData.content_clean ?? turnData.content ?? "";
    clean.textContent = cleanText;
    bubbleBody.appendChild(clean);

    if (turnData.content_raw) {
      const raw = document.createElement("pre");
      raw.className =
        "validation-bubble-text msg-body msg-body--plain validation-bubble-raw";
      raw.hidden = true;
      raw.textContent = turnData.content_raw;
      bubbleBody.appendChild(raw);
    }

    div.appendChild(bubbleBody);
    turn.appendChild(div);

    if (turnData.content_raw) {
      const footer = document.createElement("div");
      footer.className = "validation-bubble-footer";
      const tokens = document.createElement("span");
      tokens.className = "validation-bubble-tokens";
      tokens.title = "Approximate token count (len/4)";
      if (turnData.token_raw != null && turnData.reduction_pct != null) {
        tokens.textContent = `~${turnData.token_raw} tokens → ~${turnData.token_new} tokens (−${turnData.reduction_pct}%)`;
      } else if (turnData.token_new) {
        tokens.textContent = `~${turnData.token_new} tokens`;
      }
      footer.appendChild(tokens);
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "validation-bubble-toggle";
      toggle.dataset.state = "clean";
      toggle.textContent = "Show raw";
      footer.appendChild(toggle);
      turn.appendChild(footer);
    }

    const contextSize = turnData.context_size || turnData.contextSize || null;
    if (contextSize && role === "assistant") {
      const debug = document.createElement("p");
      debug.className = "msg-context-debug";
      debug.textContent = formatContextSizeLabel(contextSize);
      turn.appendChild(debug);
    }

    thread.appendChild(turn);
    if (typeof window.initMessageBodyToggle === "function") {
      window.initMessageBodyToggle(turn);
    }
    if (useMarkdown && window.ChatbotMarkdown) {
      window.ChatbotMarkdown.applyMarkdown(turn);
    }
    scrollThread();
    return turn;
  }

  function appendMessage(role, content, opts = {}) {
    if (opts.content_clean != null || opts.content_raw) {
      return renderMessageBubble({ role, content, ...opts });
    }
    return renderMessageBubble({
      role,
      content,
      content_clean: content,
      markdown: opts.markdown,
      context_size: opts.contextSize,
    });
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
    if (!data || (!data.pdf_url && !data.queued && !data.message)) {
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
        renderMessageBubble({
          ...m,
          markdown: m.markdown || m.role === "assistant",
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
    const channel = channelValue();
    saveStoredIdentity(email, phone, channel);
    abortController = new AbortController();
    setLoading(true);
    appendMessage("user", text.trim(), { content_clean: text.trim() });
    input.value = "";
    showTyping();

    const body = new FormData();
    body.append("message", text.trim());
    body.append("test_email", email);
    body.append("test_phone", phone);
    body.append("test_session", anonymousTestSessionForSend());
    body.append("channel", channel);

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
      if (data.user_message) {
        const lastUser = thread?.querySelector(".msg-turn-user:last-of-type");
        if (lastUser) lastUser.remove();
        renderMessageBubble({ ...data.user_message, markdown: false });
      }
      if (data.assistant_message) {
        renderMessageBubble({
          ...data.assistant_message,
          markdown: true,
          context_size: data.context_size || data.assistant_message.context_size,
        });
      } else {
        appendMessage("assistant", data.reply || "", {
          content_clean: data.reply || "",
          markdown: true,
          context_size: data.context_size || null,
        });
      }
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
    saveStoredIdentity(email, phone, channelValue());
    syncIdentityFields();
  });
  testPhoneInput?.addEventListener("input", () => {
    const { email, phone } = identityValues();
    saveStoredIdentity(email, phone, channelValue());
    syncIdentityFields();
  });
  testChannelSelect?.addEventListener("change", () => {
    const { email, phone } = identityValues();
    saveStoredIdentity(email, phone, channelValue());
  });

  document.getElementById("chat-clear-identity")?.addEventListener("click", () => {
    clearStoredIdentity();
    window.location.href = "?tab=chat";
  });

  if (sessionLabelEl && !sessionLabelEl.dataset.defaultSession) {
    sessionLabelEl.dataset.defaultSession = sessionLabelEl.textContent;
  }
  if (!anonymousOnly) {
    initIdentityFields();
  } else {
    syncIdentityFields();
  }
  loadInitial();
  loadInitialQuotePdf();
  scrollThread();
  input?.focus();
})();
