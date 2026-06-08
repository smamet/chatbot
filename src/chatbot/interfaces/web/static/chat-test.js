(function () {
  const panel = document.getElementById("chat-panel");
  if (!panel) return;

  const sendUrl = panel.dataset.sendUrl;
  const thread = document.getElementById("chat-thread");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const stopBtn = document.getElementById("chat-stop");
  const initialEl = document.getElementById("chat-initial");

  let abortController = null;
  let loading = false;

  const renderMarkdown = (text) =>
    window.ChatbotMarkdown?.renderMarkdown(text) ?? text;

  function scrollThread() {
    if (thread) thread.scrollTop = thread.scrollHeight;
  }

  function appendMessage(role, content, { markdown = false } = {}) {
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
    thread.appendChild(div);
    scrollThread();
    return div;
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
    const div = document.createElement("div");
    div.className = "msg msg-assistant msg-loading";
    div.id = "chat-typing";
    div.innerHTML =
      '<strong>assistant</strong><div class="msg-body"><span class="typing-dots"><span></span><span></span><span></span></span></div>';
    thread.appendChild(div);
    scrollThread();
  }

  function hideTyping() {
    document.getElementById("chat-typing")?.remove();
  }

  function loadInitial() {
    if (!initialEl) return;
    try {
      const messages = JSON.parse(initialEl.textContent || "[]");
      messages.forEach((m) => {
        appendMessage(m.role, m.content, { markdown: m.role === "assistant" });
      });
    } catch (_) {
      /* ignore malformed bootstrap JSON */
    }
    initialEl.remove();
  }

  async function sendMessage(text) {
    if (loading || !text.trim()) return;
    abortController = new AbortController();
    setLoading(true);
    appendMessage("user", text.trim());
    input.value = "";
    showTyping();

    const body = new FormData();
    body.append("message", text.trim());

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
      appendMessage("assistant", data.reply || "", { markdown: true });
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

  loadInitial();
  scrollThread();
  input?.focus();
})();
