(function () {
  function initMessageBodyToggle(root) {
    const scope = root || document;
    scope.querySelectorAll(".validation-bubble-toggle").forEach((btn) => {
      if (btn.dataset.bound === "1") return;
      btn.dataset.bound = "1";
      btn.addEventListener("click", () => {
        const turn = btn.closest(".msg-turn");
        if (!turn) return;
        const clean = turn.querySelector(".validation-bubble-clean");
        const raw = turn.querySelector(".validation-bubble-raw");
        if (!clean || !raw) return;
        const showingRaw = btn.dataset.state === "raw";
        if (showingRaw) {
          clean.hidden = false;
          raw.hidden = true;
          btn.dataset.state = "clean";
          btn.textContent = "Show raw";
        } else {
          clean.hidden = true;
          raw.hidden = false;
          btn.dataset.state = "raw";
          btn.textContent = "Show cleaned";
        }
      });
    });
  }

  window.initMessageBodyToggle = initMessageBodyToggle;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initMessageBodyToggle());
  } else {
    initMessageBodyToggle();
  }
})();
