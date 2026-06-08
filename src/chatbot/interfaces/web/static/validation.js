(function () {
  function toggleRow(row) {
    const panelId = row.dataset.panel;
    if (!panelId) return;
    const panel = document.getElementById(panelId);
    if (!panel) return;

    const willOpen = panel.hidden;
    panel.hidden = !willOpen;
    row.classList.toggle("is-open", willOpen);
    row.setAttribute("aria-expanded", willOpen ? "true" : "false");
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
})();
