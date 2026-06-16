(function () {
  function formatBytes(bytes) {
    var value = Math.max(0, Number(bytes) || 0);
    var units = ["B", "KiB", "MiB", "GiB", "TiB"];
    var i = 0;
    while (value >= 1024 && i < units.length - 1) {
      value /= 1024;
      i += 1;
    }
    if (i === 0) return String(Math.round(value)) + " " + units[i];
    return value.toFixed(1) + " " + units[i];
  }

  function compactCount(value) {
    var n = Number(value) || 0;
    if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, "") + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
    return String(n);
  }

  function makeLineChart(canvas, config) {
    if (!canvas || !window.Chart) return;
    return new Chart(canvas, config);
  }

  function initTokenChart(canvasId, data) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !data) return;
    makeLineChart(canvas, {
      type: "line",
      data: {
        labels: data.labels || [],
        datasets: [
          {
            label: "Tokens in",
            data: data.prompt_tokens || [],
            borderColor: "#2563eb",
            backgroundColor: "rgba(37, 99, 235, 0.1)",
            tension: 0.25,
            fill: false,
          },
          {
            label: "Tokens out",
            data: data.output_tokens || [],
            borderColor: "#16a34a",
            backgroundColor: "rgba(22, 163, 74, 0.1)",
            tension: 0.25,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            ticks: {
              callback: function (v) {
                return compactCount(v);
              },
            },
          },
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return ctx.dataset.label + ": " + Number(ctx.raw).toLocaleString();
              },
            },
          },
        },
      },
    });
  }

  function initDiskChart(canvasId, data) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !data) return;
    makeLineChart(canvas, {
      type: "line",
      data: {
        labels: data.labels || [],
        datasets: [
          {
            label: data.label || "Disk",
            data: data.total_bytes || [],
            borderColor: "#7c3aed",
            backgroundColor: "rgba(124, 58, 237, 0.1)",
            tension: 0.25,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            ticks: {
              callback: function (v) {
                return formatBytes(v);
              },
            },
          },
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return (data.label || "Disk") + ": " + formatBytes(ctx.raw);
              },
            },
          },
        },
      },
    });
  }

  window.MonitoringCharts = {
    initTokenChart: initTokenChart,
    initDiskChart: initDiskChart,
  };
})();
