(function () {
  var COLORS = {
    tokensIn: "#4f46e5",
    tokensOut: "#059669",
    disk: "#6366f1",
    diskHost: "#0ea5e9",
    diskUsed: "#4f46e5",
    diskFree: "#e2e8f0",
  };

  if (window.Chart) {
    Chart.defaults.font.family = '"Inter", system-ui, -apple-system, sans-serif';
    Chart.defaults.color = "#64748b";
    Chart.defaults.animation.duration = 600;
    Chart.defaults.animation.easing = "easeOutQuart";
  }

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

  function hexToRgba(hex, alpha) {
    var h = hex.replace("#", "");
    var r = parseInt(h.substring(0, 2), 16);
    var g = parseInt(h.substring(2, 4), 16);
    var b = parseInt(h.substring(4, 6), 16);
    return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
  }

  function areaGradient(canvas, color) {
    var ctx = canvas.getContext("2d");
    var h = canvas.parentElement ? canvas.parentElement.clientHeight : 260;
    var gradient = ctx.createLinearGradient(0, 0, 0, h);
    gradient.addColorStop(0, hexToRgba(color, 0.28));
    gradient.addColorStop(0.65, hexToRgba(color, 0.06));
    gradient.addColorStop(1, hexToRgba(color, 0));
    return gradient;
  }

  function lineDataset(label, data, color, canvas) {
    return {
      label: label,
      data: data,
      borderColor: color,
      backgroundColor: areaGradient(canvas, color),
      fill: true,
      tension: 0.38,
      borderWidth: 2.5,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBorderWidth: 2,
      pointBackgroundColor: "#ffffff",
      pointBorderColor: color,
      pointHitRadius: 12,
    };
  }

  function baseOptions(yTickCallback) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      layout: {
        padding: { top: 8, right: 12, bottom: 0, left: 4 },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { display: false },
          ticks: {
            maxTicksLimit: 7,
            maxRotation: 0,
            autoSkipPadding: 16,
            color: "#94a3b8",
            font: { size: 11, weight: "500" },
          },
        },
        y: {
          beginAtZero: true,
          border: { display: false },
          grid: {
            color: "rgba(148, 163, 184, 0.18)",
            drawTicks: false,
          },
          ticks: {
            padding: 8,
            color: "#94a3b8",
            font: { size: 11, weight: "500" },
            callback: yTickCallback,
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
          align: "start",
          labels: {
            usePointStyle: true,
            pointStyle: "circle",
            boxWidth: 8,
            boxHeight: 8,
            padding: 18,
            color: "#64748b",
            font: { size: 12, weight: "500" },
          },
        },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#f8fafc",
          bodyColor: "#e2e8f0",
          borderColor: "rgba(148, 163, 184, 0.25)",
          borderWidth: 1,
          cornerRadius: 10,
          padding: 12,
          boxPadding: 6,
          titleFont: { size: 12, weight: "600" },
          bodyFont: { size: 12 },
          displayColors: true,
          usePointStyle: true,
        },
      },
    };
  }

  function pieTooltipLabel(ctx) {
    var total = ctx.dataset.data.reduce(function (a, b) {
      return a + b;
    }, 0);
    var value = Number(ctx.raw) || 0;
    var pct = total > 0 ? ((value / total) * 100).toFixed(1) : "0";
    return ctx.label + ": " + formatBytes(value) + " (" + pct + "%)";
  }

  function pieOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      layout: {
        padding: 8,
      },
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            usePointStyle: true,
            pointStyle: "circle",
            boxWidth: 8,
            boxHeight: 8,
            padding: 16,
            color: "#64748b",
            font: { size: 12, weight: "500" },
          },
        },
        tooltip: {
          backgroundColor: "#0f172a",
          titleColor: "#f8fafc",
          bodyColor: "#e2e8f0",
          borderColor: "rgba(148, 163, 184, 0.25)",
          borderWidth: 1,
          cornerRadius: 10,
          padding: 12,
          callbacks: {
            label: pieTooltipLabel,
          },
        },
      },
    };
  }

  function makeLineChart(canvas, config) {
    if (!canvas || !window.Chart) return;
    return new Chart(canvas, config);
  }

  function initTokenChart(canvasId, data) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !data) return;
    var options = baseOptions(compactCount);
    options.plugins.tooltip.callbacks = {
      label: function (ctx) {
        return ctx.dataset.label + ": " + Number(ctx.raw).toLocaleString();
      },
    };
    makeLineChart(canvas, {
      type: "line",
      data: {
        labels: data.labels || [],
        datasets: [
          lineDataset("Tokens in", data.prompt_tokens || [], COLORS.tokensIn, canvas),
          lineDataset("Tokens out", data.output_tokens || [], COLORS.tokensOut, canvas),
        ],
      },
      options: options,
    });
  }

  function initDiskChart(canvasId, data, color) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !data) return;
    var stroke = color || COLORS.disk;
    var options = baseOptions(formatBytes);
    options.plugins.legend.display = false;
    options.plugins.tooltip.callbacks = {
      label: function (ctx) {
        return (data.label || "Disk") + ": " + formatBytes(ctx.raw);
      },
    };
    makeLineChart(canvas, {
      type: "line",
      data: {
        labels: data.labels || [],
        datasets: [
          lineDataset(data.label || "Disk", data.total_bytes || [], stroke, canvas),
        ],
      },
      options: options,
    });
  }

  function initDiskPieChart(canvasId, data) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || !data) return;
    var used = Number(data.used_bytes) || 0;
    var free = Number(data.free_bytes) || 0;
    if (used === 0 && free === 0) {
      free = 1;
    }
    new Chart(canvas, {
      type: "doughnut",
      data: {
        labels: ["Used", "Free"],
        datasets: [
          {
            data: [used, free],
            backgroundColor: [COLORS.diskUsed, COLORS.diskFree],
            borderWidth: 0,
            hoverOffset: 8,
            spacing: 2,
          },
        ],
      },
      options: pieOptions(),
    });
  }

  window.MonitoringCharts = {
    initTokenChart: initTokenChart,
    initDiskChart: initDiskChart,
    initDiskPieChart: initDiskPieChart,
    COLORS: COLORS,
  };
})();
