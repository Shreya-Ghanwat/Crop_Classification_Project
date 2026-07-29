const API_BASE = "http://127.0.0.1:5000";

const form = document.getElementById("query-form");
const scanBtn = document.getElementById("scan-btn");
const statusPanel = document.getElementById("status-panel");
const statusText = document.getElementById("status-text");
const errorPanel = document.getElementById("error-panel");
const errorText = document.getElementById("error-text");
const results = document.getElementById("results");

const cropNameEl = document.getElementById("crop-name");
const confidenceValueEl = document.getElementById("confidence-value");
const grainFillEl = document.getElementById("grain-fill");
const dateStartEl = document.getElementById("date-start");
const datePeakEl = document.getElementById("date-peak");
const dateEndEl = document.getElementById("date-end");
const probListEl = document.getElementById("prob-list");
const chartEl = document.getElementById("ndvi-chart");

const STATUS_STEPS = [
  "Reaching the satellite…",
  "Pulling Sentinel-2 imagery…",
  "Filtering cloud cover…",
  "Tracing the growing season…",
  "Naming the crop…",
];

let statusInterval = null;

function startStatusCycle() {
  let i = 0;
  statusText.textContent = STATUS_STEPS[0];
  statusInterval = setInterval(() => {
    i = (i + 1) % STATUS_STEPS.length;
    statusText.textContent = STATUS_STEPS[i];
  }, 1400);
}

function stopStatusCycle() {
  clearInterval(statusInterval);
  statusInterval = null;
}

function showState(state) {
  statusPanel.classList.toggle("hidden", state !== "loading");
  errorPanel.classList.toggle("hidden", state !== "error");
  results.classList.toggle("hidden", state !== "results");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const coordsRaw = document.getElementById("coords").value.trim();
  const date = document.getElementById("date").value;

  if (!coordsRaw || !date) return;

  // Accept "18.5204, 73.8567" or "18.5204 73.8567" (comma or space separated)
  const parts = coordsRaw.split(/[,\s]+/).filter(Boolean);

  if (parts.length !== 2 || parts.some((p) => isNaN(parseFloat(p)))) {
    errorText.textContent = 'Please enter coordinates as "latitude, longitude", e.g. 18.5204, 73.8567';
    showState("error");
    return;
  }

  const lat = parseFloat(parts[0]);
  const lon = parseFloat(parts[1]);

  scanBtn.disabled = true;
  showState("loading");
  startStatusCycle();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, date }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Unknown error from server.");
    }

    renderResults(data);
    showState("results");
  } catch (err) {
    errorText.textContent =
      err.message === "Failed to fetch"
        ? "Couldn't reach the backend. Make sure the Flask server (app.py) is running on port 5000."
        : err.message;
    showState("error");
  } finally {
    stopStatusCycle();
    scanBtn.disabled = false;
  }
});

function renderResults(data) {
  cropNameEl.textContent = data.crop;

  dateStartEl.textContent = data.lifecycle.start_date;
  datePeakEl.textContent = data.lifecycle.peak_date;
  dateEndEl.textContent = data.lifecycle.end_date;

  const pct = Math.round(data.confidence * 100);
  confidenceValueEl.textContent = `${pct}%`;
  requestAnimationFrame(() => {
    grainFillEl.style.width = `${pct}%`;
  });

  renderProbabilities(data.all_probabilities, data.crop);
  renderChart(data.ndvi_curve, data.lifecycle);
}

function renderProbabilities(probs, topCrop) {
  probListEl.innerHTML = "";
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([crop, p]) => {
    const pct = Math.round(p * 100);
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-crop">${crop}${crop === topCrop ? " ★" : ""}</span>
      <div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div>
      <span class="prob-pct">${pct}%</span>
    `;
    probListEl.appendChild(row);
  });
}

function renderChart(curve, lifecycle) {
  const W = 800, H = 300;
  const padL = 40, padR = 20, padT = 20, padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const ndviValues = curve.map((d) => d.ndvi);
  const minN = Math.min(...ndviValues, 0);
  const maxN = Math.max(...ndviValues, 1);
  const range = maxN - minN || 1;

  const dates = curve.map((d) => new Date(d.date).getTime());
  const minT = Math.min(...dates);
  const maxT = Math.max(...dates);
  const tRange = maxT - minT || 1;

  const x = (t) => padL + ((t - minT) / tRange) * plotW;
  const y = (v) => padT + plotH - ((v - minN) / range) * plotH;

  let pathD = "";
  curve.forEach((d, i) => {
    const px = x(new Date(d.date).getTime());
    const py = y(d.ndvi);
    pathD += (i === 0 ? "M" : "L") + px.toFixed(1) + "," + py.toFixed(1) + " ";
  });

  const areaD =
    pathD +
    `L${x(maxT).toFixed(1)},${(padT + plotH).toFixed(1)} ` +
    `L${x(minT).toFixed(1)},${(padT + plotH).toFixed(1)} Z`;

  let gridLines = "";
  const gridSteps = 4;
  for (let i = 0; i <= gridSteps; i++) {
    const v = minN + (range * i) / gridSteps;
    const gy = y(v);
    gridLines += `<line x1="${padL}" y1="${gy.toFixed(1)}" x2="${W - padR}" y2="${gy.toFixed(1)}" stroke="var(--line)" stroke-width="1" />`;
    gridLines += `<text x="${padL - 8}" y="${(gy + 3).toFixed(1)}" text-anchor="end" font-family="IBM Plex Mono" font-size="10" fill="var(--ink-faint)">${v.toFixed(2)}</text>`;
  }

  function markerAt(dateStr, colorVar) {
    const t = new Date(dateStr).getTime();
    const nearest = curve.reduce((a, b) =>
      Math.abs(new Date(b.date).getTime() - t) < Math.abs(new Date(a.date).getTime() - t) ? b : a
    );
    const mx = x(new Date(nearest.date).getTime());
    const my = y(nearest.ndvi);
    return `<circle cx="${mx.toFixed(1)}" cy="${my.toFixed(1)}" r="5" fill="${colorVar}" stroke="var(--card)" stroke-width="2" />`;
  }

  const markers =
    markerAt(lifecycle.start_date, "var(--wheat)") +
    markerAt(lifecycle.peak_date, "var(--leaf)") +
    markerAt(lifecycle.end_date, "var(--rust)");

  const firstD = curve[0].date;
  const midD = curve[Math.floor(curve.length / 2)].date;
  const lastD = curve[curve.length - 1].date;
  const axisLabels = `
    <text x="${padL}" y="${H - 10}" font-family="IBM Plex Mono" font-size="10" fill="var(--ink-faint)">${firstD}</text>
    <text x="${(padL + W - padR) / 2}" y="${H - 10}" text-anchor="middle" font-family="IBM Plex Mono" font-size="10" fill="var(--ink-faint)">${midD}</text>
    <text x="${W - padR}" y="${H - 10}" text-anchor="end" font-family="IBM Plex Mono" font-size="10" fill="var(--ink-faint)">${lastD}</text>
  `;

  chartEl.innerHTML = `
    <defs>
      <linearGradient id="ndviFill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="var(--leaf)" stop-opacity="0.28" />
        <stop offset="100%" stop-color="var(--leaf)" stop-opacity="0" />
      </linearGradient>
    </defs>
    ${gridLines}
    <path d="${areaD}" fill="url(#ndviFill)" stroke="none" />
    <path d="${pathD}" fill="none" stroke="var(--leaf)" stroke-width="2.2" />
    ${markers}
    ${axisLabels}
  `;
}
