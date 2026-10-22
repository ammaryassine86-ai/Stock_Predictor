// ---- Search autocomplete -------------------------------------------------
(function () {
  const input = document.getElementById("q");
  const list = document.getElementById("suggest");
  if (!input || !list) return;
  let timer = null, items = [], active = -1;

  const urlOf = (it) => it.kind ? `/${it.kind}/` + encodeURIComponent(it.name)
                                : "/stock/" + encodeURIComponent(it.ticker);
  function render() {
    list.innerHTML = "";
    items.forEach((it, i) => {
      const li = document.createElement("li");
      if (i === active) li.className = "active";
      if (it.kind) {
        li.innerHTML = `<span class="tag">${it.kind}</span><span class="nm"><b>${it.name}</b></span><span class="ct">${it.count} stocks</span>`;
      } else {
        const q = it.quintile ? `<span class="q q${it.quintile}">${it.quintile}/5</span>` : "";
        li.innerHTML = `<b>${it.ticker}</b><span class="nm">${it.name}</span><span class="ct">${it.country}</span>${q}`;
      }
      li.onmousedown = () => (location = urlOf(it));
      list.appendChild(li);
    });
    list.hidden = items.length === 0;
  }

  input.addEventListener("input", () => {
    clearTimeout(timer);
    const q = input.value.trim();
    if (!q) { items = []; render(); return; }
    timer = setTimeout(async () => {
      const r = await fetch("/api/search?q=" + encodeURIComponent(q));
      const d = await r.json(); items = d.groups.concat(d.stocks); active = -1; render();
    }, 120);
  });
  input.addEventListener("keydown", (e) => {
    if (list.hidden) return;
    if (e.key === "ArrowDown") { active = Math.min(active + 1, items.length - 1); render(); e.preventDefault(); }
    if (e.key === "ArrowUp") { active = Math.max(active - 1, 0); render(); e.preventDefault(); }
    if (e.key === "Enter" && active >= 0) { location = urlOf(items[active]); e.preventDefault(); }
    if (e.key === "Escape") { items = []; render(); }
  });
  input.addEventListener("blur", () => setTimeout(() => (list.hidden = true), 150));
})();

// ---- Track button ----------------------------------------------------------
(function () {
  const btn = document.getElementById("track");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const on = !btn.classList.contains("on");
    const r = await fetch("/api/track/" + encodeURIComponent(btn.dataset.ticker), { method: on ? "POST" : "DELETE" });
    const d = await r.json();
    btn.classList.toggle("on", d.tracked);
    btn.textContent = d.tracked ? "✓ Tracking" : "+ Track";
  });
})();

// ---- Price chart with the projected range -----------------------------------
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawStockChart(history, proj) {
  const canvas = document.getElementById("chart");
  if (!canvas || typeof Chart === "undefined") return;
  const pad = (n) => Array(n).fill(null);
  const nH = history.dates.length, nP = proj ? proj.dates.length : 0;
  const labels = history.dates.concat(proj ? proj.dates : []);
  const last = history.close[nH - 1];
  // the projection starts at the last real price so the lines connect
  const join = (arr) => pad(nH - 1).concat([last], arr);
  const ink = cssVar("--ink"), muted = cssVar("--muted"), grid = cssVar("--grid");
  const accent = cssVar("--accent"), band = cssVar("--band"), band2 = cssVar("--band2");

  const datasets = [{ label: "Price", data: history.close.concat(pad(nP)), borderColor: ink,
                      borderWidth: 1.6, pointRadius: 0, tension: 0.15 }];
  if (proj) {
    datasets.push(
      { label: "high95", data: join(proj.high95), borderWidth: 0, pointRadius: 0, fill: false },
      { label: "low95", data: join(proj.low95), borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: band2 },
      { label: "high68", data: join(proj.high68), borderWidth: 0, pointRadius: 0, fill: false },
      { label: "low68", data: join(proj.low68), borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: band },
      { label: "Middle estimate", data: join(proj.mid), borderColor: accent, borderDash: [5, 4],
        borderWidth: 1.8, pointRadius: 0 });
  }
  new Chart(canvas, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (i) => i.raw !== null,
          callbacks: {
            label: (c) => {
              const names = { high95: "Wide range top", low95: "Wide range bottom",
                              high68: "Likely range top", low68: "Likely range bottom" };
              return `${names[c.dataset.label] || c.dataset.label}: ${Number(c.raw).toFixed(2)}`;
            },
          },
        },
      },
      scales: {
        x: { ticks: { color: muted, maxTicksLimit: 8, maxRotation: 0 }, grid: { display: false } },
        y: { ticks: { color: muted }, grid: { color: grid } },
      },
    },
  });
}
