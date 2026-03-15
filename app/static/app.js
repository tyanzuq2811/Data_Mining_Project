/* ═══════════════════════════════════════════════
   BẢO TRÌ DỰ ĐOÁN — Main Application Script
   ═══════════════════════════════════════════════ */

const API = '';  // same origin

// ── Theme state ─────────────────────────────────
let currentTheme = localStorage.getItem('pm-theme') || 'dark';
function isDark() { return currentTheme === 'dark'; }

// ── Plotly layouts ──────────────────────────────
const darkLayout = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#b3ecff', family: 'Inter', size: 14 },
  margin: { l: 60, r: 30, t: 40, b: 60 },
  xaxis: { gridcolor: '#00bfff11', zerolinecolor: '#00bfff22', tickfont: { size: 12 } },
  yaxis: { gridcolor: '#00bfff11', zerolinecolor: '#00bfff22', tickfont: { size: 12 } },
  legend: { bgcolor: 'transparent', font: { size: 13 } },
  hoverlabel: { bgcolor: '#0a1628', bordercolor: '#00bfff', font: { color: '#e0f7ff', size: 13 } },
};
const lightLayout = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#1a202c', family: 'Inter', size: 14 },
  margin: { l: 60, r: 30, t: 40, b: 60 },
  xaxis: { gridcolor: 'rgba(0,0,0,0.06)', zerolinecolor: 'rgba(0,0,0,0.10)', tickfont: { size: 12 } },
  yaxis: { gridcolor: 'rgba(0,0,0,0.06)', zerolinecolor: 'rgba(0,0,0,0.10)', tickfont: { size: 12 } },
  legend: { bgcolor: 'transparent', font: { size: 13 } },
  hoverlabel: { bgcolor: '#ffffff', bordercolor: '#0077b6', font: { color: '#1a202c', size: 13 } },
};
function getLayout(extra) {
  const base = isDark() ? darkLayout : lightLayout;
  return { ...base, ...(extra || {}) };
}
const plotConfig = { responsive: true, displayModeBar: false };

// ── Theme toggle ────────────────────────────────
function applyTheme() {
  if (currentTheme === 'light') {
    document.body.classList.add('light-mode');
  } else {
    document.body.classList.remove('light-mode');
  }
  document.getElementById('theme-icon').innerHTML = isDark() ? '&#9728;&#65039;' : '&#127769;';
  document.getElementById('theme-label').textContent = isDark() ? 'Sáng' : 'Tối';
}

function toggleTheme() {
  currentTheme = isDark() ? 'light' : 'dark';
  localStorage.setItem('pm-theme', currentTheme);
  applyTheme();
  // Re-render active tab charts
  const activeTab = document.querySelector('.tab-btn.active');
  if (activeTab) activeTab.click();
}

// ── Helper: neon table from array of objects ────
function neonTable(data, highlight_col) {
  if (!data || !data.length) return '<p class="text-neon-300/40 text-sm">Không có dữ liệu</p>';
  const cols = Object.keys(data[0]);
  let html = '<table class="w-full text-sm"><thead><tr>';
  cols.forEach(c => {
    html += `<th class="px-3 py-2 text-left text-xs text-neon-300/60 uppercase tracking-wider border-b border-neon-400/20 font-medium">${c}</th>`;
  });
  html += '</tr></thead><tbody>';
  data.forEach((row, i) => {
    const bg = i % 2 === 0 ? '' : 'bg-neon-400/[0.03]';
    html += `<tr class="${bg} hover:bg-neon-400/10 transition-colors">`;
    cols.forEach(c => {
      let val = row[c];
      if (typeof val === 'number') val = val.toFixed ? val.toFixed(4) : val;
      const extra = c === highlight_col ? 'text-neon-400 font-bold' : '';
      html += `<td class="px-3 py-2 border-b border-neon-400/10 ${extra} font-mono">${val ?? '-'}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  return html;
}

// ── KPIs ────────────────────────────────────────
async function loadKPIs() {
  const [summary, clf] = await Promise.all([
    fetch(API + '/api/data/summary').then(r => r.json()),
    fetch(API + '/api/results/classification').then(r => r.json()),
  ]);
  const best = clf[0]; // gradient_boosting
  const kpis = [
    { label: 'Tổng bản ghi', value: summary.total_records.toLocaleString(), icon: `<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6"/></svg>`, sub: 'Bộ dữ liệu AI4I 2020', color: '#00bfff' },
    { label: 'Tỷ lệ lỗi', value: summary.failure_rate + '%', icon: `<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/></svg>`, sub: `${summary.failures} lỗi / ${summary.normal} bình thường`, color: '#ff6b6b' },
    { label: 'F1 tốt nhất', value: best.f1.toFixed(4), icon: `<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z"/></svg>`, sub: best.model.replace('_', ' ').toUpperCase(), color: '#00ff88' },
    { label: 'Đặc trưng', value: summary.features_count, icon: `<svg class="w-8 h-8" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z"/><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>`, sub: 'Trích xuất từ 5 cảm biến', color: '#ffd60a' },
  ];
  const grid = document.getElementById('kpi-grid');
  grid.innerHTML = kpis.map((k, i) => `
    <div class="neon-card p-5 animate-slide-up" style="animation-delay: ${i * 0.1}s">
      <div class="flex items-start justify-between">
        <div>
          <p class="text-xs text-neon-300/50 uppercase tracking-wider mb-1">${k.label}</p>
          <p class="text-3xl font-orbitron font-bold text-neon-400 neon-text-sm count-anim">${k.value}</p>
          <p class="text-xs text-neon-300/40 mt-1">${k.sub}</p>
        </div>
        <div class="p-2 rounded-xl" style="color:${k.color};background:${k.color}15;border:1px solid ${k.color}30">${k.icon}</div>
      </div>
    </div>
  `).join('');
}

// ── Classification Charts ───────────────────────
async function loadClassification() {
  const [clf, cv] = await Promise.all([
    fetch(API + '/api/results/classification').then(r => r.json()),
    fetch(API + '/api/results/timeseries').then(r => r.json()),  // placeholder; we use cv below
  ]);
  // Fetch CV data
  const cvData = await fetch(API + '/api/results/classification').then(r => r.json());

  // Bar chart: F1 and PR-AUC side by side
  const models = clf.map(r => r.model.replace('_', ' '));
  const neonColors = ['#00bfff', '#00e5ff', '#00ff88', '#ffd60a', '#ff6b6b'];

  Plotly.newPlot('chart-clf-bar', [
    {
      x: models, y: clf.map(r => r.f1), name: 'F1 Score', type: 'bar',
      marker: { color: neonColors.map(c => c + '99'), line: { color: neonColors, width: 1.5 } },
      text: clf.map(r => r.f1.toFixed(3)), textposition: 'auto', textfont: { color: '#e0f7ff', size: 13 }
    },
    {
      x: models, y: clf.map(r => r.pr_auc), name: 'PR-AUC', type: 'bar',
      marker: { color: neonColors.map(c => c + '44'), line: { color: neonColors, width: 1.5 } },
      text: clf.map(r => r.pr_auc.toFixed(3)), textposition: 'auto', textfont: { color: '#e0f7ff', size: 13 }
    },
  ], getLayout({ barmode: 'group' }), plotConfig);

  // CV chart (Radar/bar)
  // Re-fetch CV results
  fetch('/api/results/classification').then(r => r.json()).then(data => {
    // Use roc_auc as a proxy for CV since cv endpoint doesn't exist on FastAPI
    // Build a radar-like grouped bar with precision/recall/f1
    Plotly.newPlot('chart-cv', [
      {
        x: models, y: clf.map(r => r.precision), name: 'Precision', type: 'bar',
        marker: { color: '#00bfff66', line: { color: '#00bfff', width: 1 } },
      },
      {
        x: models, y: clf.map(r => r.recall), name: 'Recall', type: 'bar',
        marker: { color: '#00ff8866', line: { color: '#00ff88', width: 1 } },
      },
      {
        x: models, y: clf.map(r => r.roc_auc), name: 'ROC-AUC', type: 'bar',
        marker: { color: '#ffd60a66', line: { color: '#ffd60a', width: 1 } },
      },
    ], getLayout({ barmode: 'group' }), plotConfig);
  });

  // Table
  document.getElementById('clf-table').innerHTML = neonTable(clf, 'f1');
}

// ── Clustering ──────────────────────────────────
async function loadClustering() {
  const data = await fetch(API + '/api/results/clustering').then(r => r.json());
  const models = data.map(r => r.model);
  Plotly.newPlot('chart-cluster-sil', [{
    y: models, x: data.map(r => r.silhouette), type: 'bar', orientation: 'h',
    marker: {
      color: data.map(r => r.silhouette > 0 ? '#00bfff88' : '#ff6b6b88'),
      line: { color: data.map(r => r.silhouette > 0 ? '#00bfff' : '#ff6b6b'), width: 1 }
    },
    text: data.map(r => r.silhouette.toFixed(3)), textposition: 'auto',
    textfont: { color: '#e0f7ff', size: 12 }
  }], getLayout({
    margin: { l: 150, r: 20, t: 10, b: 40 },
    xaxis: { ...getLayout().xaxis, title: 'Silhouette Score' },
    yaxis: { ...getLayout().yaxis, autorange: 'reversed' },
  }), plotConfig);
  document.getElementById('cluster-table').innerHTML = neonTable(data, 'silhouette');
}

// ── Anomaly ─────────────────────────────────────
async function loadAnomaly() {
  const data = await fetch(API + '/api/results/anomaly').then(r => r.json());
  const methods = data.map(r => r.Method);
  Plotly.newPlot('chart-anomaly', [
    { x: methods, y: data.map(r => r.Precision), name: 'Precision', type: 'bar', marker: { color: '#00bfff88', line: { color: '#00bfff', width: 1.5 }} },
    { x: methods, y: data.map(r => r.Recall), name: 'Recall', type: 'bar', marker: { color: '#00ff8888', line: { color: '#00ff88', width: 1.5 }} },
    { x: methods, y: data.map(r => r.F1), name: 'F1 Score', type: 'bar', marker: { color: '#ffd60a88', line: { color: '#ffd60a', width: 1.5 }} },
  ], getLayout({ barmode: 'group' }), plotConfig);
  document.getElementById('anomaly-table').innerHTML = neonTable(data, 'F1');
}

// ── Regression ──────────────────────────────────
async function loadRegression() {
  const data = await fetch(API + '/api/results/regression').then(r => r.json());
  const models = data.map(r => r.model.replace('_', ' '));
  Plotly.newPlot('chart-reg', [
    { x: models, y: data.map(r => r.MAE), name: 'MAE', type: 'bar',
      marker: { color: '#00bfff88', line: { color: '#00bfff', width: 1.5 } },
      text: data.map(r => r.MAE.toFixed(2)), textposition: 'auto', textfont: { color: '#e0f7ff' }
    },
    { x: models, y: data.map(r => r.RMSE), name: 'RMSE', type: 'bar',
      marker: { color: '#ff6b6b88', line: { color: '#ff6b6b', width: 1.5 } },
      text: data.map(r => r.RMSE.toFixed(2)), textposition: 'auto', textfont: { color: '#e0f7ff' }
    },
  ], getLayout({ barmode: 'group' }), plotConfig);
  document.getElementById('reg-table').innerHTML = neonTable(data, 'MAE');
}

// ── Semi-supervised ─────────────────────────────
async function loadSemiSupervised() {
  const data = await fetch(API + '/api/results/semi_supervised').then(r => r.json());
  const methods = [...new Set(data.map(r => r.method))];
  const colors = { supervised_only: '#00bfff', self_training: '#00ff88', label_spreading: '#ffd60a' };
  const traces = methods.map(m => {
    const subset = data.filter(r => r.method === m);
    return {
      x: subset.map(r => (r.label_pct * 100) + '%'),
      y: subset.map(r => r.f1),
      name: m.replace('_', ' '),
      type: 'scatter', mode: 'lines+markers',
      line: { color: colors[m] || '#00bfff', width: 2 },
      marker: { size: 8, color: colors[m] || '#00bfff', line: { color: '#e0f7ff', width: 1 } },
    };
  });
  Plotly.newPlot('chart-semi', traces, getLayout({
    xaxis: { ...getLayout().xaxis, title: 'Tỷ lệ nhãn' },
    yaxis: { ...getLayout().yaxis, title: 'F1 Score' },
  }), plotConfig);
  document.getElementById('semi-table').innerHTML = neonTable(data, 'f1');
}

// ── Scatter Explorer ────────────────────────────
async function loadScatter() {
  const fx = document.getElementById('scatter-x').value;
  const fy = document.getElementById('scatter-y').value;
  const data = await fetch(`${API}/api/data/scatter/${fx}/${fy}`).then(r => r.json());
  if (data.error) return;

  const normal_idx = data.failure.map((f, i) => f === 0 ? i : null).filter(i => i !== null);
  const fail_idx = data.failure.map((f, i) => f === 1 ? i : null).filter(i => i !== null);

  Plotly.newPlot('chart-scatter', [
    {
      x: normal_idx.map(i => data.x[i]), y: normal_idx.map(i => data.y[i]),
      mode: 'markers', name: 'Bình thường', type: 'scattergl',
      marker: { color: '#00bfff44', size: 5, line: { color: '#00bfff', width: 0.5 } },
    },
    {
      x: fail_idx.map(i => data.x[i]), y: fail_idx.map(i => data.y[i]),
      mode: 'markers', name: 'Lỗi', type: 'scattergl',
      marker: { color: '#ff2d55', size: 9, symbol: 'x', line: { color: '#ff2d55', width: 1.5 } },
    },
  ], getLayout({
    xaxis: { ...getLayout().xaxis, title: data.x_label },
    yaxis: { ...getLayout().yaxis, title: data.y_label },
  }), plotConfig);
}

// ── Gallery ─────────────────────────────────────
async function loadGallery() {
  const data = await fetch(API + '/api/figures').then(r => r.json());
  const grid = document.getElementById('gallery-grid');
  grid.innerHTML = data.figures.map(f => `
    <div class="neon-card overflow-hidden cursor-pointer group" onclick="openLightbox('/figures/${f}')">
      <div class="relative">
        <img src="/figures/${f}" alt="${f}" class="w-full h-40 object-cover opacity-80 group-hover:opacity-100 transition-opacity" loading="lazy">
        <div class="absolute inset-0 bg-gradient-to-t from-neon-950 to-transparent"></div>
      </div>
      <p class="px-3 py-2 text-xs text-neon-300/60 truncate font-mono">${f.replace('.png','')}</p>
    </div>
  `).join('');
}

// ── Insights ────────────────────────────────────
async function loadInsights() {
  const data = await fetch(API + '/api/results/insights').then(r => r.json());
  const svgIcons = [
    `<svg class="w-6 h-6" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M11.42 15.17l-5.633-7.356A1.5 1.5 0 017.01 6h9.98a1.5 1.5 0 011.222 2.814l-5.633 7.356a1.5 1.5 0 01-2.38 0z"/><path stroke-linecap="round" stroke-linejoin="round" d="M21.75 12a9.75 9.75 0 11-19.5 0 9.75 9.75 0 0119.5 0z"/></svg>`,
    `<svg class="w-6 h-6" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.048 8.287 8.287 0 009 9.6a8.983 8.983 0 013.361-6.867 8.21 8.21 0 003 2.48z"/><path stroke-linecap="round" stroke-linejoin="round" d="M12 18a3.75 3.75 0 00.495-7.467 5.99 5.99 0 00-1.925 3.546 5.974 5.974 0 01-2.133-1A3.75 3.75 0 0012 18z"/></svg>`,
    `<svg class="w-6 h-6" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z"/><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>`,
    `<svg class="w-6 h-6" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"/></svg>`,
    `<svg class="w-6 h-6" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z"/></svg>`,
    `<svg class="w-6 h-6" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z"/><path stroke-linecap="round" stroke-linejoin="round" d="M6 6h.008v.008H6V6z"/></svg>`,
    `<svg class="w-6 h-6" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 21v-8.25M15.75 21v-8.25M8.25 21v-8.25M3 9l9-6 9 6m-1.5 12V10.332A48.36 48.36 0 0012 9.75c-2.551 0-5.056.2-7.5.582V21M3 21h18M12 6.75h.008v.008H12V6.75z"/></svg>`,
  ];
  const colorHex = ['#00bfff', '#ffd60a', '#ff6b6b', '#00ff88', '#f97316', '#a855f7', '#ec4899'];
  const grid = document.getElementById('insights-grid');
  grid.innerHTML = data.insights.map((ins, i) => {
    const parts = ins.match(/^\d+\.\s*([^:]+):\s*([\s\S]*)$/);
    const title = parts ? parts[1] : `Insight ${i + 1}`;
    const body = parts ? parts[2] : ins;
    const clr = colorHex[i % colorHex.length];
    return `
      <div class="neon-card p-5 animate-slide-up" style="animation-delay: ${i * 0.08}s">
        <div class="flex items-start gap-3">
          <div class="p-2 rounded-lg flex-shrink-0" style="color:${clr};background:${clr}15;border:1px solid ${clr}30">${svgIcons[i] || svgIcons[0]}</div>
          <div>
            <h4 class="font-orbitron text-xs tracking-wider mb-2" style="color:${clr}">${title}</h4>
            <p class="text-sm text-neon-200/70 leading-relaxed">${body.trim()}</p>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// ── Prediction ──────────────────────────────────
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('predict-btn');
  const btnText = document.getElementById('btn-text');
  const btnLoader = document.getElementById('btn-loader');
  btn.disabled = true;
  btnText.classList.add('hidden');
  btnLoader.classList.remove('hidden');

  const payload = {
    air_temperature: parseFloat(document.getElementById('inp-air').value),
    process_temperature: parseFloat(document.getElementById('inp-proc').value),
    rotational_speed: parseFloat(document.getElementById('inp-speed').value),
    torque: parseFloat(document.getElementById('inp-torque').value),
    tool_wear: parseFloat(document.getElementById('inp-wear').value),
    product_type: document.getElementById('inp-type').value,
  };

  try {
    const res = await fetch(API + '/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await res.json();
    showResult(result, payload);
  } catch (err) {
    alert('Dự đoán thất bại: ' + err.message);
  } finally {
    btn.disabled = false;
    btnText.classList.remove('hidden');
    btnLoader.classList.add('hidden');
  }
});

function showResult(r, inp) {
  document.getElementById('result-placeholder').classList.add('hidden');
  const content = document.getElementById('result-content');
  content.classList.remove('hidden');

  const riskClass = {
    CRITICAL: 'risk-critical',
    HIGH: 'risk-high',
    MEDIUM: 'risk-medium',
    LOW: 'risk-low',
  }[r.risk_level];

  const statusIcon = r.prediction === 1
    ? '<svg class="w-12 h-12 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"/></svg>'
    : '<svg class="w-12 h-12 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>';

  content.innerHTML = `
    <div class="flex items-center gap-4 mb-4 animate-fade-in">
      ${statusIcon}
      <div>
        <h3 class="font-orbitron text-xl font-bold ${r.prediction === 1 ? 'text-red-400' : 'text-green-400'}">
          ${r.prediction === 1 ? 'PHÁT HIỆN LỖI' : 'MÁY BÌNH THƯỜNG'}
        </h3>
        <p class="text-sm text-neon-200/60">Xác suất lỗi: <span class="font-mono font-bold text-neon-400">${(r.failure_probability * 100).toFixed(2)}%</span></p>
      </div>
    </div>

    <!-- Risk Level Badge -->
    <div class="flex items-center gap-3 mb-4">
      <span class="px-4 py-2 rounded-lg border text-sm font-orbitron font-bold ${riskClass}">
        ${r.risk_level}
      </span>
      <div class="flex-1 h-3 rounded-full bg-neon-900 overflow-hidden">
        <div class="h-full rounded-full progress-glow transition-all duration-1000 ease-out"
             style="width: ${r.failure_probability * 100}%;
                    background: linear-gradient(90deg, #00ff88, #ffd60a, #ff6b6b, #ff2d55);"></div>
      </div>
    </div>

    <!-- Probability Gauge -->
    <div class="neon-card p-4 mb-4">
      <div id="gauge-chart" style="height:180px"></div>
    </div>

    <!-- Risk Factors -->
    <div class="space-y-2 mb-4">
      <h4 class="font-orbitron text-xs text-neon-300/60 tracking-wider">YẾU TỐ RỦI RO</h4>
      ${r.risk_factors.map(f => `
        <div class="flex items-start gap-2 text-sm text-neon-200/80">
          <span class="text-red-400 mt-0.5">▶</span>
          <span>${f}</span>
        </div>
      `).join('')}
    </div>

    <!-- Recommendation -->
    <div class="neon-card p-4 border-l-4 ${r.risk_level === 'LOW' ? 'border-l-green-400' : 'border-l-red-400'}">
      <h4 class="font-orbitron text-xs tracking-wider mb-1 ${r.risk_level === 'LOW' ? 'text-green-400' : 'text-red-400'}">KHUYẾN NGHỊ</h4>
      <p class="text-sm text-neon-200/80">${r.recommendation}</p>
    </div>
  `;

  // Gauge chart
  const gaugeBg = isDark() ? '#0a1628' : '#f0f4f8';
  const gaugeBorder = isDark() ? '#00bfff33' : 'rgba(0,0,0,0.10)';
  const gaugeTickClr = isDark() ? '#00bfff44' : 'rgba(0,0,0,0.25)';
  const gaugeNumClr = isDark() ? '#00bfff' : '#0077b6';
  Plotly.newPlot('gauge-chart', [{
    type: 'indicator', mode: 'gauge+number',
    value: r.failure_probability * 100,
    number: { suffix: '%', font: { color: gaugeNumClr, family: 'Orbitron', size: 28 } },
    gauge: {
      axis: { range: [0, 100], tickcolor: gaugeTickClr, dtick: 20 },
      bar: { color: r.failure_probability > 0.5 ? '#ff2d55' : (isDark() ? '#00bfff' : '#0077b6') },
      bgcolor: gaugeBg,
      bordercolor: gaugeBorder,
      steps: [
        { range: [0, 15], color: isDark() ? '#00ff8822' : 'rgba(56,161,105,0.12)' },
        { range: [15, 40], color: isDark() ? '#ffd60a22' : 'rgba(214,158,46,0.12)' },
        { range: [40, 70], color: isDark() ? '#ff8c0022' : 'rgba(221,107,0,0.12)' },
        { range: [70, 100], color: isDark() ? '#ff2d5522' : 'rgba(229,62,62,0.12)' },
      ],
      threshold: {
        line: { color: '#ff2d55', width: 3 },
        thickness: 0.8, value: 50
      }
    }
  }], getLayout({
    margin: { l: 30, r: 30, t: 20, b: 10 },
    height: 180,
  }), plotConfig);
}

// ── Tabs ────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
      b.classList.add('text-neon-300/50');
    });
    btn.classList.add('active');
    btn.classList.remove('text-neon-300/50');

    document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
    document.getElementById('tab-' + btn.dataset.tab).classList.remove('hidden');

    // Lazy load
    const tab = btn.dataset.tab;
    if (tab === 'classification') loadClassification();
    else if (tab === 'clustering') loadClustering();
    else if (tab === 'anomaly') loadAnomaly();
    else if (tab === 'regression') loadRegression();
    else if (tab === 'semisupervised') loadSemiSupervised();
    else if (tab === 'scatter') loadScatter();
    else if (tab === 'gallery') loadGallery();
  });
});

// ── Lightbox ────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.remove('hidden');
  document.getElementById('lightbox').classList.add('flex');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.add('hidden');
  document.getElementById('lightbox').classList.remove('flex');
}

// ── Scroll ──────────────────────────────────────
function scrollToSection(id) {
  document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
}

// ── Particles Background ────────────────────────
function initParticles() {
  const canvas = document.getElementById('particles');
  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  const particles = [];
  const count = 50;
  for (let i = 0; i < count; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      size: Math.random() * 2 + 0.5,
      opacity: Math.random() * 0.3 + 0.1,
    });
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 191, 255, ${p.opacity})`;
      ctx.fill();
    });

    // Draw lines between close particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 191, 255, ${0.08 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(animate);
  }
  animate();

  window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
}

// ── INIT ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  applyTheme();
  initParticles();
  loadKPIs();
  loadClassification();
  loadInsights();
});
