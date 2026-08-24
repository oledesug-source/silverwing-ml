/* ──────────────────────────────────────────────────────────────────────────
   SILVERWING // TACTICAL COMMAND — Capabilities & Adaptive Layer

   Loads capabilities from the registry, renders the arsenal panel,
   and maintains the adaptive ranking / next-action prediction model.
   ────────────────────────────────────────────────────────────────────────── */

function toggleAdapt() {
  adaptEnabled = !adaptEnabled;
  try { localStorage.setItem('sw_adapt', adaptEnabled ? 'on' : 'off'); } catch(e) {}
  updateAdaptChip();
  renderCapabilities();
}

function updateAdaptChip() {
  const chip = document.getElementById('adaptChip');
  if (!chip) return;
  chip.textContent = 'ADAPT: ' + (adaptEnabled ? 'ON' : 'OFF');
  chip.classList.toggle('off', !adaptEnabled);
}

function sessionLastTool() {
  try { return sessionStorage.getItem('sw_last_tool'); } catch(e) { return null; }
}

function recordUsage(name) {
  const u = capUsage[name] || { count: 0, last: 0 };
  u.count += 1;
  u.last = Date.now();
  capUsage[name] = u;
  const prev = sessionLastTool();
  if (prev && prev !== name) {
    const key = prev + '>' + name;
    capTransitions[key] = (capTransitions[key] || 0) + 1;
  }
  try {
    localStorage.setItem('sw_cap_usage', JSON.stringify(capUsage));
    localStorage.setItem('sw_cap_transitions', JSON.stringify(capTransitions));
    sessionStorage.setItem('sw_last_tool', name);
  } catch(e) {}
  if (capabilities.length) renderCapabilities();
}

function usageScore(name) {
  const u = capUsage[name];
  if (!u) return 0;
  const recentHours = (Date.now() - u.last) / 3600000;
  return u.count + (recentHours < 24 ? 3 : 0);
}

/* P(next=name | last tool in this session), 0..1 */
function nextProbability(name) {
  if (!adaptEnabled) return 0;
  const prev = sessionLastTool();
  if (!prev || prev === name) return 0;
  let total = 0;
  const prefix = prev + '>';
  for (const k in capTransitions) {
    if (k.indexOf(prefix) === 0) total += capTransitions[k];
  }
  if (!total) return 0;
  return (capTransitions[prefix + name] || 0) / total;
}

function predictedNext() {
  if (!adaptEnabled) return null;
  let best = null, bestP = 0;
  for (const c of capabilities) {
    const p = nextProbability(c.name);
    if (p > bestP) { bestP = p; best = c.name; }
  }
  return bestP >= 0.34 ? best : null;
}

function sortedByUsage(caps) {
  if (!adaptEnabled) return caps.slice();
  const prev = sessionLastTool();
  return caps.map((c, i) => ({
    c, i,
    s: usageScore(c.name) + (prev ? nextProbability(c.name) * 4 : 0),
  }))
    .sort((a, b) => (b.s - a.s) || (a.i - b.i))
    .map(x => x.c);
}

/* ── load capabilities ── */
async function loadCapabilities() {
  try {
    const r = await fetch(`${API}/v1/capabilities`);
    const d = await r.json();
    if (d.success) {
      capabilities = d.data;
      renderCapabilities();
      renderToolSelect();
    }
  } catch(e) {}
}

function renderCapabilities() {
  const el = document.getElementById('capList');
  const countEl = document.getElementById('capCount');
  if (countEl) countEl.textContent = capabilities.length;
  if (!el) return;
  if (!capabilities.length) {
    el.innerHTML = '<div class="empty">REGISTRY EMPTY<span class="blink">_</span></div>';
    return;
  }
  const PERM = {low:'L0 // OBSERVE', medium:'L1 // WRITE', high:'L2 // EXECUTE', critical:'L4 // ADMIN'};
  const nextCap = predictedNext();
  el.innerHTML = sortedByUsage(capabilities).map(c => {
    const tags = (c.tags||[]).map(t=>`<span class="cap-tag">${t}</span>`).join('');
    const freq = adaptEnabled && usageScore(c.name) >= 2 ? '<span class="cap-tag freq-tag">&#9733; FREQUENT</span>' : '';
    const next = adaptEnabled && c.name === nextCap ? '<span class="cap-tag next-tag">&#9654; NEXT</span>' : '';
    return `<div class="cap-card risk-${c.risk_level||'low'}${c.name===selectedCapName?' active':''}" data-name="${c.name}" onclick="selectCap('${c.name}', this)">
      <div class="cap-name">${c.name} <span class="badge badge-${c.risk_level||'low'}"> ${(c.risk_level||'low').toUpperCase()}</span></div>
      <div class="cap-desc">${c.description||''}</div>
      <div class="cap-meta"><span>v${c.version||'1.0.0'}</span><span>${c.timeout_seconds||30}s</span>${next}${freq}${tags}</div>
      <div class="perm-line">&#9654; PERMISSION: ${PERM[c.risk_level||'low'] || 'L0 // OBSERVE'}</div>
    </div>`;
  }).join('');
  renderNextAction();
}

/* ── next-action shortcut strip ── */
function renderNextAction() {
  const slot = document.getElementById('nextActionSlot');
  const whyEl = document.getElementById('nextWhy');
  if (!slot) return;
  if (!adaptEnabled) {
    slot.innerHTML = 'ADAPTIVE LAYER OFF';
    if (whyEl) whyEl.textContent = '';
    return;
  }
  const cap = predictedNext();
  if (!cap) {
    slot.innerHTML = 'STANDBY<span class="blink">_</span>';
    if (whyEl) whyEl.textContent = '';
    return;
  }
  slot.innerHTML = '';
  const b = document.createElement('button');
  b.className = 'qa-btn';
  b.textContent = '\u25B8 ' + cap;
  b.onclick = () => quickLaunch(cap);
  slot.appendChild(b);
  const x = document.createElement('button');
  x.className = 'qa-dismiss';
  x.textContent = '\u2715';
  x.title = 'Wrong prediction? Dismiss to penalize this suggestion';
  x.setAttribute('aria-label', 'Dismiss prediction for ' + cap);
  x.onclick = () => penalizePrediction(cap);
  slot.appendChild(x);
  const prev = sessionLastTool();
  const p = nextProbability(cap);
  const n = capTransitions[prev + '>' + cap] || 0;
  if (whyEl) whyEl.textContent = `P=${p.toFixed(2)} \u00B7 ${n}\u00D7 after ${prev}`;
}

function penalizePrediction(cap) {
  const prev = sessionLastTool();
  if (!prev) return;
  const key = prev + '>' + cap;
  if (capTransitions[key]) {
    capTransitions[key] -= 1;
    if (capTransitions[key] <= 0) delete capTransitions[key];
    try { localStorage.setItem('sw_cap_transitions', JSON.stringify(capTransitions)); } catch(e) {}
  }
  renderCapabilities();
}

function quickLaunch(name) {
  const card = document.querySelector(`.cap-card[data-name="${name}"]`);
  selectCap(name, card);
  const args = document.getElementById('toolArgs');
  if (args) args.focus();
}

function selectCap(name, cardEl) {
  selectedCapName = name;
  document.querySelectorAll('.cap-card').forEach(el => el.classList.remove('active'));
  if (cardEl) cardEl.classList.add('active');
  const cap = capabilities.find(c=>c.name===name);
  if (cap) {
    const select = document.getElementById('toolSelect');
    if (select) select.value = name;
    renderToolFields(name);
  }
}
