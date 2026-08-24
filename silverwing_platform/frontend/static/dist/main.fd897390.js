/* ──────────────────────────────────────────────────────────────────────────
   SILVERWING // TACTICAL COMMAND — Main Entry Point

   Health checks, initial data loading, and periodic polling
   to keep the UI synchronised with the platform backend.
   ────────────────────────────────────────────────────────────────────────── */

/* ── health check ── */
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    if (dot) dot.className = 'dot ' + (d.success ? 'on' : 'off');
    if (text) text.textContent = d.success ? 'UPLINK ACTIVE' : 'FAULT';
  } catch(e) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    if (dot) dot.className = 'dot off';
    if (text) text.textContent = 'OFFLINE';
  }
  try {
    const r2 = await fetch(`${API}/info`);
    const d2 = await r2.json();
    const mid = (d2.success && d2.data && (d2.data.model_id || d2.data.model)) || null;
    const chip = document.getElementById('modelChip');
    if (chip) chip.textContent = 'MODEL: ' + (mid ? String(mid).toUpperCase().slice(0, 18) : 'STANDBY');
  } catch(e) {}
}

/* ── init ── */
updateAdaptChip();
checkHealth();
loadCapabilities();
loadGestures();
loadGestureStatus();
loadGestureStats();

setInterval(checkHealth, 10000);
setInterval(loadCapabilities, 15000);
setInterval(loadGestureStatus, 15000);
setInterval(loadGestureStats, 5000);

setTimeout(() => {
  const input = document.getElementById('chatInput');
  if (input) input.focus();
}, 1600);
