/* ──────────────────────────────────────────────────────────────────────────
   SILVERWING // TACTICAL COMMAND — Utilities & State

   Shared constants, global state, and helper functions used across
   all other JS modules.
   ────────────────────────────────────────────────────────────────────────── */

const API = '';

let capabilities = [];
let auditEvents = [];
let selectedCapName = '';
let gestureMapping = [];
let gestureStatusCache = null;

/* ── Adaptive layer: usage telemetry (local, transparent) ── */
/* Two signals reorder capability nodes, both surfaced in the UI
   ('FREQUENT' / 'NEXT') so no adaptation is hidden:
     1. Frequency + recency of each capability (lifetime, localStorage)
     2. Session sequence: transition counts prev-tool -> next-tool,
        the NX-style "likely next command" pattern (bigram model). */
let capUsage = {};
let capTransitions = {};
let adaptEnabled = true;

try { capUsage = JSON.parse(localStorage.getItem('sw_cap_usage') || '{}'); } catch(e) { capUsage = {}; }
try { capTransitions = JSON.parse(localStorage.getItem('sw_cap_transitions') || '{}'); } catch(e) { capTransitions = {}; }
try { adaptEnabled = localStorage.getItem('sw_adapt') !== 'off'; } catch(e) {}

/* ── helpers ── */
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setStat(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = val;
  if (id === 'metaTime') {
    const ms = parseInt(String(val)) || 0;
    el.classList.toggle('alert', ms > 2000);
  }
}
