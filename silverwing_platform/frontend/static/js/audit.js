/* ──────────────────────────────────────────────────────────────────────────
   SILVERWING // TACTICAL COMMAND — Audit Trail

   Appends, renders, and clears audit events with status-based
   colour coding and timestamp formatting.
   ────────────────────────────────────────────────────────────────────────── */

function addAuditEvent(e) {
  auditEvents.unshift(e);
  if (auditEvents.length > 100) auditEvents.pop();
  renderAudit();
}

function renderAudit() {
  const el = document.getElementById('auditList');
  if (!el) return;
  if (!auditEvents.length) {
    el.innerHTML = '<div class="empty">NO EVENTS LOGGED<span class="blink">_</span></div>';
    return;
  }
  el.innerHTML = auditEvents.slice(0, 50).map(e => {
    const status = e.status || 'pending';
    const time = e.timestamp ? new Date(e.timestamp*1000).toTimeString().slice(0,8) : '--:--:--';
    const elapsed = e.elapsed_ms ? `${Math.round(e.elapsed_ms)}ms` : '';
    return `<div class="audit-entry st-${status}">
      <div class="ae-line1">
        <span class="ae-time">[${time}]</span>
        <span class="ae-status-${status}">[${String(status).toUpperCase()}]</span>
        <span class="ae-action">${escHtml(e.action||'')}</span>
        ${e.capability_id?`<span style="color:var(--text3)">:: ${escHtml(e.capability_id)}</span>`:''}
      </div>
      ${e.detail?`<div class="ae-detail">&#9492; ${escHtml(e.detail.substring(0,180))}</div>`:''}
      ${elapsed||e.request_id?`<div class="ae-detail">${elapsed} ${e.request_id||''}</div>`:''}
    </div>`;
  }).join('');
}

function clearAudit() {
  auditEvents = [];
  renderAudit();
}
