/* ──────────────────────────────────────────────────────────────────────────
   SILVERWING // TACTICAL COMMAND — Tool Execution

   Generates labeled input fields from a capability's parameter
   schema, handles manual tool execution, and renders results.
   ────────────────────────────────────────────────────────────────────────── */

/* generate labeled input fields from the capability's parameter schema */
function renderToolFields(name) {
  const wrap = document.getElementById('toolFields');
  const ta = document.getElementById('toolArgs');
  const cap = capabilities.find(c=>c.name===name);
  const schema = (cap && cap.input_schema) || {};
  const params = Object.keys(schema);
  if (!params.length) {
    if (wrap) wrap.innerHTML = '';
    if (wrap) wrap.style.display = 'none';
    if (ta) ta.style.display = 'block';
    return;
  }
  if (ta) ta.style.display = 'none';
  if (wrap) wrap.style.display = 'block';
  if (!wrap) return;
  wrap.innerHTML = params.map(p => {
    const info = schema[p] || {};
    const desc = typeof info === 'string' ? info : (info.description || '');
    return `<label>${escHtml(p)}${desc ? ' // ' + escHtml(desc) : ''}</label>` +
      `<input class="tf-input" data-param="${escHtml(p)}" placeholder="${escHtml(desc || p)}">`;
  }).join('');
}

function renderToolSelect() {
  const sel = document.getElementById('toolSelect');
  if (!sel) return;
  sel.innerHTML = '<option value="">-- SELECT NODE --</option>' +
    sortedByUsage(capabilities).map(c=>`<option value="${c.name}">${c.name}</option>`).join('');
  if (selectedCapName && capabilities.some(c=>c.name===selectedCapName)) {
    sel.value = selectedCapName;
  }
}

/* dropdown change handler: remember selection, then rebuild parameter fields */
function onToolSelectChange(v) {
  selectedCapName = v;
  renderToolFields(v);
}

/* ── tool execute ── */
async function executeTool() {
  const tool = document.getElementById('toolSelect').value;
  if (!tool) return;
  const args = {};
  const fields = document.querySelectorAll('#toolFields .tf-input');
  if (fields.length) {
    fields.forEach(f => {
      const v = f.value.trim();
      if (v) args[f.dataset.param] = v;
    });
  } else {
    const argsStr = document.getElementById('toolArgs').value.trim();
    if (argsStr) {
      argsStr.split(',').forEach(p => {
        const [k,...v] = p.split('=');
        if (k && k.trim()) args[k.trim()] = v.join('=').trim();
      });
    }
  }
  const resultEl = document.getElementById('toolResult');
  if (!resultEl) return;
  resultEl.style.display = 'block';
  resultEl.className = 'tool-result';
  resultEl.textContent = '> EXECUTING...';
  fireFlash();

  try {
    const r = await fetch(`${API}/v1/tools/execute`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ tool, arguments: args })
    });
    const d = await r.json();
    if (d.success && d.data) {
      recordUsage(tool);
      resultEl.className = `tool-result ${d.data.success?'success':'error'}`;
      resultEl.textContent = d.data.success ? d.data.output : `ERROR: ${d.data.error}`;
      addAuditEvent({
        action: 'tool_execute',
        capability_id: tool,
        status: d.data.success ? 'success' : 'error',
        detail: d.data.success ? d.data.output : d.data.error,
        timestamp: Date.now()/1000
      });
    } else {
      resultEl.className = 'tool-result error';
      resultEl.textContent = `ERROR: ${d.error}`;
    }
  } catch(e) {
    resultEl.className = 'tool-result error';
    resultEl.textContent = `UPLINK FAILURE: ${e.message}`;
  }
}
