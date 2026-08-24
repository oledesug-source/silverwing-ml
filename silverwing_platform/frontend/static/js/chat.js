/* ──────────────────────────────────────────────────────────────────────────
   SILVERWING // TACTICAL COMMAND — Chat & Messaging

   Chat input handling, message rendering with decrypt-style
   type-reveal for assistant transmissions, and stat updates.
   ────────────────────────────────────────────────────────────────────────── */

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addMessage('user', msg);
  fireFlash();
  const sendBtn = document.getElementById('sendBtn');
  if (sendBtn) sendBtn.disabled = true;

  try {
    const r = await fetch(`${API}/v1/chat`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        message: msg,
        max_rounds: parseInt(document.getElementById('maxRounds').value) || 5
      })
    });
    const d = await r.json();
    if (d.success && d.data) {
      const data = d.data;
      for (let i = 0; i < (data.tool_calls||[]).length; i++) {
        const tc = data.tool_calls[i];
        const tr = (data.tool_results||[])[i];
        recordUsage(tc.tool);
        addToolMessage(tc, tr);
      }
      if (data.text) addMessage('assistant', data.text);
      setStat('metaRounds', data.rounds || 0);
      setStat('metaTime', Math.round((data.elapsed_seconds||0)*1000) + 'MS');
      setStat('metaCalls', (data.tool_calls||[]).length);
      if (data.audit_events) {
        data.audit_events.forEach(e => addAuditEvent(e));
      }
    } else {
      addMessage('system', `ERROR: ${d.error || 'UNKNOWN FAULT'}`);
    }
  } catch(e) {
    addMessage('system', `UPLINK FAILURE: ${e.message}`);
  }
  if (sendBtn) sendBtn.disabled = false;
  input.focus();
}

function addMessage(role, text) {
  const el = document.getElementById('chatMessages');
  if (!el) return;
  const div = document.createElement('div');
  div.className = `msg msg-${role}`;
  el.appendChild(div);
  if (role === 'assistant') {
    decryptType(div, text);
  } else {
    div.textContent = text;
  }
  el.scrollTop = el.scrollHeight;
}

/* decrypt-style reveal for assistant transmissions */
function decryptType(el, text) {
  const glyphs = '!<>-_\\/[]{}=+*^?#@%&';
  let shown = 0;
  const step = Math.max(1, Math.ceil(text.length / 36));
  const iv = setInterval(() => {
    shown = Math.min(shown + step, text.length);
    let out = text.slice(0, shown);
    if (shown < text.length) {
      for (let i = 0; i < 6; i++) out += glyphs[Math.floor(Math.random()*glyphs.length)];
    }
    el.textContent = out;
    const box = document.getElementById('chatMessages');
    if (box) box.scrollTop = box.scrollHeight;
    if (shown >= text.length) { el.textContent = text; clearInterval(iv); }
  }, 24);
}

function addToolMessage(call, result) {
  const el = document.getElementById('chatMessages');
  if (!el) return;
  const div = document.createElement('div');
  div.className = 'msg msg-tool';
  const status = result ? (result.success ? 'OK' : 'FAULT') : 'PENDING';
  const output = result ? (result.success ? result.output : result.error) : '';
  div.innerHTML = `<div class="tool-header">&lt;tool:${call.tool}&gt; [${status}]</div>` +
    `<div class="${result&&result.success?'tool-output':'tool-error'}">${escHtml(output||'')}</div>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function clearChat() {
  const el = document.getElementById('chatMessages');
  if (el) el.innerHTML = '<div class="msg msg-system">// CHANNEL PURGED //</div>';
  setStat('metaRounds', '0');
  setStat('metaTime', '0MS');
  setStat('metaCalls', '0');
}

/* ── permission change ── */
function onPermChange() {
  const level = document.getElementById('permLevel').value;
  addMessage('system', `// CLEARANCE LEVEL SET: ${level} //`);
}
