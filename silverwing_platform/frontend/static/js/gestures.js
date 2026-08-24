/* ──────────────────────────────────────────────────────────────────────────
   SILVERWING // TACTICAL COMMAND — Gesture Control

   Loads the gesture -> action mapping table, subsystem availability,
   live system stats, and handles gesture execution + IoT commands.
   ────────────────────────────────────────────────────────────────────────── */

/* Fetch the static gesture -> action mapping table */
async function loadGestures() {
  try {
    const r = await fetch(`${API}/v1/gestures`);
    const d = await r.json();
    if (d.success && d.data && d.data.gestures) {
      gestureMapping = d.data.gestures;
      renderGestureTable(gestureMapping);
      populateGestureExecSelect(gestureMapping);
    }
  } catch(e) {}
}

/* Populate the dropdown for manual gesture execution */
function populateGestureExecSelect(gestures) {
  const sel = document.getElementById('gestureExecSelect');
  if (!sel) return;
  sel.innerHTML = '<option value="">-- SELECT GESTURE --</option>' +
    gestures.map(g => `<option value="${g.gesture}">${g.gesture} // ${g.risk_level}</option>`).join('');
}

/* Render the gesture mapping table */
function renderGestureTable(gestures) {
  const wrap = document.getElementById('gestureTable');
  if (!wrap) return;
  if (!gestures.length) {
    wrap.innerHTML = '<div class="empty" style="grid-column:1/-1;padding:20px 0">' +
      '<div class="icon">&#9881;</div>NO GESTURES REGISTERED<span class="blink">_</span></div>';
    return;
  }
  let html = '<div class="gt-header">GESTURE</div>' +
    '<div class="gt-header">ACTION</div>' +
    '<div class="gt-header">RISK</div>';
  for (const g of gestures) {
    const risk = g.risk_level || 'low';
    html += `<div class="gesture-row">` +
      `<span class="g-name">${g.gesture || ''}</span>` +
      `<span class="g-action">${escHtml(g.action || '')}</span>` +
      `<span class="g-risk ${risk}">${risk}</span>` +
      `</div>`;
  }
  wrap.innerHTML = html;
}

/* Fetch subsystem availability + config snapshot */
async function loadGestureStatus() {
  try {
    const r = await fetch(`${API}/v1/gestures/status`);
    const d = await r.json();
    if (d.success && d.data) {
      gestureStatusCache = d.data;
      renderSubsystemStatus(d.data);
      updateGestureHeaderChip(d.data);
    }
  } catch(e) {
    const chip = document.getElementById('gestureChip');
    if (chip) { chip.textContent = 'GESTURE: OFFLINE'; chip.classList.add('off'); }
    const bk = document.getElementById('gestureBackendChip');
    if (bk) bk.textContent = 'OFFLINE';
  }
}

/* Update the header HUD gesture chip */
function updateGestureHeaderChip(status) {
  const chip = document.getElementById('gestureChip');
  if (!chip) return;
  if (!status.available) {
    chip.textContent = 'GESTURE: OFFLINE';
    chip.classList.remove('off');
    chip.classList.add('off');
    return;
  }
  const mp = status.subsystems && status.subsystems.mediapipe;
  const backend = (mp && mp.backend) || '---';
  const ok = mp && mp.available;
  chip.textContent = `GESTURE: ${ok ? 'ONLINE' : 'DEGRADED'}`;
  chip.classList.remove('off');
  chip.classList.toggle('off', !ok);
}

/* Render subsystem status chips */
function renderSubsystemStatus(status) {
  const grid = document.getElementById('subsystemGrid');
  if (!grid) return;
  const bk = document.getElementById('gestureBackendChip');
  if (bk) {
    if (!status.available) {
      bk.textContent = 'OFFLINE';
      bk.style.borderColor = 'var(--crimson)';
      bk.style.color = 'var(--crimson)';
    } else {
      const mp = status.subsystems && status.subsystems.mediapipe;
      const backend = (mp && mp.backend) || '---';
      bk.textContent = `${backend.toUpperCase()} // ${status.subsystems && status.subsystems.opencv ? 'OK' : 'FAUL'}`;
      bk.style.borderColor = 'var(--volt)';
      bk.style.color = 'var(--volt)';
    }
  }
  if (!status.available) {
    grid.innerHTML = '<div class="subsystem-chip off">' +
      '<span class="sc-dot off"></span>gesture_os module not importable</div>';
    return;
  }
  const subs = status.subsystems || {};
  const labels = {
    mediapipe: 'MediaPipe',
    opencv: 'OpenCV',
    numpy: 'NumPy',
    pyautogui: 'PyAutoGUI',
    yolo: 'YOLOv8',
    socketio: 'SocketIO',
    tkinter: 'Tkinter',
  };
  let html = '';
  for (const [key, val] of Object.entries(subs)) {
    const label = labels[key] || key;
    let state, text;
    if (typeof val === 'object' && val !== null) {
      state = val.available;
      text = val.backend || '';
    } else {
      state = val;
      text = '';
    }
    const cls = state ? 'on' : 'off';
    const dot = state ? 'on' : 'off';
    html += `<div class="subsystem-chip ${cls}">` +
      `<span class="sc-dot ${dot}"></span>${escHtml(label)}` +
      (text ? `<span class="sc-backend">${escHtml(text)}</span>` : '') +
      `</div>`;
  }
  grid.innerHTML = html;
}

/* Fetch and render live system stats */
async function loadGestureStats() {
  try {
    const r = await fetch(`${API}/v1/gestures/stats`);
    const d = await r.json();
    if (d.success && d.data) {
      renderStats(d.data);
    }
  } catch(e) {}
}

/* Render system stats into the stat cells with temperature-based colour */
function renderStats(stats) {
  if (stats.error) {
    if (document.getElementById('gsCpu')) document.getElementById('gsCpu').textContent = 'ERR';
    if (document.getElementById('gsMem')) document.getElementById('gsMem').textContent = 'ERR';
    return;
  }
  const cpuEl = document.getElementById('gsCpu');
  const memEl = document.getElementById('gsMem');
  const batEl = document.getElementById('gsBat');
  const netEl = document.getElementById('gsNet');
  const ipEl = document.getElementById('gsIp');

  function setVal(el, val) { if (el) el.textContent = val; }
  function setPctClass(el, pctStr) {
    if (!el) return;
    el.classList.remove('warm', 'hot');
    const num = parseFloat(pctStr);
    if (num >= 90) el.classList.add('hot');
    else if (num >= 70) el.classList.add('warm');
  }

  const cpu = stats.cpu || '--';
  const mem = stats.mem || '--';
  const bat = stats.bat || '--';
  const net = stats.net || '--';
  const ip = stats.ip || '--';

  setVal(cpuEl, cpu); setPctClass(cpuEl, String(cpu));
  setVal(memEl, mem); setPctClass(memEl, String(mem));
  setVal(batEl, bat);
  setVal(netEl, net);
  setVal(ipEl, ip);

  const gestEl = document.getElementById('gsGest');
  const handsEl = document.getElementById('gsHands');
  if (gestEl) gestEl.textContent = (gestureStatusCache && gestureStatusCache.available) ? 'LIVE' : 'DOWN';
  if (handsEl) handsEl.textContent = (gestureStatusCache && gestureStatusCache.available) ? '2' : '0';
}

/* Execute a gesture via /v1/tools/execute */
async function executeGesture() {
  const sel = document.getElementById('gestureExecSelect');
  if (!sel || !sel.value) return;
  const gesture = sel.value;
  fireFlash();
  addAuditEvent({ action: 'gesture_execute', capability_id: 'gesture_execute', status: 'pending', detail: `gesture=${gesture}`, timestamp: Date.now() / 1000 });
  try {
    const r = await fetch(`${API}/v1/tools/execute`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ tool: 'gesture_execute', arguments: { gesture: gesture } })
    });
    const d = await r.json();
    if (d.success && d.data) {
      const output = d.data.output || '';
      addAuditEvent({ action: 'gesture_execute', capability_id: 'gesture_execute', status: d.data.success ? 'success' : 'error', detail: output, timestamp: Date.now() / 1000 });
      addMessage('system', `> GESTURE: ${gesture} ${d.data.success ? '[OK]' : '[FAULT]'}\n${output}`);
    } else {
      addAuditEvent({ action: 'gesture_execute', capability_id: 'gesture_execute', status: 'error', detail: d.error || 'unknown', timestamp: Date.now() / 1000 });
    }
  } catch(e) {
    addMessage('system', `> GESTURE UPLINK FAILURE: ${e.message}`);
  }
}

/* Send an IoT command via /v1/tools/execute */
async function sendIotCommand() {
  const cmd = document.getElementById('gestureIotCmd').value.trim();
  const payload = document.getElementById('gestureIotPayload').value.trim() || '{}';
  if (!cmd) return;
  fireFlash();
  addAuditEvent({ action: 'iot_send_command', capability_id: 'iot_send_command', status: 'pending', detail: `command=${cmd}`, timestamp: Date.now() / 1000 });
  try {
    const r = await fetch(`${API}/v1/tools/execute`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ tool: 'iot_send_command', arguments: { command: cmd, payload: payload } })
    });
    const d = await r.json();
    if (d.success && d.data) {
      const output = d.data.output || '';
      addAuditEvent({ action: 'iot_send_command', capability_id: 'iot_send_command', status: d.data.success ? 'success' : 'error', detail: output, timestamp: Date.now() / 1000 });
      addMessage('system', `> IoT CMD: ${cmd} ${d.data.success ? '[OK]' : '[FAULT]'}\n${output}`);
    } else {
      addAuditEvent({ action: 'iot_send_command', capability_id: 'iot_send_command', status: 'error', detail: d.error || 'unknown', timestamp: Date.now() / 1000 });
    }
  } catch(e) {
    addMessage('system', `> IoT UPLINK FAILURE: ${e.message}`);
  }
}
