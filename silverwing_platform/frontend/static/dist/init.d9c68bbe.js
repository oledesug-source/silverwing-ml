/* ──────────────────────────────────────────────────────────────────────────
   SILVERWING // TACTICAL COMMAND — Boot & Visual Effects

   Boot sequence animation, HUD clock, brand glitch,
   screen flash, and other visual init.
   ────────────────────────────────────────────────────────────────────────── */

/* ── boot sequence ── */
(function boot(){
  const lines = [
    '> initializing core systems ............ <span class="ok">ONLINE</span>',
    '> arming capability registry ........... <span class="ok">ARMED</span>',
    '> establishing platform uplink ......... <span class="ok">LINKED</span>',
    '> <span class="dim">all systems nominal</span>',
  ];
  const el = document.getElementById('bootLines');
  if (!el) return;
  let i = 0;
  const iv = setInterval(() => {
    if (i >= lines.length) {
      clearInterval(iv);
      setTimeout(() => document.getElementById('boot').classList.add('done'), 320);
      return;
    }
    el.innerHTML += lines[i++] + '<br>';
  }, 210);
})();

/* ── occasional brand glitch ── */
setInterval(() => {
  const t = document.getElementById('brandTitle');
  if (t) {
    t.classList.add('glitching');
    setTimeout(() => t.classList.remove('glitching'), 300);
  }
}, 7000);

/* ── HUD clock ── */
setInterval(() => {
  const el = document.getElementById('hudClock');
  if (el) el.textContent = new Date().toTimeString().slice(0, 8);
}, 1000);

/* ── screen flash ── */
function fireFlash() {
  const f = document.getElementById('flash');
  if (!f) return;
  f.classList.add('firing');
  requestAnimationFrame(() => requestAnimationFrame(() => f.classList.remove('firing')));
}

/* ── first clock tick ── */
(function initClock(){
  const el = document.getElementById('hudClock');
  if (el) el.textContent = new Date().toTimeString().slice(0, 8);
})();
