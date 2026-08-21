/* =========================================================================
   Silverwing Workspace — Frontend Interaction Logic
   Handles: WebSocket, drag-and-drop workflow canvas, voice input,
   gesture detection, 3D viewport (three.js), terminal panel management.
   ========================================================================= */

(function () {
  "use strict";

  // ---- Session management ----
  const SESSION_ID = "session-" + Math.random().toString(36).slice(2, 10);
  const WS_PROTOCOL = location.protocol === "https:" ? "wss:" : "ws:";
  const WS_URL = `${WS_PROTOCOL}//${location.host}/ws/${SESSION_ID}`;

  let socket = null;
  let isConnected = false;
  let currentTerminal = "console";

  // ---- DOM helpers ----
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // ---- WebSocket connection ----
  function initWebSocket() {
    socket = new WebSocket(WS_URL);

    socket.addEventListener("open", () => {
      isConnected = true;
      updateConnectionChip(true);
      addTerminalOutput("system", "WebSocket connected");
    });

    socket.addEventListener("message", (event) => {
      const msg = JSON.parse(event.data);
      handleSocketMessage(msg);
    });

    socket.addEventListener("close", () => {
      isConnected = false;
      updateConnectionChip(false);
      addTerminalOutput("system", "WebSocket disconnected. Retrying...");
      setTimeout(initWebSocket, 3000);
    });

    socket.addEventListener("error", () => {
      addTerminalOutput("system", "WebSocket error");
    });
  }

  function updateConnectionChip(connected) {
    const chip = $("#chip-connection");
    if (chip) {
      if (connected) {
        chip.className = "sw-chip sw-chip--connected";
        chip.innerHTML = '<span class="sw-dot sw-dot--green"></span> Connected';
      } else {
        chip.className = "sw-chip sw-chip--warning";
        chip.innerHTML = '<span class="sw-dot sw-dot--yellow"></span> Reconnecting...';
      }
    }
  }

  function handleSocketMessage(msg) {
    switch (msg.type) {
      case "session_ready":
        addTerminalOutput("system", `Session ready: ${msg.session_id}`);
        addTerminalOutput("system", `Tools: ${msg.tools.join(", ")}`);
        break;

      case "typing":
        if (msg.status === "started") {
          addTerminalOutput("system", "Agent is typing…");
        }
        break;

      case "response":
        addChatMessage("agent", msg.text);
        addTerminalOutput("console", msg.text);
        if (msg.success === false) {
          addTerminalOutput("console", `Error: ${msg.error}`);
        }
        break;

      case "tool_call":
        addTerminalOutput("tools",
          `Tool: ${msg.data.tool_name}\n` +
          `Args: ${JSON.stringify(msg.data.arguments, null, 2)}\n` +
          `Result: ${msg.data.result?.output || msg.data.result?.error || "N/A"}`
        );
        break;

      case "audit":
        addTerminalOutput("console", `Audit: ${msg.data.action} — ${msg.data.status}`);
        break;

      case "tools":
        renderToolList(msg.tools);
        break;

      case "error":
        addTerminalOutput("console", `Error: ${msg.error}`);
        break;

      case "pong":
        addTerminalOutput("system", "Pong received");
        break;
    }
  }

  // ---- Chat ----
  function sendChatMessage() {
    const input = $("#message-input");
    const text = input.value.trim();
    if (!text || !isConnected) return;

    addChatMessage("user", text);
    socket.send(JSON.stringify({
      action: "message",
      content: text,
    }));
    input.value = "";
    resizeInput(input);
  }

  function addChatMessage(role, text) {
    const container = $("#chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = `sw-message ${role === "user" ? "sw-message--user" : ""}`;

    const avatar = document.createElement("div");
    avatar.className = "sw-avatar";
    avatar.textContent = role === "user" ? "👤" : "🤖";

    const bubble = document.createElement("div");
    bubble.className = `sw-bubble ${role === "user" ? "" : "sw-bubble--agent"}`;
    bubble.classList.add("sw-agent-message");

    // Auto-link and format
    bubble.innerHTML = formatMessage(text);

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
  }

  function formatMessage(text) {
    return text
      .replace(/\n/g, "<br>")
      .replace(/`([^`]+)`/g, '<code style="background:var(--sw-color-bg);padding:2px 6px;border-radius:4px;">$1</code>')
      .replace(/```(\w+)\n([\s\S]*?)```/g, '<pre class="language-$1"><code>$2</code></pre>');
  }

  // ---- Terminal panels ----
  function addTerminalOutput(panel, text) {
    const terminal = $(`#terminal-${panel}`);
    if (!terminal) return;
    const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
    terminal.innerHTML += `<span class="sw-terminal-output">[${ts}] ${escapeHtml(text)}</span>\n`;
    terminal.scrollTop = terminal.scrollHeight;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function switchTerminal(panel) {
    currentTerminal = panel;
    $$(".sw-terminal-tab").forEach((t) => t.classList.remove("sw-terminal-tab--active"));
    $(`[data-terminal="${panel}"]`).classList.add("sw-terminal-tab--active");
    $$(".sw-terminal").forEach((t) => t.classList.remove("sw-terminal--active"));
    $(`#terminal-${panel}`).classList.add("sw-terminal--active");
  }

  // ---- Tool list ----
  function renderToolList(tools) {
    const container = $("#tool-list");
    if (!container) return;
    container.innerHTML = "";

    tools.forEach((tool) => {
      const div = document.createElement("div");
      div.className = "sw-tool-item";
      div.innerHTML = `
        <div class="sw-tool-name">${tool.name}</div>
        <div style="font-size:var(--sw-font-size-xs);color:var(--sw-color-text-muted);">
          ${tool.description || ""}
        </div>
        <div style="margin-top:4px;">
          <span class="sw-chip sw-chip--sm"
            style="background:rgba(${tool.risk_level === "high" ? "239,68,68" : tool.risk_level === "medium" ? "234,179,8" : "34,197,94"},0.15)">
            ${tool.risk_level}
          </span>
          <span class="sw-chip sw-chip--sm sw-chip--muted">${tool.permission_required}</span>
        </div>
      `;
      container.appendChild(div);
    });
  }

  // ---- File tree ----
  function renderFileTree(files) {
    const container = $("#file-tree");
    if (!container) return;
    container.innerHTML = "";
    const ul = document.createElement("ul");
    ul.className = "sw-tree";
    files.forEach((f) => {
      const li = document.createElement("li");
      li.className = "sw-tree-item";
      const icon = f.endsWith("/") ? "📁" : "📄";
      li.innerHTML = `<span>${icon} ${f}</span>`;
      ul.appendChild(li);
    });
    container.appendChild(ul);
  }

  // ---- Sidebar toggle ----
  function toggleSidebar(side) {
    const sidebar = side === "left" ? $("#sidebar-left") : $("#sidebar-right");
    if (sidebar) {
      sidebar.classList.toggle("sw-sidebar-collapsed");
    }
  }

  // ---- Textarea auto-resize ----
  function resizeInput(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.max(40, textarea.scrollHeight) + "px";
  }

  // ---- Voice Input ----
  let recognition = null;
  let isListening = false;

  function initVoice() {
    if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
      $("#voice-status").textContent = "Voice not supported";
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      $("#message-input").value = transcript;
      resizeInput($("#message-input"));
    };

    recognition.onerror = () => {
      $("#voice-status").textContent = "Voice error";
      isListening = false;
    };

    recognition.onend = () => {
      isListening = false;
      $("#voice-status").textContent = "Not listening";
    };
  }

  function toggleVoice() {
    if (!recognition) {
      $("#voice-status").textContent = "Voice not available";
      return;
    }
    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
      isListening = true;
      $("#voice-status").textContent = "Listening…";
    }
  }

  // ---- Gesture Detection ----
  let gestureStart = null;

  function initGestures() {
    const workspace = $(".sw-main");
    if (!workspace) return;

    workspace.addEventListener("touchstart", (e) => {
      if (e.touches.length === 2) {
        gestureStart = {
          x: e.touches[0].clientX + e.touches[1].clientX,
          y: e.touches[0].clientY + e.touches[1].clientY,
        };
      }
    });

    workspace.addEventListener("touchmove", (e) => {
      if (e.touches.length === 2 && gestureStart) {
        const dx = (e.touches[0].clientX + e.touches[1].clientX) - gestureStart.x;
        const dy = (e.touches[0].clientY + e.touches[1].clientY) - gestureStart.y;
        // Two-finger swipe to toggle canvas
        if (Math.abs(dx) > 50) {
          toggleCanvas(true);
        }
      }
    });
  }

  // ---- Drag-and-Drop Workflow Canvas ----
  let canvasNodes = [];
  let canvasEdges = [];

  function toggleCanvas(show) {
    const overlay = $("#canvas-overlay");
    if (!overlay) return;
    overlay.classList.toggle("sw-canvas-overlay--active", show);
    if (show) {
      renderCanvas();
    }
  }

  function renderCanvas() {
    const svg = $("#canvas-svg");
    if (!svg) return;
    svg.innerHTML = "";

    // Draw edges
    canvasEdges.forEach((edge) => {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", edge.from.x);
      line.setAttribute("y1", edge.from.y);
      line.setAttribute("x2", edge.to.x);
      line.setAttribute("y2", edge.to.y);
      line.setAttribute("class", "edge");
      svg.appendChild(line);
    });

    // Draw nodes
    canvasNodes.forEach((node) => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.className = "node";
      g.setAttribute("transform", `translate(${node.x}, ${node.y})`);

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("width", 120);
      rect.setAttribute("height", 50);
      rect.setAttribute("rx", 8);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", 60);
      text.setAttribute("y", 30);
      text.setAttribute("text-anchor", "middle");
      text.textContent = node.label;

      g.appendChild(rect);
      g.appendChild(text);

      // Drag handling
      g.addEventListener("mousedown", startDrag);
      g.addEventListener("touchstart", startDrag);

      svg.appendChild(g);
    });
  }

  function startDrag(e) {
    e.preventDefault();
    const g = e.currentTarget;
    const point = svg.createSVGPoint();
    point.x = e.clientX || e.touches[0].clientX;
    point.y = e.clientY || e.touches[0].clientY;
    const svgP = g.closest("svg").getScreenCTM().inverse().transform(point);

    const move = (me) => {
      const pt = svg.createSVGPoint();
      pt.x = me.clientX || me.touches[0].clientX;
      pt.y = me.clientY || me.touches[0].clientY;
      const transformed = g.closest("svg").getScreenCTM().inverse().transform(pt);
      const dx = transformed.x - svgP.x;
      const dy = transformed.y - svgP.y;
      g.setAttribute("transform", `translate(${parseFloat(g.getAttribute("transform").match(/[\d.-]+/)[0]) + dx}, ${parseFloat(g.getAttribute("transform").match(/[\d.-]+/)[1]) + dy})`);
    };

    const end = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", end);
      document.removeEventListener("touchmove", move);
      document.removeEventListener("touchend", end);
    };

    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", end);
    document.addEventListener("touchmove", move, { passive: false });
    document.addEventListener("touchend", end);
  }

  // ---- 3D Viewport (three.js) ----
  let threeScene = null;
  let threeCamera = null;
  let threeRenderer = null;
  let threeMesh = null;
  let threeAnimating = false;

  async function init3D() {
    const canvas = $("#canvas-3d");
    if (!canvas) return;

    try {
      // Dynamically load three.js
      if (typeof THREE === "undefined") {
        await loadScript("https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js");
      }

      const width = canvas.width || 300;
      const height = canvas.height || 300;

      threeRenderer = new THREE.WebGLRenderer({ canvas, alpha: true });
      threeRenderer.setSize(width, height);

      threeScene = new THREE.Scene();
      threeCamera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
      threeCamera.position.z = 5;

      // Tetrahedron
      const geometry = new THREE.TetrahedronGeometry(2, 0);
      const material = new THREE.MeshStandardMaterial({
        color: 0x2684ff,
        wireframe: true,
        transparent: true,
        opacity: 0.8,
      });
      threeMesh = new THREE.Mesh(geometry, material);
      threeScene.add(threeMesh);

      const wireMaterial = new THREE.LineBasicMaterial({
        color: 0x2684ff,
        opacity: 0.3,
      });
      const wireframe = new THREE.WireframeGeometry(geometry);
      const wireframeLine = new THREE.LineSegments(wireframe);
      threeScene.add(wireframeLine);

      // Lights
      const ambientLight = new THREE.AmbientLight(0x404040);
      threeScene.add(ambientLight);
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
      dirLight.position.set(1, 1, 1);
      threeScene.add(dirLight);

      animate3D();
    } catch (err) {
      console.warn("3D init failed:", err);
      canvas.parentElement.innerHTML = '<div style="padding:20px;text-align:center;color:var(--sw-color-text-muted);">3D viewport unavailable</div>';
    }
  }

  function animate3D() {
    if (!threeRenderer || !threeScene || !threeCamera) return;
    requestAnimationFrame(animate3D);
    if (threeMesh && threeAnimating) {
      threeMesh.rotation.x += 0.01;
      threeMesh.rotation.y += 0.01;
    }
    threeRenderer.render(threeScene, threeCamera);
  }

  function rotate3D() { threeAnimating = true; }
  function zoom3D() { threeCamera.position.z = threeCamera.position.z === 3 ? 5 : 3; }
  function reset3D() {
    threeAnimating = false;
    threeCamera.position.z = 5;
    if (threeMesh) {
      threeMesh.rotation.set(0, 0, 0);
    }
  }

  function loadScript(url) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = url;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  // ---- Toast notifications ----
  function showToast(msg, type = "success", duration = 3000) {
    const toast = document.createElement("div");
    toast.className = `sw-toast sw-toast--${type}`;
    toast.textContent = msg;
    document.body.appendChild(toast);

    requestAnimationFrame(() => {
      toast.classList.add("sw-toast--show");
    });

    setTimeout(() => {
      toast.classList.remove("sw-toast--show");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // ---- Event listeners ----
  function bindEvents() {
    // Chat
    $("#btn-send")?.addEventListener("click", sendChatMessage);
    $("#message-input")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
    $("#message-input")?.addEventListener("input", (e) => resizeInput(e.target));

    // Sidebar toggle
    $("#btn-toggle-left")?.addEventListener("click", () => toggleSidebar("left"));

    // Terminal tabs
    $$(".sw-terminal-tab").forEach((tab) => {
      tab.addEventListener("click", () => switchTerminal(tab.dataset.terminal));
    });

    // Tabs
    $$(".sw-tab").forEach((tab) => {
      tab.addEventListener("click", function () {
        const tabName = this.dataset.tab;
        $$(".sw-tab").forEach((t) => t.classList.remove("sw-tab--active"));
        this.classList.add("sw-tab--active");
        $$(".sw-tab-pane").forEach((p) => p.classList.remove("sw-tab-pane--active"));
        $(`#tab-${tabName}`).classList.add("sw-tab-pane--active");
      });
    });

    // 3D controls
    $("#btn-3d-rotate")?.addEventListener("click", rotate3D);
    $("#btn-3d-zoom")?.addEventListener("click", zoom3D);
    $("#btn-3d-reset")?.addEventListener("click", reset3D);

    // Voice
    $("#btn-voice")?.addEventListener("click", toggleVoice);

    // Canvas overlay
    $("#btn-canvas")?.addEventListener("click", () => toggleCanvas(true));
    $("#btn-canvas-close")?.addEventListener("click", () => toggleCanvas(false));

    // Nav buttons
    $("#btn-home")?.addEventListener("click", () => showToast("Home clicked"));
    $("#btn-agents")?.addEventListener("click", () => toggleSidebar("left"));
    $("#btn-terminal")?.addEventListener("click", () => switchTerminal("console"));
    $("#btn-settings")?.addEventListener("click", () => showToast("Settings opened"));

    // Request animation frame-based input resize fix
    $("#message-input")?.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = Math.max(40, this.scrollHeight) + "px";
    });
  }

  // ---- Initialize ----
  function init() {
    bindEvents();
    initWebSocket();
    initVoice();
    initGestures();
    init3D();

    // Populate file tree with known project files
    renderFileTree([
      "README.md",
      "pyproject.toml",
      "requirements.txt",
      "configs/",
      "sw_platform/",
      "foundation/",
      "intelligence/",
      "serving/",
      "tests/",
      "datasets/",
    ]);

    showToast("Silverwing workspace loaded", "success");
  }

  // Wait for DOM
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
