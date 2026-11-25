// moshi-rt web client: microphone -> websocket -> Moshi -> speaker, with live per-frame timings.
// Protocol: see app/server.py. Audio is 24 kHz mono int16 in both directions.
// ?server=https://host  model server (default: this origin)   ?token=...   ?demo=1  simulated call, no server
(() => {
  "use strict";
  const SR = 24000, FRAME = 1920, BUDGET = 80, CHART_S = 6;
  const q = new URLSearchParams(location.search);
  const DEMO = q.has("demo");
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- state ----------
  let state = "idle";
  function setState(s, status) {
    state = s; document.body.dataset.state = s;
    if (status) $("status").textContent = status;
    $("liveChip").textContent = s === "live" ? "live" : s === "connecting" ? "connecting" : "idle";
    $("callLabel").textContent = s === "connecting" ? "Connecting…" : "Start conversation";
  }
  function toast(msg, ms = 4000) {
    const t = $("toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), ms);
  }

  // ---------- theme ----------
  $("themeBtn").onclick = () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("moshi-rt-theme", t); } catch (e) { /* private mode */ }
  };

  // ---------- audio ----------
  const WORKLETS = `
  class Capture extends AudioWorkletProcessor {
    process(inputs) { const c = inputs[0] && inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
  }
  class Player extends AudioWorkletProcessor {
    constructor() {
      super();
      this.buf = new Float32Array(${SR} * 8); this.r = 0; this.w = 0; this.n = 0; this.playing = false;
      this.prebuffer = ${FRAME * 2};
      this.port.onmessage = (e) => {
        const x = e.data, L = this.buf.length;
        for (let i = 0; i < x.length; i++) { this.buf[this.w] = x[i]; this.w = (this.w + 1) % L; }
        this.n = Math.min(this.n + x.length, L);
      };
    }
    process(_, outputs) {
      const out = outputs[0][0], L = this.buf.length;
      if (!this.playing && this.n >= this.prebuffer) this.playing = true;
      for (let i = 0; i < out.length; i++) {
        if (this.playing && this.n > 0) { out[i] = this.buf[this.r]; this.r = (this.r + 1) % L; this.n--; }
        else { out[i] = 0; if (this.playing) { this.playing = false; this.prebuffer = ${FRAME}; } }
      }
      return true;
    }
  }
  registerProcessor("capture", Capture);
  registerProcessor("player", Player);`;

  let ctx, ws, mic, cam, capNode, playNode, outAn, inAn, pingTimer, demoTimer, t0;
  let muted = false, pending = [], pendingLen = 0;
  let frames = [], lateTotal = 0, frameTotal = 0, rtts = [], captionText = "";
  let callLog = null;  // every frame and ping of the current call, for "Save call log"

  function serverURL() {
    let base = (q.get("server") || location.origin).replace(/^http/, "ws").replace(/\/$/, "");
    const tok = q.get("token");
    return `${base}/api/chat${tok ? `?token=${encodeURIComponent(tok)}` : ""}`;
  }

  function resetStats() {
    frames = []; lateTotal = 0; frameTotal = 0; rtts = []; captionText = ""; $("captions").textContent = "";
    callLog = null;
  }

  async function start() {
    if (state === "connecting" || state === "live") return;
    resetStats();
    setState("connecting", "Connecting to the model server…");
    try {
      ctx = new AudioContext({ sampleRate: SR, latencyHint: "interactive" });
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLETS], { type: "text/javascript" })));
      playNode = new AudioWorkletNode(ctx, "player", { outputChannelCount: [1] });
      outAn = ctx.createAnalyser(); outAn.fftSize = 256; outAn.smoothingTimeConstant = 0.75;
      playNode.connect(outAn).connect(ctx.destination);
      if (!DEMO) {
        mic = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
        const src = ctx.createMediaStreamSource(mic);
        capNode = new AudioWorkletNode(ctx, "capture");
        inAn = ctx.createAnalyser(); inAn.fftSize = 256;
        const sink = ctx.createGain(); sink.gain.value = 0;
        src.connect(inAn); src.connect(capNode); capNode.connect(sink).connect(ctx.destination);
        capNode.port.onmessage = (e) => onMic(e.data);
      }
    } catch (e) {
      cleanup();
      setState("error", "Microphone unavailable");
      toast("Allow microphone access in your browser to talk to Moshi.");
      return;
    }
    if (DEMO) { startDemo(); return; }
    ws = new WebSocket(serverURL());
    ws.binaryType = "arraybuffer";
    ws.onmessage = onMessage;
    ws.onerror = () => { if (state === "connecting") { cleanup(); setState("error", "Server unreachable");
      toast("Could not reach the model server. Is ./start.sh still running?"); } };
    ws.onclose = () => { if (state === "live") finish("Connection closed"); };
    // A Modal GPU that went idle has to cold-start (1-3 min); say so instead of a silent "Connecting…".
    const tc = performance.now();
    clearInterval(start.timer);
    start.timer = setInterval(() => {
      if (state !== "connecting") { clearInterval(start.timer); document.body.classList.remove("waking"); $("wake").hidden = true; return; }
      const s = Math.floor((performance.now() - tc) / 1000), clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
      if (s >= 3) {
        document.body.classList.add("waking"); $("wake").hidden = false;
        $("status").textContent = `Waking up the GPU… ${clock}`;
        $("wakeClock").textContent = clock;
      }
      if (s >= 420) { clearInterval(start.timer); document.body.classList.remove("waking"); $("wake").hidden = true; cleanup(); setState("error", "Server not answering");
        toast("The model server did not answer in 7 minutes. Check the terminal running ./start.sh."); }
    }, 1000);
  }

  // Keep a Modal GPU awake while this page is open and visible; once the tab is closed or hidden,
  // the GPU shuts down after its 3 idle minutes, so a forgotten tab in the background costs nothing.
  const HEALTH = `${(q.get("server") || location.origin).replace(/\/$/, "")}/api/health`;
  function keepWarm() {
    if (DEMO || document.hidden || state === "live") return;
    fetch(HEALTH, { mode: "no-cors", cache: "no-store" }).catch(() => {});
  }
  keepWarm();
  setInterval(keepWarm, 60000);
  document.addEventListener("visibilitychange", keepWarm);

  function onMic(chunk) {
    if (!ws || ws.readyState !== 1 || state !== "live") return;
    pending.push(muted ? new Float32Array(chunk.length) : chunk); pendingLen += chunk.length;
    while (pendingLen >= FRAME) {
      const out = new Int16Array(FRAME); let k = 0;
      while (k < FRAME) {
        const c = pending[0], take = Math.min(c.length, FRAME - k);
        for (let i = 0; i < take; i++) out[k + i] = Math.max(-1, Math.min(1, c[i])) * 32767;
        k += take;
        if (take === c.length) pending.shift(); else pending[0] = c.subarray(take);
      }
      pendingLen -= FRAME;
      const msg = new Uint8Array(1 + out.byteLength); msg[0] = 1; msg.set(new Uint8Array(out.buffer), 1);
      ws.send(msg);
    }
  }

  function goLive(info) {
    t0 = performance.now();
    callLog = { started: new Date().toISOString(), server: info, frames: [], rtt_ms: [] };
    $("config").textContent = `${info.config} · ${info.gpu}`;
    setState("live", "Live");
    pingTimer = setInterval(ping, 1000);
  }

  function onFrame(f) {
    frames.push({ t: performance.now(), ...f });
    frameTotal++; if (f.ms > BUDGET) lateTotal++;
    if (callLog) callLog.frames.push([+(performance.now() - t0).toFixed(1), f.ms, f.enc, f.lm, f.dec]);
    while (frames.length && performance.now() - frames[0].t > 30000) frames.shift();
  }

  function addText(s) {
    captionText = (captionText + s).slice(-220);
    const words = captionText.replace(/^\S*\s/, "");
    const cut = Math.max(0, words.length - 110);
    const el = $("captions");
    el.innerHTML = "";
    if (cut) { const o = document.createElement("span"); o.className = "old"; o.textContent = words.slice(0, cut); el.append(o); }
    el.append(document.createTextNode(words.slice(cut)));
  }

  function onMessage(e) {
    if (typeof e.data === "string") {
      const m = JSON.parse(e.data);
      if (m.type === "ready") goLive(m);
      else if (m.type === "busy") { cleanup(); setState("error", "Server busy");
        toast("The model server is in another conversation. Try again in a moment."); }
      else if (m.type === "frame") onFrame({ ms: m.ms, enc: m.encode, lm: m.lm, dec: m.decode });
      return;
    }
    const b = new Uint8Array(e.data), kind = b[0];
    if (kind === 1) {
      const pcm = new Int16Array(e.data.slice(1)), f = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
      playNode.port.postMessage(f, [f.buffer]);
    } else if (kind === 2) addText(new TextDecoder().decode(b.subarray(1)));
    else if (kind === 5) {
      const r = performance.now() - new Float64Array(e.data.slice(1, 9))[0];
      rtts.push(r); if (rtts.length > 20) rtts.shift();
      if (callLog) callLog.rtt_ms.push(+r.toFixed(1));
    }
  }

  function ping() {
    if (!ws || ws.readyState !== 1) return;
    const m = new Uint8Array(9); m[0] = 5; m.set(new Uint8Array(new Float64Array([performance.now()]).buffer), 1);
    ws.send(m);
  }

  function cleanup() {
    clearInterval(pingTimer); clearInterval(demoTimer);
    if (ws && ws.readyState <= 1) { ws.onclose = null; ws.close(); }
    if (mic) mic.getTracks().forEach((t) => t.stop());
    if (cam) toggleCam();
    if (ctx) ctx.close();
    ws = ctx = mic = playNode = outAn = inAn = null; pending = []; pendingLen = 0;
    setPressed("muteBtn", false); muted = false;
  }

  function finish(reason) {
    const ms = callLog ? callLog.frames.map((f) => f[1]) : frames.map((f) => f.ms);  // whole call, not the chart window
    const s = t0 ? Math.floor((performance.now() - t0) / 1000) : 0;
    if (callLog) {
      const st = (i) => { const v = callLog.frames.map((f) => f[i]);
        const r = (x) => +x.toFixed(2);
        return v.length ? { p50: r(pct(v, 0.5)), p95: r(pct(v, 0.95)), p99: r(pct(v, 0.99)), max: r(Math.max(...v)) } : null; };
      Object.assign(callLog, {
        ended: new Date().toISOString(), duration_s: s, simulated: DEMO, page: location.origin,
        user_agent: navigator.userAgent, budget_ms: BUDGET, n_frames: frameTotal, late_frames: lateTotal,
        summary_ms: { frame: st(1), encode: st(2), lm: st(3), decode: st(4),
          rtt: callLog.rtt_ms.length ? { p50: pct(callLog.rtt_ms, 0.5), max: Math.max(...callLog.rtt_ms) } : null },
        columns: ["t_ms", "frame_ms", "encode_ms", "lm_ms", "decode_ms"],
      });
    }
    $("saveLog").hidden = !(callLog && frameTotal);
    $("sDur").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    $("sP50").textContent = ms.length ? `${pct(ms, 0.5).toFixed(1)} ms` : "–";
    $("sP99").textContent = ms.length ? `${pct(ms, 0.99).toFixed(1)} ms` : "–";
    $("sLate").textContent = frameTotal ? `${lateTotal} / ${frameTotal}` : "–";
    cleanup(); t0 = null;
    setState("ended", reason || "Call ended");
  }

  // ---------- controls ----------
  function setPressed(id, on) { $(id).setAttribute("aria-pressed", on ? "true" : "false"); }
  function toggleMute() {
    if (state !== "live") return;
    muted = !muted; setPressed("muteBtn", muted);
    $("muteBtn").querySelector(".label").textContent = muted ? "Unmute" : "Mute";
    toast(muted ? "Microphone muted. Moshi hears silence." : "Microphone on.", 1800);
  }
  async function toggleCam() {
    if (cam) { cam.getTracks().forEach((t) => t.stop()); cam = null; $("pip").classList.remove("on"); setPressed("camBtn", false); return; }
    if (state !== "live") return;
    try {
      cam = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 480 } });
      $("pip").srcObject = cam; $("pip").classList.add("on"); setPressed("camBtn", true);
    } catch (e) { toast("Camera unavailable. It is only shown to you and never sent."); }
  }
  $("startBtn").onclick = start;
  $("againBtn").onclick = start;
  $("cancelBtn").onclick = () => {
    clearInterval(start.timer); document.body.classList.remove("waking"); $("wake").hidden = true;
    cleanup(); setState("idle", "Ready");
  };
  $("saveLog").onclick = () => {  // per-frame log of the call as JSON, e.g. for results/app_call_<date>.json
    if (!callLog) return;
    const name = `app_call_${callLog.started.slice(0, 16).replace(/[-:]/g, "").replace("T", "_")}.json`;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(callLog)], { type: "application/json" }));
    a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    toast(`Saved ${name} to your downloads`);
  };
  $("endBtn").onclick = () => finish("Call ended");
  $("muteBtn").onclick = toggleMute;
  $("camBtn").onclick = toggleCam;
  addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("input, textarea")) return;
    if (e.key === "m" || e.key === "M") toggleMute();
    else if (e.key === "c" || e.key === "C") toggleCam();
    else if (e.key === "Escape" && state === "live") finish("Call ended");
    else if ((e.key === "Enter" || e.key === " ") && (state === "idle" || state === "ended" || state === "error")
             && document.activeElement === document.body) { e.preventDefault(); start(); }
  });

  // ---------- drawing ----------
  function pct(a, p) { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }
  function fit(c) { const r = devicePixelRatio || 1, w = Math.round(c.clientWidth * r), h = Math.round(c.clientHeight * r);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; } return [c.getContext("2d"), w, h, r]; }

  const spec = new Uint8Array(128), specIn = new Uint8Array(128), smooth = new Float32Array(72);
  let demoLevel = 0;
  function drawRing(now) {
    const [g, w, h, r] = fit($("ring"));
    g.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, R0 = Math.min(w, h) * 0.335, N = smooth.length;
    if (outAn) outAn.getByteFrequencyData(spec);
    if (inAn) inAn.getByteFrequencyData(specIn);
    let inLvl = 0; for (let i = 2; i < 40; i++) inLvl += specIn[i]; inLvl /= 38 * 255;
    const accent = css("--lm") || "#ff7a3d", blue = css("--blue") || "#2a78d6";
    for (let i = 0; i < N; i++) {
      const bin = 2 + Math.floor((i < N / 2 ? i : N - i) * 1.1);
      let v = outAn ? spec[bin] / 255 : 0;
      if (DEMO && state === "live") v = demoLevel * (0.55 + 0.45 * Math.sin(now / 140 + i * 0.7) ** 2);
      if (state === "idle" || state === "ended" || state === "error") v = 0.06 + 0.04 * Math.sin(now / 900 + i * 0.5);
      smooth[i] = smooth[i] * 0.7 + v * 0.3;
      const a = (i / N) * Math.PI * 2 - Math.PI / 2, len = 6 * r + smooth[i] * R0 * 0.42;
      g.strokeStyle = accent; g.globalAlpha = 0.35 + smooth[i] * 0.65; g.lineWidth = 3.2 * r; g.lineCap = "round";
      g.beginPath();
      g.moveTo(cx + Math.cos(a) * R0, cy + Math.sin(a) * R0);
      g.lineTo(cx + Math.cos(a) * (R0 + len), cy + Math.sin(a) * (R0 + len));
      g.stroke();
    }
    g.globalAlpha = Math.min(1, 0.12 + inLvl * 2.2);
    g.strokeStyle = blue; g.lineWidth = 2 * r;
    g.beginPath(); g.arc(cx, cy, R0 - 10 * r, 0, Math.PI * 2); g.stroke();
    g.globalAlpha = 1;
  }

  function drawChart(now) {
    const [g, w, h, r] = fit($("chart"));
    g.clearRect(0, 0, w, h);
    const padL = 28 * r, padB = 4 * r, ymax = 100, plotW = w - padL, plotH = h - padB - 6 * r;
    const y = (v) => h - padB - (Math.min(v, ymax) / ymax) * plotH;
    const ink3 = css("--ink-3"), line = css("--line");
    g.font = `${10 * r}px -apple-system, Inter, sans-serif`; g.fillStyle = ink3; g.textBaseline = "middle";
    for (const v of [0, 40, 80]) {
      g.fillText(String(v), 2 * r, y(v));
      g.strokeStyle = line; g.lineWidth = r; g.beginPath(); g.moveTo(padL, y(v)); g.lineTo(w, y(v)); g.stroke();
    }
    const pts = frames.filter((f) => now - f.t < CHART_S * 1000);
    const bw = Math.max(1.2 * r, plotW / (CHART_S * 12.5) - 1.2 * r);
    const cols = [css("--enc"), css("--lm"), css("--dec")];
    for (const f of pts) {
      const x = padL + plotW - ((now - f.t) / (CHART_S * 1000)) * plotW;
      let base = 0;
      [f.enc, f.lm, f.dec].forEach((v, i) => {
        g.fillStyle = cols[i]; g.globalAlpha = f.ms > BUDGET ? 1 : 0.9;
        g.fillRect(x, y(base + v), bw, y(base) - y(base + v)); base += v;
      });
      if (f.ms > BUDGET) { g.fillStyle = css("--danger"); g.fillRect(x, y(Math.min(f.ms, ymax)) - 3 * r, bw, 2 * r); }
    }
    g.globalAlpha = 1;
    g.setLineDash([5 * r, 4 * r]); g.strokeStyle = ink3; g.lineWidth = 1.4 * r;
    g.beginPath(); g.moveTo(padL, y(BUDGET)); g.lineTo(w, y(BUDGET)); g.stroke(); g.setLineDash([]);
  }

  function updateStats() {
    const ms = frames.map((f) => f.ms), p50 = pct(ms, 0.5), p99 = pct(ms, 0.99);
    $("p50").textContent = isNaN(p50) ? "–" : p50.toFixed(1);
    const frac = isNaN(p50) ? 0 : Math.min(1, p50 / BUDGET);
    const gv = $("gValue");
    gv.setAttribute("stroke-dasharray", `${(226 * frac).toFixed(1)} 302`);
    gv.style.opacity = frac > 0 ? "1" : "0";
    gv.style.stroke = frac < 0.75 ? css("--green") : frac < 0.95 ? "#f5a623" : css("--danger");
    $("p99").textContent = isNaN(p99) ? "–" : `${p99.toFixed(1)} ms`;
    $("p99").className = isNaN(p99) ? "" : p99 > BUDGET ? "bad" : "good";
    $("late").textContent = frameTotal ? `${lateTotal} / ${frameTotal}` : "–";
    $("late").className = frameTotal ? (lateTotal ? "bad" : "good") : "";
    $("rtt").textContent = rtts.length ? `${pct(rtts, 0.5).toFixed(0)} ms` : "–";
    if (t0) { const s = Math.floor((performance.now() - t0) / 1000);
      $("clock").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }
    else if (state !== "ended") $("clock").textContent = "–";
  }

  function loop() { const now = performance.now(); drawRing(now); drawChart(now); requestAnimationFrame(loop); }
  setInterval(updateStats, 250);
  requestAnimationFrame(loop);

  // ---------- demo mode (no GPU): simulated frames and captions ----------
  const DEMO_LINES = ["Hey, how's it going? ", "Oh, that sounds like a long day. ", "What are you working on right now? ",
    "Real-time speech is tricky, every frame has eighty milliseconds. ", "I'm listening, go ahead. "];
  function startDemo() {
    setTimeout(() => {
      goLive({ config: "demo mode (simulated)", gpu: "no GPU" });
      let n = 0, line = 0, ch = 0;
      demoTimer = setInterval(() => {
        n++;
        const enc = 2.4 + Math.random() * 0.5, lm = 41.4 + Math.random() * 0.6, dec = 2 + Math.random() * 0.2;
        onFrame({ ms: enc + lm + dec + 0.4, enc, lm, dec });
        if (n % 6 === 0) rtts.push(300 + Math.random() * 40), rtts.length > 20 && rtts.shift();
        const s = DEMO_LINES[line % DEMO_LINES.length];
        demoLevel = ch < s.length ? 0.5 + 0.5 * Math.random() : 0.05;
        if (ch < s.length) { addText(s.slice(ch, ch + 1)); ch++; } else if (n % 25 === 0) { line++; ch = 0; }
      }, 80);
    }, 900);
  }

  if (q.get("theme") === "light" || q.get("theme") === "dark") document.documentElement.dataset.theme = q.get("theme");
  if (q.has("still")) document.documentElement.classList.add("still");  // for screenshots
  setState("idle", DEMO ? "Demo mode" : "Ready");
  if (DEMO && q.has("autostart")) start();
  if (DEMO && q.has("ended")) setTimeout(() => finish("Call ended"), 6000);
})();
