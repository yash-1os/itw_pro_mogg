const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const emptyState = document.getElementById("emptyState");
const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");
const feedback = document.getElementById("feedback");
const uploadActions = document.getElementById("uploadActions");
const cameraActions = document.getElementById("cameraActions");
const cameraVideo = document.getElementById("cameraVideo");
const preview = document.getElementById("preview");
const startCamBtn = document.getElementById("captureButton");
const stopCamBtn = document.getElementById("stopCameraButton");
const matlabPlot = document.getElementById("matlabPlot");
const navItems = document.querySelectorAll(".sidebar .nav-item");

let selectedFile = null;
let currentMode = "upload";
let telemetryInterval = null;

const emotionConfig = {
  angry: { emoji: "😠" }, disgust: { emoji: "🤢" }, fear: { emoji: "😨" },
  happy: { emoji: "😄" }, sad: { emoji: "😢" }, surprise: { emoji: "😮" },
  neutral: { emoji: "😐" }
};

function setFeedback(message, isError = false) {
  feedback.textContent = message;
  feedback.classList.toggle("error", isError);
}

/* ---------- Face-Mogging: leaderboard + mode toggle (elements are created here, no HTML change needed) ---------- */
function injectMogStyles() {
  if (document.getElementById("mogStyles")) return;
  const style = document.createElement("style");
  style.id = "mogStyles";
  style.textContent = `
    #mogLeaderboard { margin: 0 0 14px; padding: 12px 14px; border-radius: 12px; background: var(--bg-subtle); border: 1px solid var(--border-light); }
    #mogLeaderboard .mog-title { font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; }
    #mogLeaderboard .mog-empty { font-size: 13px; color: var(--text-secondary); }
    .mog-row { display: grid; grid-template-columns: 34px 1fr auto; gap: 4px 10px; align-items: center; padding: 6px 0; }
    .mog-row + .mog-row { border-top: 1px solid var(--border-light); }
    .mog-rank { font-weight: 800; font-size: 15px; color: var(--text-muted); }
    .mog-row.first .mog-rank, .mog-row.first .mog-score { color: #b45309; }
    [data-theme="dark"] .mog-row.first .mog-rank, [data-theme="dark"] .mog-row.first .mog-score { color: #ffd700; }
    .mog-emo { font-size: 13px; color: var(--text-primary); text-transform: capitalize; }
    .mog-score { font-weight: 700; font-size: 15px; color: var(--text-primary); font-variant-numeric: tabular-nums; }
    .mog-bar { grid-column: 2 / 4; height: 6px; border-radius: 3px; background: var(--border-light); overflow: hidden; }
    .mog-bar > div { height: 100%; background: var(--accent-primary); }
    .mog-row.first .mog-bar > div { background: #f59e0b; }
    .mog-toggle { display: inline-flex; align-items: center; gap: 8px; font-size: 13.5px; font-weight: 500; color: var(--text-primary); cursor: pointer; user-select: none; }
  `;
  document.head.appendChild(style);
}

function ensureLeaderboard() {
  let box = document.getElementById("mogLeaderboard");
  if (box) return box;
  const list = document.getElementById("emotionList");
  if (!list || !list.parentElement) return null;
  injectMogStyles();
  box = document.createElement("div");
  box.id = "mogLeaderboard";
  box.hidden = true;
  list.parentElement.insertBefore(box, list);
  return box;
}

// data = /stream_data JSON; pass null to hide the leaderboard.
function renderLeaderboard(data) {
  const box = ensureLeaderboard();
  if (!box) return;
  if (!data || !data.mogging_mode) {
    box.hidden = true;
    box.replaceChildren();
    return;
  }

  const title = document.createElement("div");
  title.className = "mog-title";
  title.textContent = "Mog leaderboard";
  const nodes = [title];

  const faces = Array.isArray(data.leaderboard) ? data.leaderboard : [];
  if (faces.length === 0) {
    const empty = document.createElement("div");
    empty.className = "mog-empty";
    empty.textContent = "Waiting for faces…";
    nodes.push(empty);
  }

  for (const face of faces) {
    const score = Math.max(0, Math.min(100, Number(face.score) || 0));
    const row = document.createElement("div");
    row.className = "mog-row" + (face.rank === 1 ? " first" : "");

    const rank = document.createElement("div");
    rank.className = "mog-rank";
    rank.textContent = `#${face.rank}`;

    const emo = document.createElement("div");
    emo.className = "mog-emo";
    emo.textContent = `${(emotionConfig[face.emotion] || { emoji: "•" }).emoji} ${face.emotion || "unknown"}`;

    const value = document.createElement("div");
    value.className = "mog-score";
    value.textContent = score.toFixed(1);

    const bar = document.createElement("div");
    bar.className = "mog-bar";
    const fill = document.createElement("div");
    fill.style.width = `${score}%`;
    bar.appendChild(fill);

    row.append(rank, emo, value, bar);
    nodes.push(row);
  }

  box.hidden = false;
  box.replaceChildren(...nodes);
}

async function ensureModeToggle() {
  if (document.getElementById("mogModeToggle") || !cameraActions) return;
  injectMogStyles();
  const label = document.createElement("label");
  label.className = "mog-toggle";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.id = "mogModeToggle";
  const text = document.createElement("span");
  text.textContent = "Face-Mogging Mode";
  label.append(input, text);
  document.querySelector(".media-footer").insertBefore(label, feedback);
  try {
    const response = await fetch("/mode");
    if (response.ok) input.checked = Boolean((await response.json()).mogging_mode);
  } catch (e) {
    console.error("Could not read mogging mode", e);
  }

  input.addEventListener("change", async () => {
    try {
      const response = await fetch("/set_mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mogging_mode: input.checked })
      });
      if (!response.ok) throw new Error("set_mode failed");
      if (currentMode === "upload" && selectedFile) analyzeButton.click();
    } catch (e) {
      input.checked = !input.checked;
      setFeedback("Could not change Face-Mogging mode.", true);
    }
  });
}

function setActiveNav(targetHash) {
  navItems.forEach(item => {
    item.classList.toggle("active", item.getAttribute("href") === targetHash);
  });
}

function stopCamera() {
  cameraVideo.src = "";
  cameraVideo.hidden = true;
  clearInterval(telemetryInterval);
  renderLeaderboard(null);
  startCamBtn.disabled = false;
  stopCamBtn.disabled = true;
}

function setMode(mode, navHash = null) {
  currentMode = mode;
  document.querySelectorAll(".mode-tab").forEach(tab => tab.classList.toggle("active", tab.dataset.mode === mode));
  document.getElementById("sourceTitle").textContent = mode === "upload" ? "Image upload" : "Live OpenCV Stream";

  setActiveNav(navHash || (mode === "camera" ? "#camera" : "#upload"));

  uploadActions.hidden = mode !== "upload";
  cameraActions.hidden = mode !== "camera";

  imageInput.value = "";
  selectedFile = null;
  imagePreview.hidden = true;
  matlabPlot.hidden = true;

  if (mode === "upload") {
    stopCamera();
    emptyState.hidden = false;
    renderPlaceholder();
    setFeedback("Choose an image to begin.");
  } else {
    emptyState.hidden = true;
    cameraVideo.hidden = false;
    setFeedback("Ready. Click 'Start Live Tracking' to initialize the stream.");
  }
}

document.querySelectorAll(".mode-tab").forEach(tab => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

navItems.forEach(item => {
  item.addEventListener("click", (e) => {
    e.preventDefault();
    const target = item.getAttribute("href");
    if (target === "#camera") {
      setMode("camera", "#camera");
    } else if (target === "#upload") {
      setMode("upload", "#upload");
    } else {
      setActiveNav("#home");
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    setFeedback("Please choose a JPG, PNG, or WEBP image.", true);
    return;
  }
  selectedFile = file;
  imagePreview.src = URL.createObjectURL(file);
  imagePreview.hidden = false;
  emptyState.hidden = true;
  analyzeButton.disabled = false;
  setFeedback("Image ready. Select Analyze emotions.");
});

clearButton.addEventListener("click", () => {
  selectedFile = null;
  imageInput.value = "";
  imagePreview.removeAttribute("src");
  imagePreview.hidden = true;
  emptyState.hidden = false;
  analyzeButton.disabled = true;
  renderLeaderboard(null);
  renderPlaceholder();
  setFeedback("Choose an image to begin.");
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  const formData = new FormData();
  formData.append("image", selectedFile);
  setFeedback("Analyzing image…");
  analyzeButton.disabled = true;
  try {
    const response = await fetch("/analyze", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");

    renderResults(data);
    if (data.image) imagePreview.src = data.image;
    renderLeaderboard(data);
    const nf = data.faces ?? 0, raw = data.faces_found_before_filter ?? nf;
    setFeedback(`Analysis complete. ${nf} face${nf === 1 ? "" : "s"} scored` + (raw > nf ? ` (${raw - nf} filtered out)` : "") + ".");
  } catch (error) {
    setFeedback(error.message, true);
  } finally {
    analyzeButton.disabled = false;
  }
});

startCamBtn.addEventListener("click", () => {
  setFeedback("Starting OpenCV backend stream... Please wait.");
  cameraVideo.src = "/video_feed";
  cameraVideo.hidden = false;
  matlabPlot.hidden = true;
  startCamBtn.disabled = true;
  stopCamBtn.disabled = false;

  telemetryInterval = setInterval(async () => {
    try {
      const response = await fetch("/stream_data");
      if (response.ok) {
        const data = await response.json();
        renderResults(data);
        renderLeaderboard(data);
        setFeedback("Live tracking active.");
      }
    } catch (e) {
      console.error("Telemetry error", e);
    }
  }, 350);
});

stopCamBtn.addEventListener("click", async () => {
  stopCamera();
  setFeedback("Session stopped. Generating summary timeline graph...");
  emptyState.hidden = true;
  stopCamBtn.disabled = true;

  try {
    const response = await fetch("/stop_stream", { method: "POST" });
    const data = await response.json();

    if (data.overall_dominant) {
      document.getElementById("dominantEmotion").textContent = data.overall_dominant;
      document.getElementById("dominantScore").textContent = "Overall Session Majority";
      document.getElementById("dominantFace").textContent = (emotionConfig[data.overall_dominant] || { emoji: "☺" }).emoji;
      setFeedback(`Analysis complete. Most frequent emotion: ${data.overall_dominant.toUpperCase()}.`);
    }

    if (data.plot_url) {
      matlabPlot.src = data.plot_url;
      matlabPlot.hidden = false;
    } else {
      emptyState.hidden = false;
      if (!data.overall_dominant) setFeedback("Session stopped.");
    }
  } catch (error) {
    setFeedback("Session stopped, but failed to generate summary graph.", true);
    emptyState.hidden = false;
  }
});

function renderResults(data) {
  const list = document.getElementById("emotionList");
  list.innerHTML = "";
  const emotions = data.emotions || {};
  const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);

  for (const [name, score] of sorted) {
    const config = emotionConfig[name] || { emoji: "•" };
    const row = document.createElement("div");
    row.className = "emotion-row";

    const icon = document.createElement("div");
    icon.className = "emotion-emoji";
    icon.textContent = config.emoji;

    const detail = document.createElement("div");
    const meta = document.createElement("div");
    meta.className = "emotion-meta";

    const label = document.createElement("span");
    label.className = "emotion-name";
    label.textContent = name;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${Math.max(0, Math.min(100, Number(score) || 0))}%`;

    track.appendChild(fill);
    meta.appendChild(label);
    detail.appendChild(meta);
    detail.appendChild(track);

    const value = document.createElement("div");
    value.className = "emotion-value";
    value.textContent = `${Number(score).toFixed(1)}%`;

    row.append(icon, detail, value);
    list.appendChild(row);
  }

  const dominant = data.dominant_emotion || sorted[0]?.[0] || "unknown";
  const dominantScore = emotions[dominant] ?? 0;
  document.getElementById("dominantEmotion").textContent = dominant;
  document.getElementById("dominantScore").textContent = `${Number(dominantScore).toFixed(1)}% model confidence`;
  document.getElementById("dominantFace").textContent = (emotionConfig[dominant] || { emoji: "☺" }).emoji;
}

function renderPlaceholder() {
  document.getElementById("emotionList").innerHTML =
    '<div class="result-placeholder"><span class="placeholder-icon">◌</span><p>Your emotion percentages will appear here after analysis.</p></div>';
  document.getElementById("dominantEmotion").textContent = "Waiting for result";
  document.getElementById("dominantScore").textContent = "Analyze an image to see the top score.";
  document.getElementById("dominantFace").textContent = "☺";
}

preview.addEventListener("dragover", e => { e.preventDefault(); preview.classList.add("dragging"); });
preview.addEventListener("dragleave", () => preview.classList.remove("dragging"));
preview.addEventListener("drop", e => {
  e.preventDefault();
  preview.classList.remove("dragging");
  if (currentMode !== "upload") return;
  const file = e.dataTransfer.files[0];
  if (!file) return;
  selectedFile = file;
  imageInput.files = e.dataTransfer.files;
  imagePreview.src = URL.createObjectURL(file);
  imagePreview.hidden = false;
  emptyState.hidden = true;
  analyzeButton.disabled = false;
  setFeedback("Image ready. Select Analyze emotions.");
});

/* ---------- Dark / light theme (palette lives in style.css under [data-theme="dark"]) ---------- */
(function initTheme() {
  const root = document.documentElement;
  const btn = document.getElementById("themeToggle");
  const icon = document.getElementById("themeIcon");
  const label = document.getElementById("themeLabel");

  function apply(theme) {
    root.setAttribute("data-theme", theme);
    if (icon) icon.textContent = theme === "dark" ? "\u2600" : "\u263E";
    if (label) label.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  }

  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* storage blocked: fine */ }
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  apply(saved === "dark" || saved === "light" ? saved : (prefersDark ? "dark" : "light"));

  if (btn) {
    btn.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      apply(next);
      try { localStorage.setItem("theme", next); } catch (e) { /* ignore */ }
    });
  }
})();

ensureModeToggle();
