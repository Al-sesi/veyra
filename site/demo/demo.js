/* Veyra live demo — browser client for the FastAPI pipeline.
 *
 * Flow: pick a language -> record (mic -> WAV) / sample clip / file upload ->
 * POST multipart to {API_BASE}/voice-note -> render transcript + reply + saved
 * entries as chat bubbles -> refresh "your book" from the GET endpoints.
 * The user_id returned on the first note is kept in localStorage so the book
 * persists per browser.
 */

const LOCAL_FALLBACK = "http://127.0.0.1:8000";
const qs = new URLSearchParams(window.location.search);
const API_BASE = (function () {
  if (qs.has("api")) return qs.get("api").replace(/\/+$/, "");
  if (window.location.protocol === "file:") return LOCAL_FALLBACK;
  const probe = window.location.origin;
  if (/127\.0\.0\.1|localhost/.test(probe) && window.location.port) {
    return probe;
  }
  return LOCAL_FALLBACK;
})();
const UID_KEY = "veyra_demo_uid";

const LANG_NAMES = { yo: "Yorùbá", ha: "Hausa", ig: "Igbo", en: "English", pcm: "Pidgin" };

const chatLog = document.getElementById("chatLog");
const apiStatus = document.getElementById("apiStatus");
const micBtn = document.getElementById("micBtn");
const recordHint = document.getElementById("recordHint");
const fileInput = document.getElementById("fileInput");
const pendingBar = document.getElementById("pendingBar");
const pendingName = document.getElementById("pendingName");
const pendingPlayer = document.getElementById("pendingPlayer");
const sendBtn = document.getElementById("sendBtn");
const discardBtn = document.getElementById("discardBtn");
const resetBookBtn = document.getElementById("resetBookBtn");
const bookOwner = document.getElementById("bookOwner");
const bookEmpty = document.getElementById("bookEmpty");
const bookContent = document.getElementById("bookContent");
const bookTotals = document.getElementById("bookTotals");
const bookDebts = document.getElementById("bookDebts");
const bookEntries = document.getElementById("bookEntries");

let currentLang = "yo";
let pendingBlob = null;
let pendingFilename = "";
let recorder = null; // {stream, ctx, source, processor, chunks}
let objectUrl = null;

// ---------------------------------------------------------------------------
// Chat rendering
// ---------------------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtNaira(amount) {
  const sign = amount < 0 ? "−" : "";
  return `${sign}₦${Math.abs(Math.round(amount)).toLocaleString("en-NG")}`;
}

function fmtTime(iso) {
  try {
    return new Date(iso.replace(" ", "T") + "Z").toLocaleString("en-NG", {
      day: "numeric", month: "short", hour: "numeric", minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function scrollChat() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addBubble(kind, build) {
  const bubble = el("div", `bubble bubble-${kind}`);
  const name = kind === "user" ? "You" : "Veyra";
  bubble.appendChild(el("p", "bubble-name", name));
  build(bubble);
  chatLog.appendChild(bubble);
  scrollChat();
  return bubble;
}

function addText(bubble, text) {
  bubble.appendChild(el("p", null, text));
}

function addMeta(bubble, text) {
  bubble.appendChild(el("span", "bubble-meta", text));
}

function addEntryChips(bubble, entries) {
  if (!entries || entries.length === 0) return;
  const ul = el("ul", "bubble-entries");
  for (const e of entries) {
    const li = el("li");
    const qty = e.quantity ? `${e.quantity} × ` : "";
    li.appendChild(el("span", null, `${qty}${e.item} (${e.type})`));
    li.appendChild(el("strong", null, fmtNaira(e.amount)));
    ul.appendChild(li);
  }
  bubble.appendChild(ul);
}

function addErrorBubble(text) {
  const bubble = el("div", "bubble bubble-error");
  bubble.appendChild(el("p", null, text));
  chatLog.appendChild(bubble);
  scrollChat();
}

function showTyping() {
  const bubble = el("div", "bubble bubble-veyra");
  bubble.id = "typingBubble";
  bubble.appendChild(el("p", "bubble-name", "Veyra"));
  const t = el("span", "typing");
  t.append(el("i"), el("i"), el("i"));
  bubble.appendChild(t);
  chatLog.appendChild(bubble);
  scrollChat();
}

function hideTyping() {
  document.getElementById("typingBubble")?.remove();
}

// ---------------------------------------------------------------------------
// Pending audio (recorded / sampled / uploaded)
// ---------------------------------------------------------------------------

function setPending(blob, filename) {
  clearPending();
  pendingBlob = blob;
  pendingFilename = filename;
  pendingName.textContent = filename;
  objectUrl = URL.createObjectURL(blob);
  pendingPlayer.src = objectUrl;
  pendingPlayer.hidden = false;
  pendingBar.hidden = false;
}

function clearPending() {
  pendingBlob = null;
  pendingFilename = "";
  pendingBar.hidden = true;
  pendingPlayer.hidden = true;
  pendingPlayer.removeAttribute("src");
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
}

// ---------------------------------------------------------------------------
// Mic recording -> 16-bit PCM WAV (ffmpeg-safe)
// ---------------------------------------------------------------------------

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);
  const writeStr = (offset, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true); // bit depth
  writeStr(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (e) => {
    chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(ctx.destination);
  recorder = { stream, ctx, source, processor, chunks };
}

async function stopRecording() {
  if (!recorder) return null;
  const { stream, ctx, source, processor, chunks } = recorder;
  recorder = null;
  processor.onaudioprocess = null;
  source.disconnect();
  processor.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  await ctx.close();
  const total = chunks.reduce((n, c) => n + c.length, 0);
  if (total === 0) return null;
  const merged = new Float32Array(total);
  let at = 0;
  for (const c of chunks) {
    merged.set(c, at);
    at += c.length;
  }
  return encodeWav(merged, ctx.sampleRate);
}

micBtn.addEventListener("click", async () => {
  if (recorder) {
    micBtn.classList.remove("is-recording");
    recordHint.classList.remove("is-recording");
    recordHint.textContent = "Tap the mic and speak";
    micBtn.disabled = true;
    try {
      const blob = await stopRecording();
      if (blob) {
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        setPending(blob, `voice-note-${stamp}.wav`);
      }
    } finally {
      micBtn.disabled = false;
    }
    return;
  }
  try {
    await startRecording();
    micBtn.classList.add("is-recording");
    recordHint.classList.add("is-recording");
    recordHint.textContent = "Recording… tap again to stop";
  } catch {
    addErrorBubble("Microphone unavailable. Check the browser permission, or use a sample / upload instead.");
  }
});

// ---------------------------------------------------------------------------
// Language pills
// ---------------------------------------------------------------------------

document.querySelectorAll(".lang-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".lang-pill").forEach((p) => {
      p.classList.remove("is-active");
      p.setAttribute("aria-checked", "false");
    });
    pill.classList.add("is-active");
    pill.setAttribute("aria-checked", "true");
    currentLang = pill.dataset.lang;
  });
});

// ---------------------------------------------------------------------------
// Samples + file upload
// ---------------------------------------------------------------------------

document.querySelectorAll(".sample-btn[data-sample]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      const res = await fetch(btn.dataset.sample);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      setPending(blob, btn.dataset.name);
    } catch {
      addErrorBubble("Could not load the sample clip. Is the site being served from site/ (so samples/ exists)?");
    } finally {
      btn.disabled = false;
    }
  });
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  setPending(file, file.name);
  fileInput.value = "";
});

// ---------------------------------------------------------------------------
// Send to the pipeline
// ---------------------------------------------------------------------------

async function sendPending() {
  if (!pendingBlob) return;
  const blob = pendingBlob;
  const filename = pendingFilename;
  const lang = currentLang;
  clearPending();

  addBubble("user", (b) => {
    addText(b, `Voice note (${LANG_NAMES[lang]}) — ${filename}`);
    addMeta(b, "Sending…");
  });

  showTyping();
  sendBtn.disabled = true;

  try {
    const form = new FormData();
    form.append("file", blob, filename);
    form.append("language", lang);
    const uid = localStorage.getItem(UID_KEY);
    if (uid) form.append("user_id", uid);

    const res = await fetch(`${API_BASE}/voice-note`, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    localStorage.setItem(UID_KEY, String(data.user_id));

    hideTyping();
    addBubble("user", (b) => {
      addText(b, data.transcript || "(no transcript)");
      addMeta(b, `${LANG_NAMES[lang]} · ${filename}`);
    });
    addBubble("veyra", (b) => {
      addText(b, data.reply_text || "Done.");
      addEntryChips(b, data.entries);
      if (data.language_notice) addMeta(b, data.language_notice);
    });

    await refreshBook();
  } catch (err) {
    hideTyping();
    addErrorBubble(
      err.message.includes("Failed to fetch")
        ? `Cannot reach Veyra at ${API_BASE}. Start it with: python -m uvicorn app.main:app --port 8000`
        : `Something went wrong: ${err.message}`
    );
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendPending);
discardBtn.addEventListener("click", clearPending);

// ---------------------------------------------------------------------------
// Your book
// ---------------------------------------------------------------------------

function renderTotals(summary) {
  bookTotals.textContent = "";
  const chips = [
    { label: "Sales", value: summary.total_sales, cls: "" },
    { label: "Spent", value: summary.total_expenses, cls: "" },
    {
      label: "Profit",
      value: summary.profit,
      cls: summary.profit >= 0 ? "t-profit" : "t-loss",
    },
    { label: "Owed to you", value: summary.total_owed_to_me, cls: "t-profit" },
    { label: "You owe", value: summary.total_i_owe, cls: "t-loss" },
    { label: "Top sale", value: summary.top_sale_item?.total_amount ?? 0, cls: "" },
  ];
  for (const c of chips) {
    const chip = el("div", `total-chip ${c.cls}`);
    chip.appendChild(el("span", "t-label", c.label));
    chip.appendChild(el("span", "t-value", fmtNaira(c.value)));
    bookTotals.appendChild(chip);
  }
}

function renderDebts(debts) {
  bookDebts.textContent = "";
  if (!debts.length) {
    bookDebts.appendChild(el("li", "book-note", "No open debts. Nice and clean."));
    return;
  }
  for (const d of debts) {
    const li = el("li");
    const main = el("span", "b-main");
    main.appendChild(el("span", "b-desc", d.person));
    main.appendChild(el("span", "b-time", d.direction === "owed_to_me" ? "owes you" : "you owe"));
    li.appendChild(main);
    li.appendChild(el("strong", "b-amt " + (d.direction === "owed_to_me" ? "is-income" : "is-expense"), fmtNaira(d.amount)));
    bookDebts.appendChild(li);
  }
}

function renderEntries(entries) {
  bookEntries.textContent = "";
  if (!entries.length) {
    bookEntries.appendChild(el("li", "book-note", "No entries yet."));
    return;
  }
  for (const e of entries.slice(0, 20)) {
    const li = el("li");
    const main = el("span", "b-main");
    const qty = e.quantity ? `${e.quantity} × ` : "";
    main.appendChild(el("span", "b-desc", `${qty}${e.item}${e.status === "voided" ? " (voided)" : ""}`));
    main.appendChild(el("span", "b-time", fmtTime(e.created_at)));
    li.appendChild(main);
    li.appendChild(el("strong", `b-amt is-${e.type}`, fmtNaira(e.amount)));
    bookEntries.appendChild(li);
  }
}

async function refreshBook() {
  const uid = localStorage.getItem(UID_KEY);
  if (!uid) {
    bookOwner.textContent = "";
    bookEmpty.hidden = false;
    bookContent.hidden = true;
    return;
  }
  bookOwner.textContent = `Book #${uid} — kept in this browser`;
  try {
    const [ledgerRes, summaryRes, debtsRes] = await Promise.all([
      fetch(`${API_BASE}/ledger/${uid}?limit=20`),
      fetch(`${API_BASE}/summary/${uid}?days=7`),
      fetch(`${API_BASE}/debts/${uid}`),
    ]);
    if (!ledgerRes.ok || !summaryRes.ok || !debtsRes.ok) throw new Error("book fetch failed");
    const ledger = await ledgerRes.json();
    const summary = await summaryRes.json();
    const debts = await debtsRes.json();
    renderTotals(summary);
    renderDebts(debts.debts);
    renderEntries(ledger.entries);
    bookEmpty.hidden = true;
    bookContent.hidden = false;
  } catch {
    bookOwner.textContent = `Book #${uid} — could not load details (API offline?)`;
  }
}

resetBookBtn.addEventListener("click", () => {
  localStorage.removeItem(UID_KEY);
  clearPending();
  refreshBook();
  addBubble("veyra", (b) => addText(b, "Fresh book started. Your next voice note opens a new one."));
});

// ---------------------------------------------------------------------------
// Connection check + initial paint
// ---------------------------------------------------------------------------

(async function init() {
  try {
    const res = await fetch(`${API_BASE}/ledger/1?limit=1`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    apiStatus.textContent = "Connected — speak in your language";
    apiStatus.classList.add("is-ok");
  } catch {
    apiStatus.textContent = `API offline (${API_BASE}) — start: python -m uvicorn app.main:app --port 8000`;
    apiStatus.classList.add("is-bad");
  }
  await refreshBook();
})();
