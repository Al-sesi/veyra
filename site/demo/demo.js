/* Veyra live demo — browser client for the FastAPI pipeline.
 *
 * Flow: pick a language -> record (mic -> WAV) / sample clip / file upload ->
 * POST multipart to {API_BASE}/voice-note -> render transcript + reply + saved
 * entries as chat bubbles -> refresh "your book" from the GET endpoints.
 * The user_id returned on the first note is kept in localStorage so the book
 * persists per browser.
 */

const LOCAL_FALLBACK = "http://127.0.0.1:8000";
const AZURE_BACKEND = "https://ca-veyra-backend.livelysand-b99847dc.southafricanorth.azurecontainerapps.io";
const qs = new URLSearchParams(window.location.search);
const API_BASE = (function () {
  if (qs.has("api")) return qs.get("api").replace(/\/+$/, "");
  if (window.location.protocol === "file:") return LOCAL_FALLBACK;
  const probe = window.location.origin;
  if (probe.includes("vercel.app")) return AZURE_BACKEND;
  if (/127\.0\.0\.1|localhost/.test(probe) && window.location.port) {
    return probe;
  }
  return AZURE_BACKEND;
})();

// Debug: Log the API base being used
console.log("API_BASE:", API_BASE);
console.log("window.location.origin:", window.location.origin);
const UID_KEY = "veyra_demo_uid";

const LANG_NAMES = { yo: "Yorùbá", ha: "Hausa", ig: "Igbo", en: "English", pcm: "Pidgin" };

const chatLog = document.getElementById("chatLog");
const micBtn = document.getElementById("micBtn");
const fileInput = document.getElementById("fileInput");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const resetBookBtn = document.getElementById("resetBookBtn");
const bookOwner = document.getElementById("bookOwner");
const bookEmpty = document.getElementById("bookEmpty");
const bookContent = document.getElementById("bookContent");
const bookTotals = document.getElementById("bookTotals");
const bookDebts = document.getElementById("bookDebts");
const bookEntries = document.getElementById("bookEntries");
const bookCard = document.querySelector(".book-card");
const bookOverlay = document.getElementById("bookOverlay");
const downloadExcelBtn = document.getElementById("downloadExcelBtn");
const downloadCsvBtn = document.getElementById("downloadCsvBtn");
const languageSelect = document.getElementById("languageSelect");
const teachVocabBtn = document.getElementById("teachVocabBtn");
const vocabModal = document.getElementById("vocabModal");
const closeVocabModal = document.getElementById("closeVocabModal");
const vocabForm = document.getElementById("vocabForm");
const vocabList = document.getElementById("vocabList");

let currentLang = "en"; // Default to English
let recorder = null; // {stream, ctx, source, processor, chunks}

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
    const item = e.item || "Transaction";
    const amount = e.amount !== null && e.amount !== undefined ? e.amount : 0;
    
    // Map entry types to user-friendly labels
    let typeLabel = "";
    if (e.type === "sale") {
      typeLabel = "sale";
    } else if (e.type === "expense") {
      typeLabel = "expense";
    } else if (e.type === "debt_owed_to_me") {
      typeLabel = "owes you";
    } else if (e.type === "debt_i_owe") {
      typeLabel = "you owe";
    } else if (e.type === "debt_paid") {
      typeLabel = "debt paid";
    }
    
    // Only show type label if it's meaningful (not for sale which is the default)
    const displayText = typeLabel ? `${qty}${item} (${typeLabel})` : `${qty}${item}`;
    li.appendChild(el("span", null, displayText));
    li.appendChild(el("strong", null, fmtNaira(amount)));
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
  const stream = await navigator.mediaDevices.getUserMedia({ 
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      sampleRate: 44100
    }
  });
  const ctx = new AudioContext({ sampleRate: 44100 });
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
  // Resample to 16kHz for better compatibility
  const targetRate = 16000;
  const originalRate = 44100;
  const resampled = new Float32Array(Math.floor(merged.length * targetRate / originalRate));
  for (let i = 0; i < resampled.length; i++) {
    const srcIdx = Math.floor(i * originalRate / targetRate);
    resampled[i] = merged[srcIdx];
  }
  return encodeWav(resampled, targetRate);
}

micBtn.addEventListener("click", async () => {
  if (recorder) {
    micBtn.classList.remove("is-recording");
    micBtn.disabled = true;
    try {
      const blob = await stopRecording();
      if (blob) {
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        const filename = `voice-note-${stamp}.wav`;
        // Send immediately like WhatsApp
        await sendAudioMessage(blob, filename);
      }
    } finally {
      micBtn.disabled = false;
    }
    return;
  }
  try {
    await startRecording();
    micBtn.classList.add("is-recording");
  } catch {
    addErrorBubble("Microphone unavailable. Check the browser permission, or upload an audio file instead.");
  }
});

// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  // Send immediately like WhatsApp
  sendAudioMessage(file, file.name);
  fileInput.value = "";
});

// ---------------------------------------------------------------------------
// Text input - send on Enter (Shift+Enter for new line)
// ---------------------------------------------------------------------------

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendPending();
  }
});

sendBtn.addEventListener("click", () => {
  sendPending();
});

// ---------------------------------------------------------------------------
// Send to the pipeline
// ---------------------------------------------------------------------------

async function sendAudioMessage(blob, filename) {
  const lang = currentLang;

  addBubble("user", (b) => {
    addText(b, "Voice recording");
    addMeta(b, "Sending…");
  });

  showTyping();
  sendBtn.disabled = true;

  try {
    const form = new FormData();
    form.append("file", blob, filename);
    form.append("language", lang);
    const uid = localStorage.getItem(UID_KEY);
    if (uid && uid !== "null" && !isNaN(parseInt(uid))) form.append("user_id", uid);

    const res = await fetch(`${API_BASE}/voice-note`, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    localStorage.setItem(UID_KEY, String(data.user_id));

    hideTyping();
    const detectedLang = data.detected_language || lang;
    addBubble("user", (b) => {
      addText(b, data.transcript || "(no transcript)");
      addMeta(b, LANG_NAMES[detectedLang] || detectedLang);
    });
    addBubble("veyra", (b) => {
      addText(b, data.reply_text || "Done.");
      addEntryChips(b, data.entries);
      if (data.language_notice) addMeta(b, data.language_notice);
    });

    await refreshBook();
  } catch (err) {
    hideTyping();
    console.error("API Error:", err);
    console.error("API_BASE:", API_BASE);
    addErrorBubble(
      err.message.includes("Failed to fetch")
        ? `Cannot reach Veyra at ${API_BASE}. Please check your internet connection.`
        : `Something went wrong: ${err.message}`
    );
  } finally {
    sendBtn.disabled = false;
  }
}

async function sendPending() {
  const text = textInput.value.trim();
  if (!text) return;

  textInput.value = "";
  const lang = currentLang;

  addBubble("user", (b) => {
    addText(b, text);
    addMeta(b, "Text");
  });

  showTyping();
  sendBtn.disabled = true;

  try {
    const form = new FormData();
    form.append("transcript", text);
    // For text, don't send language - let backend auto-detect
    const uid = localStorage.getItem(UID_KEY);
    if (uid && uid !== "null" && !isNaN(parseInt(uid))) form.append("user_id", uid);

    const res = await fetch(`${API_BASE}/voice-note`, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

    localStorage.setItem(UID_KEY, String(data.user_id));

    hideTyping();
    const detectedLang = data.detected_language || "en";
    addBubble("user", (b) => {
      addText(b, data.transcript || text || "(no transcript)");
      addMeta(b, LANG_NAMES[detectedLang] || detectedLang);
    });
    addBubble("veyra", (b) => {
      addText(b, data.reply_text || "Done.");
      addEntryChips(b, data.entries);
      if (data.language_notice) addMeta(b, data.language_notice);
    });

    await refreshBook();
  } catch (err) {
    hideTyping();
    console.error("API Error:", err);
    console.error("API_BASE:", API_BASE);
    addErrorBubble(
      err.message.includes("Failed to fetch")
        ? `Cannot reach Veyra at ${API_BASE}. Please check your internet connection.`
        : `Something went wrong: ${err.message}`
    );
  } finally {
    sendBtn.disabled = false;
  }
}

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
    const person = d.person || "Unknown person";
    main.appendChild(el("span", "b-desc", person));
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
    const item = e.item || "Unknown item";
    main.appendChild(el("span", "b-desc", `${qty}${item}${e.status === "voided" ? " (voided)" : ""}`));
    main.appendChild(el("span", "b-time", fmtTime(e.created_at)));
    li.appendChild(main);
    li.appendChild(el("strong", `b-amt is-${e.type}`, fmtNaira(e.amount)));
    bookEntries.appendChild(li);
  }
}

async function refreshBook() {
  const uid = localStorage.getItem(UID_KEY);
  if (!uid || uid === "null" || isNaN(parseInt(uid))) {
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
  textInput.value = "";
  refreshBook();
  addBubble("veyra", (b) => addText(b, "Fresh book started. Your next voice note opens a new one."));
});

// Language selector change handler
languageSelect.addEventListener("change", (e) => {
  currentLang = e.target.value;
  addBubble("veyra", (b) => {
    addText(b, `Language changed to ${LANG_NAMES[currentLang]}. I'll use this for your next message.`);
  });
});

// ---------------------------------------------------------------------------
// Download Excel/CSV
// ---------------------------------------------------------------------------

downloadExcelBtn.addEventListener("click", async () => {
  const uid = localStorage.getItem(UID_KEY);
  if (!uid) {
    addErrorBubble("No book to download. Record some entries first.");
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/ledger/${uid}/export.xlsx`);
    if (!res.ok) throw new Error("Download failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `veyra_ledger_${uid}.xlsx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addBubble("veyra", (b) => addText(b, "Excel file downloaded!"));
  } catch (err) {
    console.error("Download error:", err);
    addErrorBubble("Failed to download Excel file. Please try again.");
  }
});

downloadCsvBtn.addEventListener("click", async () => {
  const uid = localStorage.getItem(UID_KEY);
  if (!uid) {
    addErrorBubble("No book to download. Record some entries first.");
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/ledger/${uid}/export.csv`);
    if (!res.ok) throw new Error("Download failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `veyra_ledger_${uid}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    addBubble("veyra", (b) => addText(b, "CSV file downloaded! You can import this into Google Sheets."));
  } catch (err) {
    console.error("Download error:", err);
    addErrorBubble("Failed to download CSV file. Please try again.");
  }
});

// Mobile book panel toggle
const bookToggleBtn = document.getElementById("bookToggleBtn");
const bookCloseBtn = document.getElementById("bookCloseBtn");

function toggleBookPanel(show) {
  if (show) {
    bookCard.classList.add("is-visible");
    bookOverlay.classList.add("is-visible");
    bookToggleBtn.classList.add("is-active");
  } else {
    bookCard.classList.remove("is-visible");
    bookOverlay.classList.remove("is-visible");
    bookToggleBtn.classList.remove("is-active");
  }
}

if (bookToggleBtn) {
  bookToggleBtn.addEventListener("click", () => {
    const isVisible = bookCard.classList.contains("is-visible");
    toggleBookPanel(!isVisible);
  });
}

if (bookCloseBtn) {
  bookCloseBtn.addEventListener("click", () => {
    toggleBookPanel(false);
  });
}

if (bookOverlay) {
  bookOverlay.addEventListener("click", () => {
    toggleBookPanel(false);
  });
}

// ---------------------------------------------------------------------------
// Connection check + initial paint
// ---------------------------------------------------------------------------

async function checkModelStatus() {
  try {
    const res = await fetch(`${API_BASE}/model-status`);
    const data = await res.json();
    console.log("Model status:", data);
    return data;
  } catch (err) {
    console.error("Failed to check model status:", err);
    return null;
  }
}

(async function init() {
  await refreshBook();

  // Check if AI models are loaded
  const modelStatus = await checkModelStatus();
  if (modelStatus && modelStatus.cached_models && modelStatus.cached_models.length === 0) {
    // Models not loaded yet, show warning
    addBubble("veyra", (b) => {
      addText(b, "Note: AI voice models are loading. First voice request may take 1-2 minutes. You can use text input in the meantime.");
    });
  }
})();

// ---------------------------------------------------------------------------
// Vocabulary Teaching Modal
// ---------------------------------------------------------------------------

teachVocabBtn.addEventListener("click", () => {
  vocabModal.hidden = false;
  loadVocabularyList();
});

closeVocabModal.addEventListener("click", () => {
  vocabModal.hidden = true;
});

vocabModal.addEventListener("click", (e) => {
  if (e.target === vocabModal) {
    vocabModal.hidden = true;
  }
});

// Also close on Escape key
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !vocabModal.hidden) {
    vocabModal.hidden = true;
  }
});

vocabForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  
  const language = document.getElementById("vocabLanguage").value;
  const word = document.getElementById("vocabWord").value.trim();
  const meaning = document.getElementById("vocabMeaning").value.trim();
  const category = document.getElementById("vocabCategory").value;
  
  if (!word || !meaning) {
    alert("Please fill in all fields");
    return;
  }
  
  try {
    const form = new FormData();
    form.append("word", word);
    form.append("meaning", meaning);
    form.append("language", language);
    form.append("category", category);
    
    const res = await fetch(`${API_BASE}/vocabulary`, { method: "POST", body: form });
    const data = await res.json();
    
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    
    // Clear form
    document.getElementById("vocabWord").value = "";
    document.getElementById("vocabMeaning").value = "";
    
    // Refresh vocabulary list
    await loadVocabularyList();
    
    // Show success message
    addBubble("veyra", (b) => {
      addText(b, `I learned "${word}" in ${language}! Thanks for teaching me.`);
    });
    
  } catch (err) {
    console.error("Vocabulary teaching error:", err);
    alert(`Failed to teach vocabulary: ${err.message}`);
  }
});

async function loadVocabularyList() {
  const language = document.getElementById("vocabLanguage").value;
  
  try {
    const res = await fetch(`${API_BASE}/vocabulary/${language}`);
    const data = await res.json();
    
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    
    vocabList.innerHTML = "";
    
    if (!data.vocabulary || data.vocabulary.length === 0) {
      vocabList.innerHTML = '<div class="vocab-empty">No vocabulary learned yet for this language</div>';
      return;
    }
    
    for (const vocab of data.vocabulary) {
      const item = document.createElement("div");
      item.className = "vocab-item";
      item.innerHTML = `
        <div class="vocab-item-word">${vocab.word}</div>
        <div class="vocab-item-meaning">${vocab.meaning}</div>
        <div class="vocab-item-meta">
          <span>${vocab.category}</span>
          <span>•</span>
          <span>used ${vocab.usage_count} times</span>
        </div>
      `;
      vocabList.appendChild(item);
    }
    
  } catch (err) {
    console.error("Failed to load vocabulary:", err);
    vocabList.innerHTML = '<div class="vocab-empty">Failed to load vocabulary</div>';
  }
}

// Load vocabulary when language selection changes
document.getElementById("vocabLanguage").addEventListener("change", loadVocabularyList);
