/**
 * RAG Training Studio — Frontend Interactive Engine
 * Orchestrates the guided curriculum, visual labs, and live chat studio.
 */

// Step Definitions for the Interactive Guide Banner
const CURRICULUM_STEPS = [
  {
    step: 1,
    tab: "tab-chunking",
    title: "Step 1: Document Loading & Semantic Chunking",
    instruction: "Adjust the <strong>Chunk Size (350)</strong> and <strong>Overlap (70)</strong> sliders below, then click <strong>'Run Chunking Analysis'</strong>. Notice how sentences are preserved and how chunk overlap keeps thoughts intact across boundaries.",
    actionLabel: "▶ Run Chunking Demo",
    runAction: () => runChunkingAnalysis(),
  },
  {
    step: 2,
    tab: "tab-vectors",
    title: "Step 2: Embeddings & Vector Mathematics",
    instruction: "Observe how sentences are mapped into 1,536-dimensional vectors. Click <strong>'Calculate Vectors & Cosine Similarities'</strong> to see why Sentence A and B have high similarity despite having zero identical words!",
    actionLabel: "▶ Run Vector Math Demo",
    runAction: () => runVectorMath(),
  },
  {
    step: 3,
    tab: "tab-store",
    title: "Step 3: Vector Store Indexing & Top-K Retrieval",
    instruction: "See how the vector database works! Enter a query like <em>'What is the warp frequency?'</em> and click <strong>'Find Nearest Chunks'</strong> to inspect how cosine dot products rank relevant context.",
    actionLabel: "▶ Run Retrieval Demo",
    runAction: () => runVectorSearch(),
  },
  {
    step: 4,
    tab: "tab-prompt",
    title: "Step 4: Prompt Augmentation & Anti-Hallucination Guardrails",
    instruction: "Inspect the exact prompt constructed for the LLM. Toggle <strong>'Enable RAG Grounding'</strong> ON and OFF to compare how context injection prevents hallucinations and enforces source citations.",
    actionLabel: "▶ Assemble Prompt Demo",
    runAction: () => assemblePromptDemo(),
  },
  {
    step: 5,
    tab: "tab-chat",
    title: "Step 5: Live RAG Chat Studio & Pipeline Trace",
    instruction: "Ask questions to the RAG Assistant! Click any of the suggested chips or type a question. Watch the <strong>Live Pipeline Trace</strong> on the right to see exact latency and retrieved sources.",
    actionLabel: "▶ Test Sample Chat",
    runAction: () => sendSampleChatQuestion("What is the emergency warp surge limit and what happens if exceeded?"),
  }
];

let currentStepIndex = 0;
let sampleDocs = {
  quantum: "",
  policy: ""
};

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupSliders();
  setupGuide();
  setupSamplePicker();
  setupSettingsModal();
  setupChat();
  setupReindex();
  setupTerminalTaskSolver();

  await checkSystemStatus();
  await loadSampleDocuments();
  await loadAllDbChunks();

  // Initialize with Step 1
  setGuideStep(1);
});

// --------------------------------------------------------------------------
// Tab Navigation
// --------------------------------------------------------------------------
function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTabId = tab.getAttribute("data-tab");
      switchTab(targetTabId);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

  const targetBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  const targetPanel = document.getElementById(tabId);

  if (targetBtn) targetBtn.classList.add("active");
  if (targetPanel) targetPanel.classList.add("active");

  // Sync guide step indicator if applicable
  const stepMatch = CURRICULUM_STEPS.findIndex(s => s.tab === tabId);
  if (stepMatch !== -1) {
    updateGuideUI(stepMatch);
  }
}

// --------------------------------------------------------------------------
// Interactive Guide Engine
// --------------------------------------------------------------------------
function setupGuide() {
  document.getElementById("btnNextStep").addEventListener("click", () => {
    const nextIdx = (currentStepIndex + 1) % CURRICULUM_STEPS.length;
    setGuideStep(nextIdx + 1);
  });

  document.getElementById("btnGuideAction").addEventListener("click", () => {
    const stepConfig = CURRICULUM_STEPS[currentStepIndex];
    if (stepConfig && stepConfig.runAction) {
      stepConfig.runAction();
    }
  });

  document.querySelectorAll(".step-dot").forEach(dot => {
    dot.addEventListener("click", () => {
      const stepNum = parseInt(dot.getAttribute("data-step"));
      setGuideStep(stepNum);
    });
  });
}

function setGuideStep(stepNumber) {
  currentStepIndex = stepNumber - 1;
  const config = CURRICULUM_STEPS[currentStepIndex];
  if (!config) return;

  switchTab(config.tab);
  updateGuideUI(currentStepIndex);
}

function updateGuideUI(stepIdx) {
  currentStepIndex = stepIdx;
  const config = CURRICULUM_STEPS[stepIdx];

  document.getElementById("guideCurrentTitle").textContent = config.title;
  document.getElementById("guideInstructionText").innerHTML = config.instruction;
  document.getElementById("btnGuideAction").textContent = config.actionLabel;

  document.querySelectorAll(".step-dot").forEach((dot, idx) => {
    dot.classList.remove("active");
    if (idx === stepIdx) dot.classList.add("active");
    if (idx < stepIdx) dot.classList.add("completed");
  });
}

// --------------------------------------------------------------------------
// Status & Config
// --------------------------------------------------------------------------
async function checkSystemStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const badge = document.getElementById("statusBadge");
    const text = document.getElementById("statusText");

    if (data.has_api_key) {
      badge.className = "status-pill status-live";
      text.textContent = `OpenAI Connected (${data.chat_model})`;
    } else {
      badge.className = "status-pill status-simulated";
      text.textContent = `Offline Simulation Mode (${data.total_chunks} chunks indexed)`;
    }

    // Populate docs list in tab 6
    renderDocList(data.documents);
    document.getElementById("docCountBadge").textContent = `${data.documents.length} files`;
  } catch (err) {
    console.error("Failed to check status:", err);
  }
}

// --------------------------------------------------------------------------
// Sample Documents & Content Loading
// --------------------------------------------------------------------------
async function loadSampleDocuments() {
  try {
    const [resQ, resP] = await Promise.all([
      fetch("/api/document-content?name=quantum_orbit_manual.md"),
      fetch("/api/document-content?name=company_policy_2026.md")
    ]);
    const dataQ = await resQ.json();
    const dataP = await resP.json();

    sampleDocs.quantum = dataQ.content || "";
    sampleDocs.policy = dataP.content || "";

    // Set default in Tab 1
    const textArea = document.getElementById("chunkInputText");
    if (textArea && !textArea.value) {
      textArea.value = sampleDocs.quantum;
    }
  } catch (err) {
    console.error("Failed to load sample docs:", err);
  }
}

function setupSamplePicker() {
  const select = document.getElementById("chunkSampleSelect");
  const textarea = document.getElementById("chunkInputText");

  select.addEventListener("change", () => {
    const val = select.value;
    if (val === "quantum") textarea.value = sampleDocs.quantum;
    else if (val === "policy") textarea.value = sampleDocs.policy;
    else if (val === "custom") textarea.value = "";
  });
}

// --------------------------------------------------------------------------
// Step 1: Chunking Lab
// --------------------------------------------------------------------------
function setupSliders() {
  const sizeSlider = document.getElementById("chunkSizeSlider");
  const sizeVal = document.getElementById("chunkSizeValue");
  const overlapSlider = document.getElementById("chunkOverlapSlider");
  const overlapVal = document.getElementById("chunkOverlapValue");

  sizeSlider.addEventListener("input", () => {
    sizeVal.textContent = sizeSlider.value;
  });

  overlapSlider.addEventListener("input", () => {
    overlapVal.textContent = overlapSlider.value;
  });

  document.getElementById("btnRunChunking").addEventListener("click", runChunkingAnalysis);
}

async function runChunkingAnalysis() {
  const text = document.getElementById("chunkInputText").value.trim();
  const chunkSize = parseInt(document.getElementById("chunkSizeSlider").value);
  const chunkOverlap = parseInt(document.getElementById("chunkOverlapSlider").value);
  const container = document.getElementById("chunksContainer");

  if (!text) {
    alert("Please enter or select some text to chunk.");
    return;
  }

  container.innerHTML = '<div class="empty-state">Running recursive semantic splitter...</div>';

  try {
    const res = await fetch("/api/chunk-preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, chunk_size: chunkSize, chunk_overlap: chunkOverlap }),
    });
    const data = await res.json();

    document.getElementById("statTotalChars").textContent = `${data.total_chars} chars`;
    document.getElementById("statTotalTokens").textContent = `${data.total_tokens} tokens`;
    document.getElementById("statTotalChunks").textContent = `${data.chunk_count} chunks`;

    container.innerHTML = "";
    data.chunks.forEach(chunk => {
      const card = document.createElement("div");
      card.className = "chunk-card";
      card.innerHTML = `
        <div class="chunk-card-header">
          <strong>Chunk #${chunk.index}</strong>
          <span>${chunk.chars} chars | ${chunk.tokens} tokens</span>
        </div>
        <div class="chunk-card-body">${escapeHtml(chunk.text)}</div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<div class="empty-state">Error analyzing chunks: ${err}</div>`;
  }
}

// --------------------------------------------------------------------------
// Step 2: Vector Math Lab
// --------------------------------------------------------------------------
document.getElementById("btnRunVectorMath").addEventListener("click", runVectorMath);

async function runVectorMath() {
  const sA = document.getElementById("sentenceA").value.trim();
  const sB = document.getElementById("sentenceB").value.trim();
  const sC = document.getElementById("sentenceC").value.trim();

  if (!sA || !sB || !sC) {
    alert("Please provide all 3 sentences for comparison.");
    return;
  }

  try {
    const res = await fetch("/api/vector-math", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentences: [sA, sB, sC] }),
    });
    const data = await res.json();

    const m = data.matrix;
    updateSimCell("sim_AB", m[0][1]);
    updateSimCell("sim_AC", m[0][2]);
    updateSimCell("sim_BA", m[1][0]);
    updateSimCell("sim_BC", m[1][2]);
    updateSimCell("sim_CA", m[2][0]);
    updateSimCell("sim_CB", m[2][1]);

    // Update explanation
    const mathText = document.getElementById("mathInsightText");
    mathText.innerHTML = `
      Sim(A, B) = <strong>${m[0][1]}</strong> (High semantic match despite having 0 common words!)<br>
      Sim(A, C) = <strong>${m[0][2]}</strong> (Nearly orthogonal / unrelated topic).
    `;

    // Render sample dimensions
    const previewBox = document.getElementById("vectorPreviewList");
    let previewHtml = "";
    data.vectors_preview.forEach((v, idx) => {
      const letter = String.fromCharCode(65 + idx);
      previewHtml += `<div><strong>Vector ${letter} (Dim: ${v.dimension}, Norm: ${v.norm.toFixed(2)}):</strong>\n[${v.first_ten.join(", ")} ...]</div>\n`;
    });
    previewBox.innerHTML = `<pre>${previewHtml}</pre>`;
  } catch (err) {
    alert("Vector math computation failed: " + err);
  }
}

function updateSimCell(cellId, val) {
  const cell = document.getElementById(cellId);
  if (!cell) return;
  cell.textContent = val >= 0 ? `+${val.toFixed(4)}` : val.toFixed(4);
  cell.className = "sim-cell " + (val > 0.4 ? "sim-high" : val < 0.15 ? "sim-low" : "");
}

// --------------------------------------------------------------------------
// Step 3: Vector Store Lab
// --------------------------------------------------------------------------
document.getElementById("btnRunSearch").addEventListener("click", runVectorSearch);

async function loadAllDbChunks() {
  try {
    const res = await fetch("/api/chunks");
    const data = await res.json();
    const container = document.getElementById("dbChunksList");
    document.getElementById("dbTotalCount").textContent = `${data.total} chunks in DB`;

    container.innerHTML = "";
    data.chunks.forEach((chunk, i) => {
      const item = document.createElement("div");
      item.className = "trace-chunk-item";
      const source = chunk.metadata.source || "Unknown";
      const part = chunk.metadata.chunk_index || (i + 1);
      item.innerHTML = `
        <div style="display:flex; justify-content:space-between; color:var(--accent-cyan); font-weight:600; margin-bottom:0.25rem;">
          <span>[${source} - Part ${part}]</span>
          <span>${chunk.metadata.token_count || 0} tokens</span>
        </div>
        <div>${escapeHtml(chunk.text.substring(0, 160))}...</div>
      `;
      container.appendChild(item);
    });
  } catch (err) {
    console.error("Failed to load chunks:", err);
  }
}

async function runVectorSearch() {
  const query = document.getElementById("vectorSearchInput").value.trim();
  const topK = parseInt(document.getElementById("searchTopK").value);
  const resultsContainer = document.getElementById("searchResultsList");

  if (!query) return;

  resultsContainer.innerHTML = '<div class="empty-state">Computing cosine dot-product search...</div>';

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    const data = await res.json();

    document.getElementById("retrievalCountBadge").textContent = `${data.results.length} matches`;
    resultsContainer.innerHTML = "";

    if (data.results.length === 0) {
      resultsContainer.innerHTML = '<div class="empty-state">No matching chunks found.</div>';
      return;
    }

    data.results.forEach(item => {
      const card = document.createElement("div");
      card.className = "retrieval-result-card";
      const pct = Math.max(0, Math.min(100, (item.score * 100)));
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong style="color:var(--accent-cyan);">Rank #${item.rank}: ${escapeHtml(item.metadata.source || '')}</strong>
          <span class="stat-badge highlight">Score: ${item.score.toFixed(4)}</span>
        </div>
        <div class="score-progress-bar">
          <div class="score-progress-fill" style="width: ${pct}%;"></div>
        </div>
        <div style="font-size:0.83rem; color:#e5e7eb; margin-top:0.4rem; white-space:pre-wrap;">${escapeHtml(item.text)}</div>
      `;
      resultsContainer.appendChild(card);
    });
  } catch (err) {
    resultsContainer.innerHTML = `<div class="empty-state">Search error: ${err}</div>`;
  }
}

// --------------------------------------------------------------------------
// Step 4: Prompt Augmentation & Guardrails Lab
// --------------------------------------------------------------------------
document.getElementById("btnAssemblePrompt").addEventListener("click", assemblePromptDemo);
document.getElementById("toggleRagPrompt").addEventListener("change", assemblePromptDemo);

async function assemblePromptDemo() {
  const question = document.getElementById("promptTestQuery").value.trim();
  const includeRag = document.getElementById("toggleRagPrompt").checked;
  const sysBox = document.getElementById("systemPromptBox");
  const userBox = document.getElementById("userPromptBox");

  sysBox.textContent = (
    "SYSTEM PROMPT (Anti-Hallucination Guardrail):\n\n" +
    "You are an expert enterprise research assistant.\n" +
    "Answer the user's question accurately and concisely, relying STRICTLY and ONLY on the provided Reference Context.\n" +
    "Do NOT invent facts, extrapolate, or rely on outside speculation.\n" +
    "If the answer cannot be determined from the context, state: 'I do not have enough information in the provided documents to answer this question.'\n" +
    "Always cite source documents using square brackets (e.g. [Source 1])."
  );

  if (!includeRag) {
    userBox.textContent = `[WITHOUT RAG CONTEXT GROUNDING]\n\nUser Question: ${question}\n\nNotice: The model is given NO context. If this question asks about proprietary technical specs, it has to guess or admit it does not know.`;
    return;
  }

  userBox.textContent = "Retrieving nearest chunks and formatting augmented prompt...";

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, top_k: 2 }),
    });
    const searchData = await res.json();

    let contextPayload = "REFERENCE CONTEXT:\n";
    searchData.results.forEach((chunk, i) => {
      contextPayload += `\n--- [Source ${i + 1}: ${chunk.metadata.source} | Score: ${chunk.score.toFixed(3)}] ---\n${chunk.text}\n`;
    });

    contextPayload += `\nQUESTION:\n${question}\n\nGROUNDED ANSWER (with citations):`;
    userBox.textContent = contextPayload;
  } catch (err) {
    userBox.textContent = "Error assembling prompt: " + err;
  }
}

// --------------------------------------------------------------------------
// Step 5: Live RAG Chat Studio
// --------------------------------------------------------------------------
function setupChat() {
  const form = document.getElementById("chatForm");
  const input = document.getElementById("chatInput");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (q) {
      sendChatMessage(q);
      input.value = "";
    }
  });

  // Quick Chips
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q");
      sendChatMessage(q);
    });
  });
}

function sendSampleChatQuestion(q) {
  switchTab("tab-chat");
  sendChatMessage(q);
}

async function sendChatMessage(question) {
  const messagesBox = document.getElementById("chatMessages");

  // Append user bubble
  const userDiv = document.createElement("div");
  userDiv.className = "chat-bubble user-bubble";
  userDiv.innerHTML = `<p>${escapeHtml(question)}</p>`;
  messagesBox.appendChild(userDiv);
  messagesBox.scrollTop = messagesBox.scrollHeight;

  // Append temporary bot thinking bubble
  const botDiv = document.createElement("div");
  botDiv.className = "chat-bubble bot-bubble";
  botDiv.innerHTML = `
    <div class="bubble-header">
      <strong>RAG Assistant</strong>
      <span class="bubble-time">Retrieving & generating...</span>
    </div>
    <p>⚡ Searching vector store and synthesizing grounded answer...</p>
  `;
  messagesBox.appendChild(botDiv);
  messagesBox.scrollTop = messagesBox.scrollHeight;

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, include_rag: true }),
    });
    const data = await res.json();

    // Render formatted answer with clickable citation badges
    const formattedAnswer = renderAnswerWithCitations(data.answer, data.retrieved_chunks);
    botDiv.innerHTML = `
      <div class="bubble-header">
        <strong>RAG Assistant</strong>
        <span class="bubble-time">${data.total_ms}ms</span>
      </div>
      <div>${formattedAnswer}</div>
    `;
    messagesBox.scrollTop = messagesBox.scrollHeight;

    // Update Trace Drawer on the right
    renderTrace(data);
  } catch (err) {
    botDiv.innerHTML = `<p style="color:var(--accent-rose);">Error executing query: ${err}</p>`;
  }
}

function renderAnswerWithCitations(rawText, chunks) {
  let html = escapeHtml(rawText);
  // Match citations like [Source 1] or [Source 1: filename]
  html = html.replace(/\[Source\s*(\d+)(?::\s*([^\]]+))?\]/gi, (match, num, file) => {
    const idx = parseInt(num) - 1;
    const sourceName = file || (chunks[idx] ? chunks[idx].metadata.source : `Source ${num}`);
    return `<span class="citation-tag" onclick="highlightTraceChunk(${idx})" title="Click to view retrieved chunk">${match}</span>`;
  });
  return html.replace(/\n/g, "<br>");
}

function renderTrace(data) {
  document.getElementById("traceTimingBadge").textContent = `${data.total_ms} ms total`;
  document.getElementById("traceRetrievalMs").textContent = `${data.retrieval_ms} ms`;
  document.getElementById("traceGenMs").textContent = `${data.generation_ms} ms`;
  document.getElementById("traceTotalMs").textContent = `${data.total_ms} ms`;

  const traceList = document.getElementById("traceChunksList");
  traceList.innerHTML = "";

  if (!data.retrieved_chunks || data.retrieved_chunks.length === 0) {
    traceList.innerHTML = '<div class="empty-state">No chunks retrieved for this question.</div>';
    return;
  }

  data.retrieved_chunks.forEach((c, idx) => {
    const item = document.createElement("div");
    item.className = "trace-chunk-item";
    item.id = `traceChunkItem_${idx}`;
    item.innerHTML = `
      <div style="display:flex; justify-content:space-between; color:var(--accent-cyan); font-weight:600; margin-bottom:0.25rem;">
        <span>[Source ${idx + 1}: ${escapeHtml(c.metadata.source)}]</span>
        <span class="stat-badge highlight">Sim: ${c.score.toFixed(4)}</span>
      </div>
      <div>${escapeHtml(c.text)}</div>
    `;
    traceList.appendChild(item);
  });
}

window.highlightTraceChunk = function(idx) {
  const item = document.getElementById(`traceChunkItem_${idx}`);
  if (item) {
    item.scrollIntoView({ behavior: "smooth", block: "center" });
    item.style.borderColor = "var(--accent-primary)";
    item.style.backgroundColor = "rgba(99, 102, 241, 0.25)";
    setTimeout(() => {
      item.style.borderColor = "";
      item.style.backgroundColor = "";
    }, 2000);
  }
};

// --------------------------------------------------------------------------
// Tab 6: Document Manager
// --------------------------------------------------------------------------
function renderDocList(docs) {
  const list = document.getElementById("docList");
  list.innerHTML = "";

  docs.forEach((doc, i) => {
    const li = document.createElement("li");
    li.className = "doc-item" + (i === 0 ? " active" : "");
    li.innerHTML = `
      <div>
        <strong style="display:block;">${escapeHtml(doc.name)}</strong>
        <span style="font-size:0.75rem; color:var(--text-dim);">${(doc.size_bytes / 1024).toFixed(1)} KB</span>
      </div>
      <button class="btn btn-subtle" style="font-size:0.75rem; padding:0.25rem 0.5rem;">View</button>
    `;
    li.addEventListener("click", () => {
      document.querySelectorAll(".doc-item").forEach(d => d.classList.remove("active"));
      li.classList.add("active");
      previewDocument(doc.name);
    });
    list.appendChild(li);
  });

  if (docs.length > 0) {
    previewDocument(docs[0].name);
  }
}

async function previewDocument(filename) {
  document.getElementById("docViewerTitle").textContent = filename;
  const viewer = document.getElementById("docViewerContent");
  viewer.textContent = "Loading document content...";
  try {
    const res = await fetch(`/api/document-content?name=${encodeURIComponent(filename)}`);
    const data = await res.json();
    viewer.textContent = data.content;
  } catch (err) {
    viewer.textContent = "Failed to load document content: " + err;
  }
}

// --------------------------------------------------------------------------
// Re-index & Settings Modal
// --------------------------------------------------------------------------
function setupReindex() {
  const btn = document.getElementById("btnReindex");
  btn.addEventListener("click", async () => {
    if (!confirm("Re-index all documents in data/?")) return;
    btn.textContent = "⏳ Re-indexing...";
    btn.disabled = true;

    try {
      const res = await fetch("/api/reindex", { method: "POST" });
      const data = await res.json();
      alert(data.message);
      await checkSystemStatus();
      await loadAllDbChunks();
    } catch (err) {
      alert("Re-index failed: " + err);
    } finally {
      btn.textContent = "🔄 Re-index Data";
      btn.disabled = false;
    }
  });
}

function setupSettingsModal() {
  const modal = document.getElementById("settingsModal");
  const openBtn = document.getElementById("btnOpenSettings");
  const closeBtn = document.getElementById("btnCloseSettings");
  const cancelBtn = document.getElementById("btnCancelSettings");
  const saveBtn = document.getElementById("btnSaveSettings");

  openBtn.addEventListener("click", () => modal.classList.add("open"));
  closeBtn.addEventListener("click", () => modal.classList.remove("open"));
  cancelBtn.addEventListener("click", () => modal.classList.remove("open"));

  saveBtn.addEventListener("click", async () => {
    const apiKey = document.getElementById("inputApiKey").value.trim();
    const chatModel = document.getElementById("selectChatModel").value;
    const topK = parseInt(document.getElementById("inputTopK").value);

    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey, chat_model: chatModel, top_k: topK }),
      });
      const data = await res.json();
      alert("Settings saved successfully!");
      modal.classList.remove("open");
      await checkSystemStatus();
    } catch (err) {
      alert("Failed to save settings: " + err);
    }
  });
}

// Helper: Escape HTML
function escapeHtml(text) {
  if (!text) return "";
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return text.replace(/[&<>"']/g, m => map[m]);
}

// --------------------------------------------------------------------------
// Task Solver Terminal
// --------------------------------------------------------------------------
function setupTerminalTaskSolver() {
  const input = document.getElementById("terminalTaskInput");
  const solveBtn = document.getElementById("btnSolveTask");
  const clearBtn = document.getElementById("btnClearTerminal");
  const output = document.getElementById("terminalOutput");

  if (!solveBtn || !input) return;

  // Preset chips
  document.querySelectorAll(".terminal-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const task = chip.getAttribute("data-task");
      input.value = task;
      executeTaskSolve(task);
    });
  });

  solveBtn.addEventListener("click", () => {
    const task = input.value.trim();
    if (!task) {
      alert("Please paste or type a task first.");
      return;
    }
    executeTaskSolve(task);
  });

  clearBtn.addEventListener("click", () => {
    output.innerHTML = `
      <div class="terminal-welcome">
        <p>Terminal cleared. Paste any task above and press <strong>"Execute & Generate Step-by-Step Solution"</strong>.</p>
      </div>
    `;
    input.value = "";
  });
}

async function executeTaskSolve(taskText) {
  const output = document.getElementById("terminalOutput");
  output.innerHTML = `
    <div style="color:var(--accent-cyan); font-family:var(--font-mono); line-height:1.8;">
      <div>[+] Analyzing task requirements & target architecture...</div>
      <div>[+] Synthesizing step-by-step implementation plan...</div>
      <div>[+] Writing clean, production-grade Python solution...</div>
      <div>[+] Assembling verification & test commands...</div>
      <div style="color:var(--accent-amber);">⚡ Compiling response...</div>
    </div>
  `;

  try {
    const res = await fetch("/api/solve-task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: taskText }),
    });
    const data = await res.json();

    if (data.error) {
      output.innerHTML = `<div style="color:var(--accent-rose);">Error: ${escapeHtml(data.error)}</div>`;
      return;
    }

    const rendered = renderTerminalMarkdown(data.solution_markdown);
    output.innerHTML = `
      <div class="terminal-solution-rendered">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; padding-bottom:0.5rem; border-bottom:1px solid var(--border-subtle);">
          <span class="stat-badge highlight">${data.is_live_ai ? "⚡ Live AI Architect (GPT-4o)" : "💡 Pattern-Matched Solution Engine"}</span>
          <span style="font-size:0.75rem; color:var(--text-dim);">Task processed successfully</span>
        </div>
        ${rendered}
      </div>
    `;

    // Attach copy button handlers
    output.querySelectorAll(".terminal-copy-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const code = btn.getAttribute("data-code");
        navigator.clipboard.writeText(code).then(() => {
          btn.textContent = "✓ Copied!";
          setTimeout(() => (btn.textContent = "📋 Copy Code"), 2000);
        });
      });
    });

  } catch (err) {
    output.innerHTML = `<div style="color:var(--accent-rose);">Execution failed: ${err}</div>`;
  }
}

function renderTerminalMarkdown(md) {
  if (!md) return "";

  // Convert code blocks
  let html = md.replace(/```([a-zA-Z0-9_]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const escapedCode = escapeHtml(code.trim());
    const rawAttr = encodeURIComponent(code.trim());
    return `
      <pre>
        <button class="terminal-copy-btn" data-code="${decodeURIComponent(rawAttr)}">📋 Copy Code</button>
        <code>${escapedCode}</code>
      </pre>
    `;
  });

  // Headers
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Bold & Italics
  html = html.replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/gim, '<em>$1</em>');

  // Lists
  html = html.replace(/^\s*\d+\.\s+(.*$)/gim, '<div style="margin:0.25rem 0 0.25rem 1rem;"><strong>•</strong> $1</div>');
  html = html.replace(/^\s*[-*]\s+(.*$)/gim, '<div style="margin:0.25rem 0 0.25rem 1rem;"><strong>-</strong> $1</div>');

  // Paragraph line breaks
  html = html.replace(/\n\n/g, '<br><br>');

  return html;
}

