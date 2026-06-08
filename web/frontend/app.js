const state = {
  taskId: null,
  pollTimer: null,
  lastEventCount: 0,
  lastTask: null,
  selectedView: null,
  selectedStageKey: null,
};

const stages = [
  ["phase1", "Phase1", "PICO 分解"],
  ["phase1_review", "PICO 确认", "等待编辑确认"],
  ["phase2", "Phase2", "文献检索"],
  ["phase2_review", "检索确认", "等待确认检索"],
  ["phase3_record", "Phase3", "题录筛选"],
  ["phase3_record_review", "PDF 上传", "等待文件"],
  ["phase3_full_text", "全文评估", "RAG 评估"],
  ["phase4", "Phase4", "证据评价"],
  ["phase5", "Phase5", "推荐形成"],
  ["completed", "完成", "最终推荐"],
];

const $ = (id) => document.getElementById(id);

function setHidden(el, hidden) {
  el.classList.toggle("hidden", hidden);
}

function jsonText(value) {
  return JSON.stringify(value ?? null, null, 2);
}

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function setBusy(busy) {
  $("submitBtn").disabled = busy;
  $("submitBtn").textContent = busy ? "处理中" : "提交";
}

function resetSelectedView() {
  state.selectedView = null;
  state.selectedStageKey = null;
}

function resetPanels() {
  stopPolling();
  state.taskId = null;
  state.lastEventCount = 0;
  state.lastTask = null;
  resetSelectedView();
  setHidden($("answerPanel"), true);
  setHidden($("taskPanel"), false);
  setHidden($("stageHistoryPanel"), true);
  setHidden($("finalPanel"), true);
  setHidden($("picoEditorPanel"), true);
  setHidden($("phase2ReviewPanel"), true);
  setHidden($("pdfPanel"), true);
  $("stageList").innerHTML = "";
  $("eventLog").innerHTML = "";
}

function showRoutingState(disease) {
  resetPanels();
  $("workspaceTitle").textContent = "正在处理";
  $("workspaceSubtitle").textContent = `${disease} · 等待后端响应`;
  $("taskBadge").textContent = "routing";
  $("taskBadge").classList.remove("muted");
  appendLocalEvent("info", "已提交问题，正在判断响应路径");
}

function updateHealth(ok) {
  $("healthBadge").textContent = ok ? "online" : "offline";
  $("healthBadge").classList.toggle("muted", !ok);
}

async function loadDiseases() {
  try {
    await request("/api/health");
    updateHealth(true);
    const data = await request("/api/diseases");
    $("diseaseSelect").innerHTML = "";
    for (const disease of data.diseases) {
      const option = document.createElement("option");
      option.value = disease;
      option.textContent = disease;
      $("diseaseSelect").appendChild(option);
    }
  } catch (err) {
    updateHealth(false);
    appendLocalEvent("error", err.message);
  }
}

function appendLocalEvent(level, message) {
  const row = document.createElement("div");
  row.className = `event-row ${level}`;
  row.innerHTML = `<span>${new Date().toLocaleTimeString()}</span><span class="event-level">${level}</span><span>${escapeHtml(message)}</span>`;
  $("eventLog").appendChild(row);
  $("eventLog").scrollTop = $("eventLog").scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function showKnowledgeAnswer(payload) {
  stopPolling();
  setHidden($("answerPanel"), false);
  setHidden($("taskPanel"), true);
  $("workspaceTitle").textContent = "知识库可信响应";
  $("workspaceSubtitle").textContent = payload.evidence?.route?.理由 || "检索候选可直接回答";
  $("taskBadge").textContent = "answered";
  $("taskBadge").classList.remove("muted");
  $("answerText").textContent = payload.answer || "(空)";
  $("evidenceJson").textContent = jsonText(payload.evidence);
}

function showReasoningTask(payload) {
  state.taskId = payload.task_id;
  resetSelectedView();
  setHidden($("answerPanel"), true);
  setHidden($("taskPanel"), false);
  $("workspaceTitle").textContent = "推理任务";
  $("workspaceSubtitle").textContent = `task_id: ${payload.task_id}`;
  renderTask(payload.task);
  startPolling();
}

async function submitQuestion(event) {
  event.preventDefault();
  const disease = $("diseaseSelect").value;
  const question = $("questionInput").value.trim();
  if (!question) {
    appendLocalEvent("error", "请输入临床问题");
    return;
  }
  setBusy(true);
  showRoutingState(disease);
  try {
    const payload = await request("/api/ask", {
      method: "POST",
      body: JSON.stringify({ disease, question }),
    });
    if (payload.mode === "knowledge_base") {
      showKnowledgeAnswer(payload);
    } else {
      showReasoningTask(payload);
    }
  } catch (err) {
    $("workspaceTitle").textContent = "请求失败";
    $("workspaceSubtitle").textContent = err.message;
    $("taskBadge").textContent = "error";
    $("taskBadge").classList.remove("muted");
    appendLocalEvent("error", err.message);
  } finally {
    setBusy(false);
  }
}

function stageStateClass(task, key) {
  const order = stages.map((item) => item[0]);
  const currentIndex = order.indexOf(task.current_stage);
  const itemIndex = order.indexOf(key);
  if (key === task.current_stage) return "active";
  if (task.status === "completed" || (currentIndex >= 0 && itemIndex < currentIndex)) return "done";
  return "";
}

function stageViewFor(key) {
  if (key === "phase1" || key === "phase1_review") return "pico";
  if (key === "phase2" || key === "phase2_review") return "literature";
  if (key === "phase3_record" || key === "phase3_record_review") return "screening";
  if (key === "phase5" || key === "completed") return "final";
  return "history";
}

function defaultStageForView(view, task) {
  if (view === "pico") return "phase1_review";
  if (view === "literature") return "phase2_review";
  if (view === "screening") return "phase3_record_review";
  if (view === "final") return "completed";
  return task.current_stage || "completed";
}

function hasViewData(task, view) {
  const artifacts = task.artifacts || {};
  if (view === "pico") return Boolean(artifacts.pico?.data);
  if (view === "literature") return Boolean(artifacts.literature_search);
  if (view === "screening") {
    return Boolean(artifacts.screening_summary || artifacts.record_included || artifacts.pdf_manifest);
  }
  if (view === "final") return Boolean(artifacts.final_recommendation);
  return true;
}

function stageReached(task, key) {
  return stageStateClass(task, key) !== "";
}

function defaultViewForTask(task) {
  if (task.awaiting === "pico_review" && hasViewData(task, "pico")) return ["pico", "phase1_review"];
  if (task.awaiting === "phase2_review" && hasViewData(task, "literature")) return ["literature", "phase2_review"];
  if (task.awaiting === "pdf_upload" && hasViewData(task, "screening")) return ["screening", "phase3_record_review"];
  if (task.status === "completed" && hasViewData(task, "final")) return ["final", "completed"];

  const currentView = stageViewFor(task.current_stage || "");
  if (hasViewData(task, currentView)) {
    return [currentView, task.current_stage || defaultStageForView(currentView, task)];
  }
  return ["history", task.current_stage || "completed"];
}

function activeViewForTask(task) {
  if (!state.selectedView || !hasViewData(task, state.selectedView)) {
    const [view, stageKey] = defaultViewForTask(task);
    state.selectedView = view;
    state.selectedStageKey = stageKey;
  }
  return state.selectedView;
}

function switchStageView(task, key) {
  if (!stageReached(task, key)) return;
  const preferredView = stageViewFor(key);
  state.selectedView = hasViewData(task, preferredView) ? preferredView : "history";
  state.selectedStageKey = key;
  renderTask(task);
}

function renderStages(task) {
  $("stageList").innerHTML = "";
  for (const [key, title, sub] of stages) {
    const cls = stageStateClass(task, key);
    const reached = stageReached(task, key);
    const selected = state.selectedStageKey === key;
    const item = document.createElement("div");
    item.className = `stage-item ${cls} ${selected ? "selected" : ""} ${reached ? "clickable" : ""}`;
    item.dataset.stageKey = key;
    if (reached) {
      item.setAttribute("role", "button");
      item.tabIndex = 0;
      item.title = "查看该阶段结果";
      item.addEventListener("click", () => switchStageView(task, key));
      item.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          switchStageView(task, key);
        }
      });
    }
    item.innerHTML = `
      <div class="stage-info">
        <div class="stage-name">${escapeHtml(title)}</div>
        <div class="stage-meta">${escapeHtml(sub)}</div>
      </div>
    `;
    $("stageList").appendChild(item);
  }
}

function renderEvents(task) {
  const events = task.events || [];
  if (events.length < state.lastEventCount) {
    $("eventLog").innerHTML = "";
    state.lastEventCount = 0;
  }
  for (const event of events.slice(state.lastEventCount)) {
    const row = document.createElement("div");
    row.className = `event-row ${event.level || "info"}`;
    row.innerHTML = `<span>${escapeHtml(event.time || "")}</span><span class="event-level">${escapeHtml(event.level || "info")}</span><span>${escapeHtml(event.message || "")}</span>`;
    $("eventLog").appendChild(row);
  }
  state.lastEventCount = events.length;
  $("eventLog").scrollTop = $("eventLog").scrollHeight;
}

function stageDisplayName(key) {
  const stage = stages.find((item) => item[0] === key);
  return stage ? `${stage[1]} · ${stage[2]}` : "阶段结果";
}

function historyTermsForStage(key) {
  const map = {
    phase1: ["Phase1", "PICO"],
    phase1_review: ["PICO"],
    phase2: ["Phase2", "文献检索"],
    phase2_review: ["文献检索", "检索结果"],
    phase3_record: ["Phase3 题录", "题录筛选"],
    phase3_record_review: ["题录筛选", "PDF"],
    phase3_full_text: ["Phase3 全文", "Phase3_full_text", "全文评估"],
    phase4: ["Phase4", "证据评价"],
    phase5: ["Phase5", "推荐形成"],
    completed: ["推理任务完成", "所有阶段结果", "最终推荐"],
  };
  return map[key] || [key];
}

function renderStageHistory(task, activeView) {
  const shouldShow = activeView === "history";
  setHidden($("stageHistoryPanel"), !shouldShow);
  if (!shouldShow) return;

  const stageKey = state.selectedStageKey || task.current_stage || "completed";
  const terms = historyTermsForStage(stageKey);
  const matchedEvents = (task.events || []).filter((event) => {
    const message = event.message || "";
    return terms.some((term) => message.includes(term));
  });
  const phaseOutput = task.phase_outputs?.[stageKey] || [];

  $("stageHistoryTitle").textContent = stageDisplayName(stageKey);
  if (phaseOutput.length) {
    $("stageHistoryText").textContent = `该阶段有 ${phaseOutput.length} 条运行输出，详见下方日志。`;
  } else if (matchedEvents.length) {
    $("stageHistoryText").textContent = `该阶段共有 ${matchedEvents.length} 条相关事件。`;
  } else {
    $("stageHistoryText").textContent = "暂无单独的结构化结果，完整过程可在实时日志中查看。";
  }
  $("stageHistoryJson").textContent = jsonText({
    stage: stageKey,
    events: matchedEvents,
    phase_outputs: phaseOutput,
  });
}

/* ── Phase1: PICO Editor ─────────────────── */

function renderPico(task, activeView) {
  const artifact = task.artifacts?.pico;
  const shouldShow = activeView === "pico" && artifact?.data;
  setHidden($("picoEditorPanel"), !shouldShow);
  if (shouldShow && $("picoEditor").dataset.loadedForTask !== task.task_id) {
    $("picoEditor").value = jsonText(artifact.data);
    $("picoEditor").dataset.loadedForTask = task.task_id;
  }
  if (shouldShow) {
    const canContinue = task.awaiting === "pico_review";
    setHidden($("continuePicoBtn"), !canContinue);
    $("picoEditor").readOnly = !canContinue;
  }
}

/* ── Phase2: Literature Search Review ────── */

function renderPhase2Review(task, activeView) {
  const artifact = task.artifacts?.literature_search;
  const shouldShow = activeView === "literature" && artifact;
  setHidden($("phase2ReviewPanel"), !shouldShow);
  if (!shouldShow) return;
  setHidden($("continuePhase2Btn"), task.awaiting !== "phase2_review");

  $("lsTotalCount").textContent = artifact.total_count || 0;

  const sample = artifact.sample || [];
  $("lsSampleList").innerHTML = "";
  for (const paper of sample) {
    const card = document.createElement("div");
    card.className = "paper-card";
    card.innerHTML = `
      <div class="paper-title">${escapeHtml(paper.title || "(无标题)")}</div>
      <div class="paper-meta">
        <span>PMID: ${escapeHtml(paper.pmid || "—")}</span>
        <span>${escapeHtml(paper.year || "—")}</span>
      </div>
    `;
    $("lsSampleList").appendChild(card);
  }
}

/* ── Phase3: Screening + PDF Upload ──────── */

function renderPhase3Review(task, activeView) {
  const shouldShow = activeView === "screening" && hasViewData(task, "screening");
  setHidden($("pdfPanel"), !shouldShow);
  if (!shouldShow) return;
  const canContinue = task.awaiting === "pdf_upload";
  setHidden($("continuePdfBtn"), !canContinue);

  // Screening summary
  const summary = task.artifacts?.screening_summary || {};
  $("scTotalScreened").textContent = summary.total_screened ?? "—";
  $("scTotalIncluded").textContent = summary.total_included ?? "—";
  $("scTotalExcluded").textContent = summary.total_excluded ?? "—";

  const sample = summary.sample_included || [];
  $("scSampleList").innerHTML = "";
  for (const paper of sample) {
    const card = document.createElement("div");
    card.className = "paper-card";
    card.innerHTML = `
      <div class="paper-title">${escapeHtml(paper.title || "(无标题)")}</div>
      <div class="paper-meta">
        <span>PMID: ${escapeHtml(paper.pmid || "—")}</span>
        <span>${escapeHtml(paper.year || "—")}</span>
        <span style="color:var(--green)">✓ ${escapeHtml(paper.verdict || "")}</span>
      </div>
    `;
    $("scSampleList").appendChild(card);
  }

  // PDF status
  const manifest = task.artifacts?.pdf_manifest?.data || {};
  const missingCount = manifest.missing_pdf_count ?? 0;
  const totalPapers = manifest.total_paper_count ?? 0;
  const existingCount = manifest.existing_pdf_count ?? 0;
  const missingPdfs = manifest.missing_pdfs || [];

  const uploadSection = $("pdfUploadSection");
  const fileRow = $("pdfFileRow");
  const uploadBtn = $("uploadBtn");
  const pdfInput = $("pdfInput");
  const continueBtn = $("continuePdfBtn");

  if (totalPapers === 0) {
    uploadSection.className = "upload-strip";
    $("pdfStatusIcon").textContent = "INFO";
    $("pdfStatusText").textContent = "无文献需要上传";
    setHidden(fileRow, false);
    pdfInput.disabled = true;
    uploadBtn.disabled = true;
    uploadBtn.textContent = "无文献需要上传";
    continueBtn.disabled = false;
  } else if (missingCount === 0) {
    uploadSection.className = "upload-strip has-files";
    $("pdfStatusIcon").textContent = "OK";
    $("pdfStatusText").textContent = `文献已齐全（${existingCount}/${totalPapers}），您无需上传`;
    setHidden(fileRow, false);
    pdfInput.disabled = true;
    uploadBtn.disabled = true;
    uploadBtn.textContent = "文献已存在，您无需上传";
    continueBtn.disabled = false;
  } else {
    uploadSection.className = "upload-strip needs-files";
    $("pdfStatusIcon").textContent = "NEED";
    $("pdfStatusText").textContent = `缺失 ${missingCount}/${totalPapers} 篇文献 PDF，请上传`;
    setHidden(fileRow, false);
    pdfInput.disabled = !canContinue;
    uploadBtn.disabled = !canContinue;
    uploadBtn.textContent = "上传 PDF";
    continueBtn.disabled = false; // user can continue even without uploading all
  }

  $("missingPdfList").innerHTML = "";
  if (missingCount > 0) {
    for (const paper of missingPdfs.slice(0, 6)) {
      const card = document.createElement("div");
      card.className = "paper-card";
      card.innerHTML = `
        <div class="paper-title">${escapeHtml(paper.title || "(无标题)")}</div>
        <div class="paper-meta">
          <span>UID: ${escapeHtml(paper.paper_uid || "—")}</span>
          <span>PMID: ${escapeHtml(paper.pmid || "—")}</span>
          <span>DOI: ${escapeHtml(paper.doi || "—")}</span>
        </div>
      `;
      $("missingPdfList").appendChild(card);
    }
    if (missingPdfs.length > 6) {
      const note = document.createElement("div");
      note.className = "small-note";
      note.textContent = `还有 ${missingPdfs.length - 6} 篇缺失文献，完整清单见下方 PDF 清单。`;
      $("missingPdfList").appendChild(note);
    }
  }

  // Record / manifest JSON
  $("recordJson").textContent = jsonText(task.artifacts?.record_included?.data || []);
  $("manifestJson").textContent = jsonText(manifest);
}

/* ── Final Recommendation ────────────────── */

function renderFinal(task) {
  const final = task.artifacts?.final_recommendation;
  const shouldShow = activeViewForTask(task) === "final" && final;
  setHidden($("finalPanel"), !shouldShow);
  if (!shouldShow) return;
  $("finalRecommendation").textContent = final.recommendation || "(空)";
  $("finalJson").textContent = jsonText(final.final_result || final.raw);
}

/* ── Main Render ──────────────────────────── */

function renderTask(task) {
  if (!task) return;
  state.lastTask = task;
  const activeView = activeViewForTask(task);
  $("taskBadge").textContent = task.status || "task";
  $("taskBadge").classList.toggle("muted", task.status !== "running" && task.status !== "waiting");
  $("workspaceSubtitle").textContent = `task_id: ${task.task_id} · ${task.current_stage || ""}`;
  renderStages(task);
  renderEvents(task);
  renderStageHistory(task, activeView);
  renderPico(task, activeView);
  renderPhase2Review(task, activeView);
  renderPhase3Review(task, activeView);
  renderFinal(task);
  if (task.status === "completed" || task.status === "failed") {
    stopPolling();
  }
}

/* ── Polling ──────────────────────────────── */

async function pollTask() {
  if (!state.taskId) return;
  try {
    const payload = await request(`/api/tasks/${state.taskId}`);
    renderTask(payload.task);
  } catch (err) {
    appendLocalEvent("error", err.message);
    stopPolling();
  }
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(pollTask, 1800);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

/* ── Phase1: Continue ─────────────────────── */

async function continuePico() {
  $("picoError").textContent = "";
  let pico;
  try {
    pico = JSON.parse($("picoEditor").value);
  } catch (err) {
    $("picoError").textContent = err.message;
    return;
  }
  try {
    resetSelectedView();
    const payload = await request(`/api/tasks/${state.taskId}/continue`, {
      method: "POST",
      body: JSON.stringify({ pico }),
    });
    $("picoEditor").dataset.loadedForTask = "";
    renderTask(payload.task);
    startPolling();
  } catch (err) {
    $("picoError").textContent = err.message;
  }
}

/* ── Phase2: Continue ─────────────────────── */

async function continuePhase2() {
  try {
    resetSelectedView();
    const payload = await request(`/api/tasks/${state.taskId}/continue`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    renderTask(payload.task);
    startPolling();
  } catch (err) {
    appendLocalEvent("error", err.message);
  }
}

/* ── PDF Upload ───────────────────────────── */

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function uploadPdfs() {
  const files = Array.from($("pdfInput").files || []);
  if (!files.length) {
    $("uploadStatus").textContent = "未选择文件";
    return;
  }
  $("uploadBtn").disabled = true;
  $("uploadStatus").textContent = "上传中…";
  try {
    const payloadFiles = [];
    for (const file of files) {
      payloadFiles.push({
        name: file.name,
        content_base64: await readFileAsBase64(file),
      });
    }
    const payload = await request(`/api/tasks/${state.taskId}/upload`, {
      method: "POST",
      body: JSON.stringify({ files: payloadFiles }),
    });
    $("uploadStatus").textContent = `已保存 ${payload.saved.length} 个文件`;
    renderTask(payload.task);
  } catch (err) {
    $("uploadStatus").textContent = err.message;
  } finally {
    $("uploadBtn").disabled = false;
  }
}

/* ── Phase3: Continue ─────────────────────── */

async function continuePdf() {
  try {
    resetSelectedView();
    const payload = await request(`/api/tasks/${state.taskId}/continue`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    renderTask(payload.task);
    startPolling();
  } catch (err) {
    $("uploadStatus").textContent = err.message;
  }
}

/* ── Sample ───────────────────────────────── */

function useSample() {
  $("diseaseSelect").value = "Rheumatoid Arthritis (RA)";
  $("questionInput").value =
    "Should patients with RA on DMARDs who are in low disease activity gradually taper off DMARDs, abruptly withdraw DMARDs, or continue DMARDS at the same doses?";
}

/* ── Boot ─────────────────────────────────── */

function boot() {
  $("askForm").addEventListener("submit", submitQuestion);
  $("sampleBtn").addEventListener("click", useSample);
  $("continuePicoBtn").addEventListener("click", continuePico);
  $("continuePhase2Btn").addEventListener("click", continuePhase2);
  $("uploadBtn").addEventListener("click", uploadPdfs);
  $("continuePdfBtn").addEventListener("click", continuePdf);
  $("clearLogBtn").addEventListener("click", () => {
    $("eventLog").innerHTML = "";
    state.lastEventCount = 0;
  });
  loadDiseases();
}

boot();
