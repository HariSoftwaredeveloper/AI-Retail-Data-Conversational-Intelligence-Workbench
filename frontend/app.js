// Frontend Client Application for Retail Workbench

const API_BASE = "";
let currentDataset = "";
let sessionId = "session_" + Math.random().toString(36).substring(2, 9);
let rawDatasetData = [];
let cleanDatasetData = [];

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initUpload();
  initDatasetControls();
  initChat();
  initRetailHub();
  initEvalRunner();
  refreshDatasets();
});

// 1. Tabs Navigation
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      tab.classList.add("active");
      const target = tab.getAttribute("data-tab");
      document.getElementById(target).classList.add("active");

      if (target === "retail-tab") loadRetailHub();
      if (target === "eval-tab") loadEvaluationResults();
    });
  });
}

// 2. Ingestion & Upload
function initUpload() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("csv-file-input");

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("hover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("hover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("hover");
    if (e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadFile(e.target.files[0]);
    }
  });
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/datasets/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    alert(`Dataset '${data.dataset_name}' uploaded successfully (${data.rows} rows)!`);
    await refreshDatasets();
    selectDataset(data.dataset_name);
  } catch (err) {
    alert("Upload failed: " + err.message);
  }
}

// 3. Dataset Selection & Controls
function initDatasetControls() {
  const select = document.getElementById("active-dataset-select");
  const btnClean = document.getElementById("btn-run-clean");

  select.addEventListener("change", (e) => {
    selectDataset(e.target.value);
  });

  btnClean.addEventListener("click", async () => {
    const activeName = document.getElementById("active-dataset-select").value || currentDataset;
    if (!activeName) return;
    btnClean.disabled = true;
    btnClean.textContent = "Cleaning...";
    try {
      const formData = new FormData();
      formData.append("dataset_name", activeName);
      const res = await fetch(`${API_BASE}/api/pipeline/clean`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(await res.text());
      const cleanRes = await res.json();
      renderCleaningResults(cleanRes);
      await refreshDatasets();
      document.getElementById("active-dataset-select").value = activeName;
      currentDataset = activeName;
    } catch (err) {
      alert("Cleaning error: " + err.message);
    } finally {
      btnClean.disabled = false;
      btnClean.textContent = "Clean & Validate";
    }
  });

  document.getElementById("toggle-view-raw").addEventListener("click", () => {
    renderTable(rawDatasetData);
    document.getElementById("toggle-view-raw").classList.add("btn-primary");
    document.getElementById("toggle-view-clean").classList.remove("btn-primary");
  });

  document.getElementById("toggle-view-clean").addEventListener("click", () => {
    renderTable(cleanDatasetData.length > 0 ? cleanDatasetData : rawDatasetData);
    document.getElementById("toggle-view-clean").classList.add("btn-primary");
    document.getElementById("toggle-view-raw").classList.remove("btn-primary");
  });
}

async function refreshDatasets() {
  try {
    const res = await fetch(`${API_BASE}/api/datasets`);
    const data = await res.json();
    const select = document.getElementById("active-dataset-select");
    const chatSelect = document.getElementById("chat-dataset-select");

    select.innerHTML = "";
    chatSelect.innerHTML = "";

    const rawDatasets = data.datasets.filter(d => !d.name.endsWith("_clean") && !d.name.endsWith("_raw"));
    const allDatasets = data.datasets;

    rawDatasets.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.name;
      opt.textContent = `${d.name} (${d.rows} rows)`;
      select.appendChild(opt);
    });

    allDatasets.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.name;
      opt.textContent = `${d.name} (${d.rows} rows)`;
      chatSelect.appendChild(opt);
    });

    document.getElementById("datasets-count-badge").textContent = `Datasets: ${rawDatasets.length}`;

    if (!currentDataset && rawDatasets.length > 0) {
      selectDataset(rawDatasets[0].name);
    }
  } catch (err) {
    console.error("Failed to refresh datasets", err);
  }
}

async function selectDataset(name) {
  currentDataset = name;
  document.getElementById("active-dataset-select").value = name;

  // Reset cleaning plan display to neutral state for the selected dataset
  const planList = document.getElementById("plan-list");
  planList.innerHTML = `<div class="empty-state">Click "Clean & Validate" to generate and execute an auditable cleaning plan.</div>`;
  document.getElementById("steps-count-badge").textContent = "0 Steps";
  document.getElementById("profile-idempotent").textContent = "Pending Run";

  // Profile dataset
  try {
    const formData = new FormData();
    formData.append("dataset_name", name);
    const res = await fetch(`${API_BASE}/api/pipeline/profile`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) return;
    const data = await res.json();
    renderProfileSnapshot(data.profile, data.issues);

    // Populate and display raw preview immediately
    rawDatasetData = data.preview || [];
    cleanDatasetData = [];
    renderTable(rawDatasetData);
    document.getElementById("toggle-view-raw").classList.add("btn-primary");
    document.getElementById("toggle-view-clean").classList.remove("btn-primary");
  } catch (err) {
    console.error(err);
  }
}

function renderProfileSnapshot(profile, issues) {
  document.getElementById("profile-rows").textContent = profile.row_count;
  document.getElementById("profile-cols").textContent = profile.column_count;
  document.getElementById("profile-dups").textContent = profile.duplicate_row_count;
  document.getElementById("profile-idempotent").textContent = "Pending Run";

  document.getElementById("issues-count-badge").textContent = `${issues.length} Issues`;
  const issuesList = document.getElementById("issues-list");
  issuesList.innerHTML = "";

  if (issues.length === 0) {
    issuesList.innerHTML = `<div class="empty-state">No quality issues detected! Data is clean.</div>`;
  } else {
    issues.forEach(iss => {
      const item = document.createElement("div");
      item.className = "list-item";
      const badgeClass = iss.severity === "high" ? "badge-danger" : iss.severity === "medium" ? "badge-warning" : "badge-info";
      item.innerHTML = `
        <div class="list-item-header">
          <span class="list-item-title">${iss.field || "Dataset"}</span>
          <span class="badge ${badgeClass}">${iss.severity.toUpperCase()}</span>
        </div>
        <div class="list-item-meta">${iss.description}</div>
      `;
      issuesList.appendChild(item);
    });
  }
}

function renderCleaningResults(cleanRes) {
  document.getElementById("profile-rows").textContent = `${cleanRes.validation.rows_before} → ${cleanRes.validation.rows_after}`;
  document.getElementById("profile-idempotent").textContent = cleanRes.validation.idempotent ? "100% Verified" : "Failed";

  // Render Plan
  const planList = document.getElementById("plan-list");
  planList.innerHTML = "";
  document.getElementById("steps-count-badge").textContent = `${cleanRes.cleaning_plan.length} Steps`;

  cleanRes.cleaning_plan.forEach(step => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <div class="list-item-header">
        <span class="list-item-title">${step.step_id}</span>
        <span class="badge badge-success">${step.status.toUpperCase()}</span>
      </div>
      <div class="list-item-meta">${step.reason}</div>
      <div class="list-item-meta" style="color:var(--text-muted); font-size:0.75rem;">Fields: ${step.affected_fields.join(", ")} | Risk: ${step.risk} | Source: ${step.source}</div>
    `;
    planList.appendChild(item);
  });

  // Store data previews and display clean data table
  rawDatasetData = cleanRes.raw_preview || [];
  cleanDatasetData = cleanRes.clean_preview || [];
  renderTable(cleanDatasetData.length > 0 ? cleanDatasetData : rawDatasetData);
  document.getElementById("toggle-view-clean").classList.add("btn-primary");
  document.getElementById("toggle-view-raw").classList.remove("btn-primary");
}

function renderTable(rows) {
  const container = document.getElementById("data-preview-table-wrap");
  if (!rows || rows.length === 0) {
    container.innerHTML = `<div class="empty-state">No records to display.</div>`;
    return;
  }
  const keys = Object.keys(rows[0]);
  let html = `<table><thead><tr>${keys.map(k => `<th>${escapeHtml(k)}</th>`).join("")}</tr></thead><tbody>`;
  rows.slice(0, 15).forEach(r => {
    html += `<tr>${keys.map(k => `<td>${r[k] !== null ? escapeHtml(r[k]) : ""}</td>`).join("")}</tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
}

// 4. Retail Intelligence Hub
function initRetailHub() {}

async function loadRetailHub() {
  try {
    const [metricsRes, scoresRes] = await Promise.all([
      fetch(`${API_BASE}/api/retail/metrics`),
      fetch(`${API_BASE}/api/retail/scores`),
    ]);

    const metricsData = await metricsRes.json();
    const scoresData = await scoresRes.json();

    // Render KPIs
    const s = metricsData.sales_kpis;
    if (s && s.gross_revenue) {
      document.getElementById("kpi-gross-rev").textContent = `$${s.gross_revenue.value.toLocaleString()}`;
      document.getElementById("kpi-gross-rev-cite").textContent = s.gross_revenue.citation;
      document.getElementById("kpi-net-rev").textContent = `$${s.net_revenue.value.toLocaleString()}`;
      document.getElementById("kpi-net-rev-cite").textContent = s.net_revenue.citation;
      document.getElementById("kpi-aov").textContent = `$${s.aov.value.toLocaleString()}`;
      document.getElementById("kpi-aov-cite").textContent = s.aov.citation;
      document.getElementById("kpi-return-rate").textContent = `${s.return_rate_pct.value}%`;
      document.getElementById("kpi-return-rate-cite").textContent = s.return_rate_pct.citation;
    }

    // Render Product Health Scores
    const prodList = document.getElementById("product-scores-list");
    prodList.innerHTML = "";
    if (scoresData.product_health_scores.length === 0) {
      prodList.innerHTML = `<div class="empty-state">No products loaded.</div>`;
    } else {
      scoresData.product_health_scores.forEach(p => {
        const item = document.createElement("div");
        item.className = "list-item";
        const badgeColor = p.health_score >= 80 ? "badge-success" : p.health_score >= 50 ? "badge-warning" : "badge-danger";
        const factors = p.contributing_factors.map(f => `<span class="op-badge">${f.factor}: ${f.points > 0 ? "+" : ""}${f.points}pts</span>`).join(" ");
        const missing = p.missing_evidence.length > 0 ? `<div style="color:var(--accent-warning); font-size:0.75rem; margin-top:4px;">Missing: ${p.missing_evidence.join("; ")}</div>` : "";

        item.innerHTML = `
          <div class="list-item-header">
            <span class="list-item-title">${p.product_name} (${p.product_id})</span>
            <span class="badge ${badgeColor}">Score: ${p.health_score}/100</span>
          </div>
          <div class="list-item-meta"><strong>Recommendation:</strong> ${p.recommendation}</div>
          <div style="margin-top:6px;">${factors}</div>
          ${missing}
        `;
        prodList.appendChild(item);
      });
    }

    // Render RFM Segments
    const rfmList = document.getElementById("customer-rfm-list");
    rfmList.innerHTML = "";
    if (scoresData.customer_rfm_segments.length === 0) {
      rfmList.innerHTML = `<div class="empty-state">No customer order data loaded.</div>`;
    } else {
      scoresData.customer_rfm_segments.forEach(c => {
        const item = document.createElement("div");
        item.className = "list-item";
        item.innerHTML = `
          <div class="list-item-header">
            <span class="list-item-title">${c.customer_id}</span>
            <span class="badge badge-purple">${c.segment}</span>
          </div>
          <div class="list-item-meta">${c.audit_evidence}</div>
        `;
        rfmList.appendChild(item);
      });
    }
  } catch (err) {
    console.error("Failed to load retail hub", err);
  }
}

// 5. Conversational Assistant
function initChat() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;
    input.value = "";
    sendChatMessage(query);
  });

  document.querySelectorAll(".quick-chips .chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      sendChatMessage(q);
    });
  });
}

async function sendChatMessage(question) {
  const chatMessages = document.getElementById("chat-messages");
  const targetDataset = document.getElementById("chat-dataset-select").value;

  // Render User Bubble
  const userMsg = document.createElement("div");
  userMsg.className = "message user-message";
  userMsg.innerHTML = `<div class="msg-bubble">${escapeHtml(question)}</div>`;
  chatMessages.appendChild(userMsg);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Loading indicator
  const loadingMsg = document.createElement("div");
  loadingMsg.className = "message assistant-message";
  loadingMsg.innerHTML = `<div class="msg-bubble">Computing verifiable answer...</div>`;
  chatMessages.appendChild(loadingMsg);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        session_id: sessionId,
        dataset: targetDataset,
      }),
    });

    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    loadingMsg.remove();

    // Render Assistant Bubble
    const asstMsg = document.createElement("div");
    asstMsg.className = "message assistant-message";
    const statusBadge = data.status === "ok" ? "badge-success" : data.status === "refuse" ? "badge-warning" : "badge-info";
    asstMsg.innerHTML = `
      <div class="msg-bubble">
        <div style="margin-bottom:6px;"><span class="badge ${statusBadge}">${data.status.toUpperCase()}</span></div>
        <div>${data.answer.summary || data.answer.error}</div>
      </div>
    `;
    chatMessages.appendChild(asstMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Render Evidence Drawer
    renderEvidence(data);
  } catch (err) {
    loadingMsg.innerHTML = `<div class="msg-bubble text-danger">Error: ${escapeHtml(err.message)}</div>`;
  }
}

function renderEvidence(data) {
  const container = document.getElementById("evidence-body");
  const ev = data.evidence;

  if (!ev) {
    container.innerHTML = `<div class="empty-state">No execution evidence for this turn (${data.status}).</div>`;
    return;
  }

  const opsBadges = ev.operations.map(op => `<span class="op-badge">${escapeHtml(op)}</span>`).join("<br>");

  let tableHtml = "";
  if (ev.preview && ev.preview.length > 0) {
    const cols = Object.keys(ev.preview[0]);
    tableHtml = `
      <div class="table-responsive" style="margin-top:8px;">
        <table>
          <thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
          <tbody>
            ${ev.preview.map(row => `<tr>${cols.map(c => `<td>${row[c] !== null ? row[c] : ""}</td>`).join("")}</tr>`).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  container.innerHTML = `
    <div class="evidence-card-section">
      <h3>Target Dataset & Scope</h3>
      <div><strong>Version:</strong> ${ev.dataset_version}</div>
      <div><strong>Rows Considered:</strong> ${ev.row_count_considered}</div>
      <div><strong>Columns Used:</strong> ${ev.columns_used.join(", ")}</div>
    </div>

    <div class="evidence-card-section">
      <h3>Execution Pipeline Trace</h3>
      <div>${opsBadges}</div>
    </div>

    <div class="evidence-card-section">
      <h3>Bounded Computation Preview (${ev.rows_returned} rows)</h3>
      ${tableHtml || "<div class='empty-state'>Scalar metric evaluated</div>"}
    </div>
  `;
}

// 6. Batch Evaluator
function initEvalRunner() {
  const btn = document.getElementById("btn-run-eval");
  if (btn) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "Executing Batch...";
      const wrap = document.getElementById("eval-results-wrap");
      wrap.innerHTML = `<div class="loading-state" style="padding:24px; text-align:center;">⏳ Running evaluation batch across cases.json...</div>`;

      try {
        const res = await fetch(`${API_BASE}/api/evaluate/run`, { method: "POST" });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        renderEvaluationReport(data.results || data);
      } catch (err) {
        wrap.innerHTML = `<div class="text-danger" style="padding:16px;">Evaluation batch error: ${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Run Evaluation Batch";
      }
    });
  }
}

async function loadEvaluationResults() {
  const wrap = document.getElementById("eval-results-wrap");
  if (!wrap) return;
  try {
    const res = await fetch(`${API_BASE}/api/evaluate/results`);
    if (res.ok) {
      const data = await res.json();
      renderEvaluationReport(data);
    }
  } catch (err) {
    console.error("Could not fetch evaluation results", err);
  }
}

function renderEvaluationReport(results) {
  const wrap = document.getElementById("eval-results-wrap");
  if (!wrap) return;
  if (!results) {
    wrap.innerHTML = `<div class="empty-state">No evaluation results loaded yet. Click "Run Evaluation Batch".</div>`;
    return;
  }
  const caseList = Array.isArray(results) ? results : [results];

  let html = `<div class="eval-cases-container" style="display:flex; flex-direction:column; gap:20px; margin-top:16px;">`;

  caseList.forEach((c, idx) => {
    const val = c.validation || {};
    const chatTurns = c.chat_evaluation || [];
    const issues = c.issues || [];
    const steps = c.cleaning_plan || [];
    const idempotentBadge = val.idempotent ? `<span class="badge badge-success">Idempotent: Verified</span>` : `<span class="badge badge-danger">Idempotent: Failed</span>`;

    html += `
      <div class="card" style="border:1px solid rgba(255,255,255,0.08); background:rgba(255,255,255,0.02); border-radius:10px; padding:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
          <div>
            <h3 style="margin:0; font-size:1.15rem; color:#f1f5f9; font-weight:600;">Case ${idx + 1}: ${escapeHtml(c.case_id)}</h3>
            <span style="font-size:0.8rem; color:#94a3b8;">Dataset: <strong>${escapeHtml(c.dataset?.name || "unknown")}</strong> | Rows: ${val.rows_before ?? "-"} → ${val.rows_after ?? "-"} (Delta: ${val.row_delta ?? 0})</span>
          </div>
          <div style="display:flex; gap:8px;">
            ${idempotentBadge}
            <span class="badge badge-info">${chatTurns.length} Queries Evaluated</span>
          </div>
        </div>

        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px; margin-bottom:16px; font-size:0.85rem;">
          <div style="background:rgba(0,0,0,0.25); padding:12px; border-radius:6px; border:1px solid rgba(255,255,255,0.05);">
            <strong style="color:#cbd5e1;">Detected Issues (${issues.length}):</strong>
            <ul style="margin:6px 0 0 16px; padding:0; color:#94a3b8;">
              ${issues.slice(0, 4).map(iss => `<li>[${iss.severity.toUpperCase()}] ${escapeHtml(iss.description)}</li>`).join("") || "<li>No issues</li>"}
            </ul>
          </div>
          <div style="background:rgba(0,0,0,0.25); padding:12px; border-radius:6px; border:1px solid rgba(255,255,255,0.05);">
            <strong style="color:#cbd5e1;">Executed Cleaning Steps (${steps.length}):</strong>
            <ul style="margin:6px 0 0 16px; padding:0; color:#94a3b8;">
              ${steps.slice(0, 4).map(s => `<li>${escapeHtml(s.step_id)}: ${escapeHtml(s.reason)}</li>`).join("") || "<li>No steps required</li>"}
            </ul>
          </div>
        </div>

        <div>
          <h4 style="margin:0 0 10px 0; font-size:0.95rem; color:#cbd5e1;">Conversational Verification Results:</h4>
          <div style="display:flex; flex-direction:column; gap:8px;">
            ${chatTurns.map(t => {
              const statusBadge = t.status === "ok" ? `<span class="badge badge-success">OK</span>` : t.status === "refuse" ? `<span class="badge badge-danger">REFUSE</span>` : `<span class="badge badge-warning">CLARIFY</span>`;
              return `
                <div style="background:rgba(0,0,0,0.2); padding:10px 14px; border-radius:6px; font-size:0.85rem; border-left: 3px solid ${t.status === 'ok' ? '#10b981' : t.status === 'refuse' ? '#ef4444' : '#f59e0b'};">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:#f1f5f9;">Q: ${escapeHtml(t.question)}</span>
                    ${statusBadge}
                  </div>
                  <div style="color:#cbd5e1; margin-top:4px;">
                    <strong>Answer:</strong> ${escapeHtml(t.answer?.summary || JSON.stringify(t.answer))}
                  </div>
                </div>
              `;
            }).join("")}
          </div>
        </div>
      </div>
    `;
  });

  html += `
    <details style="margin-top:12px; cursor:pointer;">
      <summary style="font-size:0.9rem; color:#60a5fa; font-weight:600;">View Raw Results JSON Contract Output</summary>
      <pre style="margin-top:8px; max-height:400px; overflow-y:auto; background:rgba(0,0,0,0.4); padding:14px; border-radius:6px; font-size:0.8rem; color:#94a3b8;">${escapeHtml(JSON.stringify(results, null, 2))}</pre>
    </details>
  </div>`;

  wrap.innerHTML = html;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
