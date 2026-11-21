// Frontend Application Logic for Street Litter & Person Detection

let currentSelectedFile = null;
let currentSelectedSample = null;
let currentDetectionData = null;
let currentImageMode = "annotated"; // 'annotated' | 'original' | 'split'

document.addEventListener("DOMContentLoaded", () => {
  initSystemInfo();
  initSampleImages();
  setupDropzone();
  setupSlider();
  setupForm();
  loadIncidents();
});

// System Hardware Info
async function initSystemInfo() {
  const badge = document.getElementById("deviceBadge");
  const text = document.getElementById("deviceText");
  try {
    const res = await fetch("/api/system_info");
    const data = await res.json();
    if (data.device === "CUDA") {
      badge.classList.add("cuda-active");
      text.textContent = `${data.device_name} (${data.vram || "CUDA"})`;
    } else {
      text.textContent = "CPU Fallback Mode";
    }
    updateIncidentsBadge(data.incidents_count || 0);
  } catch (err) {
    text.textContent = "Offline";
  }
}

// Sample Images
async function initSampleImages() {
  const container = document.getElementById("sampleChips");
  try {
    const res = await fetch("/api/sample_images");
    const data = await res.json();
    if (data.samples && data.samples.length > 0) {
      container.innerHTML = "";
      data.samples.forEach((filename) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "sample-chip";
        chip.innerHTML = `<i class="fa-regular fa-image"></i> <span>${filename}</span>`;
        chip.onclick = () => selectSampleImage(filename, chip);
        container.appendChild(chip);
      });
    } else {
      container.innerHTML = '<span class="empty-tag">No test images found</span>';
    }
  } catch (err) {
    container.innerHTML = '<span class="empty-tag">Error loading samples</span>';
  }
}

function selectSampleImage(filename, chipElement) {
  // Clear any file selection
  clearSelectedFile(false);

  // Toggle active chip
  document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));
  chipElement.classList.add("active");
  currentSelectedSample = filename;

  // Show preview bar
  const previewBar = document.getElementById("selectedFilePreview");
  const thumb = document.getElementById("previewThumbnail");
  const nameEl = document.getElementById("selectedFileName");
  const sizeEl = document.getElementById("selectedFileSize");

  thumb.src = `/test_images/${encodeURIComponent(filename)}`;
  nameEl.textContent = filename;
  sizeEl.textContent = "Sample Image";
  previewBar.style.display = "flex";
  document.getElementById("dropzonePrompt").style.display = "none";

  document.getElementById("analyzeBtn").disabled = false;
}

// File Drag & Drop
function setupDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });
}

function handleFile(file) {
  const validTypes = ["image/jpeg", "image/png", "image/webp", "image/avif"];
  const validExts = [".jpg", ".jpeg", ".png", ".webp", ".avif"];
  const ext = "." + file.name.split(".").pop().toLowerCase();

  if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
    alert("Please upload an image file (JPG, JPEG, PNG, WEBP).");
    return;
  }

  // Clear sample chip selection
  currentSelectedSample = null;
  document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));

  currentSelectedFile = file;

  const previewBar = document.getElementById("selectedFilePreview");
  const thumb = document.getElementById("previewThumbnail");
  const nameEl = document.getElementById("selectedFileName");
  const sizeEl = document.getElementById("selectedFileSize");

  nameEl.textContent = file.name;
  sizeEl.textContent = formatBytes(file.size);

  const reader = new FileReader();
  reader.onload = (e) => {
    thumb.src = e.target.result;
    previewBar.style.display = "flex";
    document.getElementById("dropzonePrompt").style.display = "none";
  };
  reader.readAsDataURL(file);

  document.getElementById("analyzeBtn").disabled = false;
}

function clearSelectedFile(resetChips = true) {
  currentSelectedFile = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("selectedFilePreview").style.display = "none";
  document.getElementById("dropzonePrompt").style.display = "block";

  if (resetChips) {
    currentSelectedSample = null;
    document.querySelectorAll(".sample-chip").forEach((c) => c.classList.remove("active"));
    document.getElementById("analyzeBtn").disabled = true;
  }
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

// Slider
function setupSlider() {
  const slider = document.getElementById("confSlider");
  const val = document.getElementById("confValue");
  slider.addEventListener("input", (e) => {
    val.textContent = parseFloat(e.target.value).toFixed(2);
  });
}

// Form Submission
function setupForm() {
  const form = document.getElementById("uploadForm");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!currentSelectedFile && !currentSelectedSample) return;

    const analyzeBtn = document.getElementById("analyzeBtn");
    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultsContent = document.getElementById("resultsContent");

    // UI state
    analyzeBtn.disabled = true;
    emptyState.style.display = "none";
    resultsContent.style.display = "none";
    loadingState.style.display = "block";

    const formData = new FormData();
    if (currentSelectedFile) {
      formData.append("image", currentSelectedFile);
    } else if (currentSelectedSample) {
      formData.append("sample_image", currentSelectedSample);
    }
    formData.append("conf", document.getElementById("confSlider").value);

    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        body: formData,
      });

      const json = await res.json();
      if (!res.ok || json.error) {
        throw new Error(json.error || "Detection failed.");
      }

      currentDetectionData = json.data;
      renderResults(json.data);
    } catch (err) {
      alert("Error: " + err.message);
      emptyState.style.display = "block";
    } finally {
      loadingState.style.display = "none";
      analyzeBtn.disabled = false;
    }
  });
}

// Render Results
function renderResults(data) {
  const resultsContent = document.getElementById("resultsContent");
  const decisionBanner = document.getElementById("decisionBanner");
  const decisionIcon = document.getElementById("decisionIcon");
  const decisionTitle = document.getElementById("decisionTitle");
  const decisionDesc = document.getElementById("decisionDesc");
  const decisionBadge = document.getElementById("decisionBadge");
  const incidentSavedNotice = document.getElementById("incidentSavedNotice");

  // Reset classes
  decisionBanner.className = "decision-banner";

  const code = data.decision.code;
  const personCount = data.summary.person_count;
  const litterCount = data.summary.litter_count;

  if (code === "POTENTIAL_INCIDENT") {
    decisionBanner.classList.add("potential-incident");
    decisionIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
    decisionTitle.textContent = "Potential Littering Incident";
    decisionDesc.textContent = `Both person (${personCount}) and litter item${litterCount > 1 ? "s" : ""} (${litterCount}) detected in photograph. Flagged for review.`;
    decisionBadge.textContent = "Pending Verification";
    decisionBadge.style.display = "block";
    incidentSavedNotice.style.display = "flex";
    loadIncidents(); // Refresh log
  } else if (code === "LITTER_ONLY") {
    decisionBanner.classList.add("litter-only");
    decisionIcon.innerHTML = '<i class="fa-solid fa-bottle-water"></i>';
    decisionTitle.textContent = "Litter-Only Detection";
    decisionDesc.textContent = `${litterCount} street litter item${litterCount > 1 ? "s" : ""} detected. No people detected.`;
    decisionBadge.textContent = "Litter Present";
    decisionBadge.style.display = "block";
    incidentSavedNotice.style.display = "none";
  } else if (code === "PERSON_ONLY") {
    decisionBanner.classList.add("person-only");
    decisionIcon.innerHTML = '<i class="fa-solid fa-person"></i>';
    decisionTitle.textContent = "Person-Only Detection";
    decisionDesc.textContent = `${personCount} person detected. No street litter detected.`;
    decisionBadge.textContent = "Clear of Litter";
    decisionBadge.style.display = "block";
    incidentSavedNotice.style.display = "none";
  } else {
    decisionBanner.classList.add("no-detections");
    decisionIcon.innerHTML = '<i class="fa-regular fa-circle-check"></i>';
    decisionTitle.textContent = "No Detections";
    decisionDesc.textContent = "Neither person nor litter items were detected above the confidence threshold.";
    decisionBadge.textContent = "Clear";
    decisionBadge.style.display = "block";
    incidentSavedNotice.style.display = "none";
  }

  // Timestamp
  document.getElementById("resultsTimestamp").textContent = data.timestamp;

  // Images
  document.getElementById("mainDisplayImg").src = data.annotated_image_url;
  document.getElementById("splitOriginalImg").src = data.original_image_url;
  document.getElementById("splitAnnotatedImg").src = data.annotated_image_url;
  document.getElementById("downloadAnnotatedLink").href = data.annotated_image_url;

  // Metrics
  document.getElementById("metricPersonCount").textContent = personCount;
  document.getElementById("metricLitterCount").textContent = litterCount;

  const categoriesContainer = document.getElementById("metricCategoriesList");
  categoriesContainer.innerHTML = "";
  if (data.summary.litter_categories && data.summary.litter_categories.length > 0) {
    data.summary.litter_categories.forEach((cat) => {
      const tag = document.createElement("span");
      tag.className = "cat-tag";
      tag.textContent = cat;
      categoriesContainer.appendChild(tag);
    });
  } else {
    categoriesContainer.innerHTML = '<span class="empty-tag">No litter categories</span>';
  }

  // Detections Table
  const tableBody = document.getElementById("detectionsTableBody");
  tableBody.innerHTML = "";
  let rowIdx = 1;

  // Add people detections
  data.detections.people.forEach((det) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${rowIdx++}</td>
      <td><span class="type-pill type-person">Person</span></td>
      <td><strong>Person</strong></td>
      <td>${(det.confidence * 100).toFixed(1)}%</td>
      <td><code>[${det.bbox.join(", ")}]</code></td>
    `;
    tableBody.appendChild(tr);
  });

  // Add litter detections
  data.detections.litter.forEach((det) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${rowIdx++}</td>
      <td><span class="type-pill type-litter">Litter</span></td>
      <td><strong>${det.label}</strong></td>
      <td>${(det.confidence * 100).toFixed(1)}%</td>
      <td><code>[${det.bbox.join(", ")}]</code></td>
    `;
    tableBody.appendChild(tr);
  });

  if (rowIdx === 1) {
    tableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No detections to display.</td></tr>';
  }

  // Reset to default annotated image mode
  setImageMode("annotated");

  resultsContent.style.display = "block";
}

// Image View Modes
function setImageMode(mode) {
  currentImageMode = mode;
  const singleViewer = document.getElementById("singleImageViewer");
  const splitViewer = document.getElementById("splitImageViewer");
  const mainImg = document.getElementById("mainDisplayImg");

  document.querySelectorAll(".image-mode-toggles .btn-toggle").forEach((btn) => btn.classList.remove("active"));

  if (!currentDetectionData) return;

  if (mode === "annotated") {
    document.getElementById("btnShowAnnotated").classList.add("active");
    mainImg.src = currentDetectionData.annotated_image_url;
    singleViewer.style.display = "flex";
    splitViewer.style.display = "none";
  } else if (mode === "original") {
    document.getElementById("btnShowOriginal").classList.add("active");
    mainImg.src = currentDetectionData.original_image_url;
    singleViewer.style.display = "flex";
    splitViewer.style.display = "none";
  } else if (mode === "split") {
    document.getElementById("btnShowSplit").classList.add("active");
    singleViewer.style.display = "none";
    splitViewer.style.display = "grid";
  }
}

// Tab Switching
function switchTab(tabId) {
  const detectSec = document.getElementById("detectSection");
  const incidentsSec = document.getElementById("incidentsSection");
  const tabDetectBtn = document.getElementById("tabDetectBtn");
  const tabIncidentsBtn = document.getElementById("tabIncidentsBtn");

  if (tabId === "detect") {
    detectSec.style.display = "block";
    incidentsSec.style.display = "none";
    tabDetectBtn.classList.add("active");
    tabIncidentsBtn.classList.remove("active");
  } else {
    detectSec.style.display = "none";
    incidentsSec.style.display = "block";
    tabDetectBtn.classList.remove("active");
    tabIncidentsBtn.classList.add("active");
    loadIncidents();
  }
}

// Incidents Log Handling
async function loadIncidents() {
  const grid = document.getElementById("incidentsGrid");
  const emptyState = document.getElementById("incidentsEmptyState");

  try {
    const res = await fetch("/api/incidents");
    const data = await res.json();
    const incidents = data.incidents || [];
    updateIncidentsBadge(incidents.length);

    if (incidents.length === 0) {
      grid.innerHTML = "";
      emptyState.style.display = "block";
      return;
    }

    emptyState.style.display = "none";
    grid.innerHTML = "";

    incidents.forEach((inc) => {
      const card = document.createElement("div");
      card.className = "incident-card";

      let statusClass = "";
      if (inc.incident_status === "Verified Incident") statusClass = "verified";
      else if (inc.incident_status.includes("Dismissed")) statusClass = "dismissed";

      const categoriesBadges = (inc.detected_litter_categories || [])
        .map((cat) => `<span class="cat-tag">${cat}</span>`)
        .join(" ");

      card.innerHTML = `
        <div class="incident-card-header">
          <span class="incident-id">${inc.incident_id}</span>
          <span class="incident-status-tag ${statusClass}">${inc.incident_status}</span>
        </div>
        <div class="incident-card-images">
          <img src="/incidents/${inc.original_image_path}" alt="Original" title="Original" onclick="window.open(this.src)" />
          <img src="/incidents/${inc.annotated_image_path}" alt="Annotated" title="Annotated (Red: Person, Green: Litter)" onclick="window.open(this.src)" />
        </div>
        <div class="incident-card-body">
          <div class="incident-meta-row">
            <span><i class="fa-regular fa-clock"></i> ${inc.timestamp}</span>
            <span><strong>${inc.person_detection_status}</strong></span>
          </div>
          <div class="incident-meta-row">
            <span><strong>Litter items:</strong> ${inc.litter_count}</span>
          </div>
          <div class="category-tags" style="margin-top: 0.5rem;">
            ${categoriesBadges || '<span class="empty-tag">None</span>'}
          </div>
          ${inc.review_notes ? `<div class="incident-notes-box"><strong>Reviewer Notes:</strong> ${inc.review_notes}</div>` : ""}
        </div>
        <div class="incident-card-footer">
          <button class="btn btn-sm btn-secondary" onclick="openReviewModal('${inc.incident_id}', '${inc.incident_status}', \`${(inc.review_notes || '').replace(/`/g, '\\`')}\`)">
            <i class="fa-solid fa-pen-to-square"></i> Review Incident
          </button>
        </div>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    grid.innerHTML = `<p class="empty-tag">Failed to load incidents log: ${err.message}</p>`;
  }
}

function updateIncidentsBadge(count) {
  document.getElementById("incidentsCountBadge").textContent = count;
}

// Modal Review Actions
function openReviewModal(incidentId, status, notes) {
  document.getElementById("modalIncidentId").value = incidentId;
  document.getElementById("modalStatusSelect").value = status;
  document.getElementById("modalNotesText").value = notes;
  document.getElementById("reviewModal").style.display = "flex";
}

function closeReviewModal() {
  document.getElementById("reviewModal").style.display = "none";
}

async function submitReviewUpdate() {
  const id = document.getElementById("modalIncidentId").value;
  const status = document.getElementById("modalStatusSelect").value;
  const notes = document.getElementById("modalNotesText").value;

  try {
    const res = await fetch(`/api/incidents/${id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, notes }),
    });
    if (!res.ok) throw new Error("Could not update review.");
    closeReviewModal();
    loadIncidents();
  } catch (err) {
    alert("Error saving review: " + err.message);
  }
}
