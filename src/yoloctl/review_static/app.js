(function () {
  const body = document.body;
  const VIEWER_SPLIT_STORAGE_KEY = "yoloctl-review-viewer-split-ratio";
  const MIN_VIEWER_HEIGHT = 260;
  const MIN_FOCUS_HEIGHT = 220;
  const state = {
    classNames: JSON.parse(body.dataset.classNames || "[]"),
    statuses: JSON.parse(body.dataset.statuses || "[]"),
    splits: JSON.parse(body.dataset.splits || "[]"),
    items: [],
    queueBatchSize: 80,
    queueRenderCount: 0,
    currentImageId: null,
    current: null,
    selectedDraftId: null,
    highlightedPredictionId: null,
    selectedIssueId: null,
    selectedDiagnosticLayer: "combined",
    activeInspectorTab: "diagnostic",
    activeObjectTab: "draft",
    dirty: false,
    mode: "select",
    zoom: 1,
    panX: 0,
    panY: 0,
    pointerAction: null,
    drawPreview: null,
    liveDraftDiagnostics: { summary: emptyIssueSummary(), items: [] },
    viewerSplitRatio: 0.6,
    layoutResize: null,
  };

  const STATUS_LABELS = {
    todo: "待审核",
    reviewed_ok: "确认无误",
    fixed: "已修正",
    needs_followup: "需复查",
  };

  const SPLIT_LABELS = {
    train: "训练集",
    val: "验证集",
    test: "测试集",
  };

  const ISSUE_LABELS = {
    cls: "类别混淆",
    fn: "漏检",
    fp: "误检",
    no_issue: "无问题",
  };

  const els = {
    sessionSummary: document.getElementById("session-summary"),
    queueMeta: document.getElementById("queue-meta"),
    queueList: document.getElementById("queue-list"),
    centerPanel: document.querySelector(".center-panel"),
    viewerShell: document.querySelector(".viewer-shell"),
    focusPanel: document.querySelector(".focus-panel"),
    workspaceResizer: document.getElementById("workspace-resizer"),
    workspaceResizerGrip: document.getElementById("workspace-resizer-grip"),
    itemHeader: document.getElementById("item-header"),
    itemProgress: document.getElementById("item-progress"),
    viewerEmpty: document.getElementById("viewer-empty"),
    viewerStage: document.getElementById("viewer-stage"),
    stageTransform: document.getElementById("stage-transform"),
    mainImage: document.getElementById("main-image"),
    overlay: document.getElementById("overlay"),
    focusEmpty: document.getElementById("focus-empty"),
    focusStage: document.getElementById("focus-stage"),
    focusLens: document.getElementById("focus-lens"),
    focusViewport: document.getElementById("focus-viewport"),
    focusImage: document.getElementById("focus-image"),
    focusOverlay: document.getElementById("focus-overlay"),
    issueChipRow: document.getElementById("issue-chip-row"),
    selectedIssueSummary: document.getElementById("selected-issue-summary"),
    selectedIssueEvidence: document.getElementById("selected-issue-evidence"),
    baselineState: document.getElementById("baseline-state"),
    draftState: document.getElementById("draft-state"),
    baselineSummaryCard: document.getElementById("baseline-summary-card"),
    draftSummaryCard: document.getElementById("draft-summary-card"),
    selectedIssueDetail: document.getElementById("selected-issue-detail"),
    issueActions: document.getElementById("issue-actions"),
    issueList: document.getElementById("issue-list"),
    itemStatus: document.getElementById("item-status"),
    itemNote: document.getElementById("item-note"),
    loadPredictions: document.getElementById("load-predictions"),
    useAllPredictions: document.getElementById("use-all-predictions"),
    saveItem: document.getElementById("save-item"),
    dirtyIndicator: document.getElementById("dirty-indicator"),
    selectedDraftEmpty: document.getElementById("selected-draft-empty"),
    selectedDraftPanel: document.getElementById("selected-draft-panel"),
    selectedDraftClass: document.getElementById("selected-draft-class"),
    draftList: document.getElementById("draft-list"),
    predictionList: document.getElementById("prediction-list"),
    filterSplit: document.getElementById("filter-split"),
    filterStatus: document.getElementById("filter-status"),
    filterIssue: document.getElementById("filter-issue"),
    filterGtClass: document.getElementById("filter-gt-class"),
    filterPredClass: document.getElementById("filter-pred-class"),
    filterEditedOnly: document.getElementById("filter-edited-only"),
    filterQuery: document.getElementById("filter-query"),
    filterOrder: document.getElementById("filter-order"),
    modeSelect: document.getElementById("mode-select"),
    modeDraw: document.getElementById("mode-draw"),
    modePan: document.getElementById("mode-pan"),
    deleteObject: document.getElementById("delete-object"),
    zoomIn: document.getElementById("zoom-in"),
    zoomOut: document.getElementById("zoom-out"),
    zoomReset: document.getElementById("zoom-reset"),
    toggleGt: document.getElementById("toggle-gt"),
    togglePred: document.getElementById("toggle-pred"),
    toggleDraft: document.getElementById("toggle-draft"),
    prevItem: document.getElementById("prev-item"),
    nextItem: document.getElementById("next-item"),
    tabDiagnostic: document.getElementById("tab-diagnostic"),
    tabEdit: document.getElementById("tab-edit"),
    inspectorDiagnostic: document.getElementById("inspector-diagnostic"),
    inspectorEdit: document.getElementById("inspector-edit"),
    tabDraftObjects: document.getElementById("tab-draft-objects"),
    tabPredictionObjects: document.getElementById("tab-prediction-objects"),
    layerButtons: Array.from(document.querySelectorAll(".layer-btn")),
  };

  function emptyIssueSummary() {
    return { fp: 0, fn: 0, cls: 0, severity: 0, types: ["no_issue"] };
  }

  function loadViewerSplitRatio() {
    try {
      const raw = window.localStorage.getItem(VIEWER_SPLIT_STORAGE_KEY);
      const ratio = Number(raw);
      if (!Number.isFinite(ratio)) return 0.6;
      return Math.max(0.35, Math.min(0.78, ratio));
    } catch (error) {
      return 0.6;
    }
  }

  function saveViewerSplitRatio() {
    try {
      window.localStorage.setItem(VIEWER_SPLIT_STORAGE_KEY, String(state.viewerSplitRatio));
    } catch (error) {
      // localStorage is best-effort only.
    }
  }

  function svgEl(name, attrs) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs || {}).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  async function fetchJSON(url, options) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({ detail: "Request failed" }));
      throw new Error(payload.detail || "Request failed");
    }
    return response.json();
  }

  function setDirty(value) {
    state.dirty = Boolean(value);
    els.dirtyIndicator.textContent = state.dirty ? "有未保存修改" : "已保存";
    els.dirtyIndicator.classList.toggle("dirty", state.dirty);
  }

  function statusLabel(value) {
    return STATUS_LABELS[value] || value;
  }

  function splitLabel(value) {
    return SPLIT_LABELS[value] || value;
  }

  function issueLabel(value) {
    return ISSUE_LABELS[value] || value;
  }

  function issueCounts(summary) {
    return `混淆 ${summary.cls || 0} · 漏检 ${summary.fn || 0} · 误检 ${summary.fp || 0}`;
  }

  function showGtLayer() {
    return Boolean(els.toggleGt.checked);
  }

  function showPredLayer() {
    return Boolean(els.togglePred.checked);
  }

  function showDraftLayer() {
    return Boolean(els.toggleDraft.checked);
  }

  function populateSelect(select, values, includeBlank, formatter) {
    const previous = select.value;
    select.innerHTML = "";
    if (includeBlank) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "全部";
      select.appendChild(option);
    }
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = formatter ? formatter(value) : value;
      select.appendChild(option);
    });
    if ([...select.options].some((option) => option.value === previous)) {
      select.value = previous;
    }
  }

  function currentFilters() {
    return {
      split: els.filterSplit.value,
      status: els.filterStatus.value,
      issueType: els.filterIssue.value,
      gtClass: els.filterGtClass.value,
      predClass: els.filterPredClass.value,
      editedOnly: els.filterEditedOnly.checked,
      query: els.filterQuery.value,
      orderBy: els.filterOrder.value,
    };
  }

  function queryString(params) {
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value === "" || value === false || value == null) return;
      search.set(key, value);
    });
    return search.toString();
  }

  function updateToolbarModes() {
    [els.modeSelect, els.modeDraw, els.modePan].forEach((button) => button.classList.remove("active"));
    if (state.mode === "select") els.modeSelect.classList.add("active");
    if (state.mode === "draw") els.modeDraw.classList.add("active");
    if (state.mode === "pan") els.modePan.classList.add("active");
  }

  function updateInspectorTabs() {
    els.tabDiagnostic.classList.toggle("active", state.activeInspectorTab === "diagnostic");
    els.tabEdit.classList.toggle("active", state.activeInspectorTab === "edit");
    els.inspectorDiagnostic.classList.toggle("hidden", state.activeInspectorTab !== "diagnostic");
    els.inspectorEdit.classList.toggle("hidden", state.activeInspectorTab !== "edit");
  }

  function updateObjectTabs() {
    els.tabDraftObjects.classList.toggle("active", state.activeObjectTab === "draft");
    els.tabPredictionObjects.classList.toggle("active", state.activeObjectTab === "prediction");
    els.draftList.classList.toggle("hidden", state.activeObjectTab !== "draft");
    els.predictionList.classList.toggle("hidden", state.activeObjectTab !== "prediction");
  }

  function updateLayerButtons() {
    els.layerButtons.forEach((button) => {
      button.classList.toggle("active", button.dataset.layer === state.selectedDiagnosticLayer);
    });
  }

  function currentItemIndex() {
    return state.items.findIndex((item) => item.image_id === state.currentImageId);
  }

  function ensureQueueRenderCount(index) {
    const safeIndex = typeof index === "number" && index >= 0 ? index : -1;
    const required = safeIndex >= 0 ? safeIndex + 1 : state.queueBatchSize;
    const batches = Math.max(1, Math.ceil(required / state.queueBatchSize));
    state.queueRenderCount = Math.min(state.items.length, batches * state.queueBatchSize);
  }

  function visibleQueueItems() {
    const count = Math.min(state.queueRenderCount || state.queueBatchSize, state.items.length);
    return state.items.slice(0, count);
  }

  function ensureSelectedIssue() {
    const issues = currentIssueCollection();
    if (!issues.length) {
      state.selectedIssueId = null;
      return;
    }
    if (issues.some((issue) => issue.id === state.selectedIssueId)) {
      return;
    }
    state.selectedIssueId = issues[0].id;
  }

  function currentIssueCollection() {
    if (!state.current) {
      return [];
    }
    const baseline = state.current.baseline_issue_items || [];
    if (baseline.length) {
      return baseline;
    }
    return state.liveDraftDiagnostics.items || [];
  }

  function selectedIssue() {
    return currentIssueCollection().find((issue) => issue.id === state.selectedIssueId) || null;
  }

  function selectedDraft() {
    if (!state.current) return null;
    return state.current.draft_objects.find((item) => item.id === state.selectedDraftId) || null;
  }

  function getDraftById(objectId) {
    return state.current?.draft_objects.find((item) => item.id === objectId) || null;
  }

  function getGtById(objectId) {
    return state.current?.gt_objects.find((item) => item.id === objectId) || null;
  }

  function getPredictionById(objectId) {
    return state.current?.prediction_objects.find((item) => item.id === objectId) || null;
  }

  function issueBadgesForObject(objectId, issueItems, key) {
    return issueItems.filter((issue) => issue[key] === objectId).map((issue) => issue.id);
  }

  function renderIssueBadges(group, box, issueIds, fill) {
    if (!issueIds.length) return;
    issueIds.forEach((issueId, index) => {
      const x = box.x1 + index * 22;
      const badge = svgEl("g", { class: "issue-badge" });
      badge.appendChild(svgEl("rect", {
        x,
        y: Math.max(0, box.y1 - 44),
        width: 18,
        height: 18,
        rx: 9,
        fill,
      }));
      const text = svgEl("text", {
        x: x + 5,
        y: Math.max(0, box.y1 - 31),
      });
      text.textContent = String(issueId);
      badge.appendChild(text);
      group.appendChild(badge);
    });
  }

  function overlayRect(box, className, options) {
    const settings = options || {};
    const group = svgEl("g", {});
    const rect = svgEl("rect", {
      x: box.x1,
      y: box.y1,
      width: Math.max(1, box.x2 - box.x1),
      height: Math.max(1, box.y2 - box.y1),
      class: `${className} ${settings.selected ? "box-selected" : ""} ${settings.focus ? "box-focus" : ""}`.trim(),
      "stroke-width": settings.selected || settings.focus ? 3 : 2,
      "data-object-id": box.id,
    });
    group.appendChild(rect);
    const labelText = box.source === "pred" && box.confidence != null
      ? `${box.class_name} ${box.confidence.toFixed(2)}`
      : box.class_name;
    const labelWidth = Math.max(60, labelText.length * 7 + 14);
    const labelRect = svgEl("rect", {
      x: box.x1,
      y: Math.max(0, box.y1 - 22),
      width: labelWidth,
      height: 18,
      rx: 6,
      fill: className === "box-gt" ? "var(--gt)" : className === "box-pred" ? "var(--pred)" : "var(--draft)",
    });
    const label = svgEl("text", {
      x: box.x1 + 7,
      y: Math.max(0, box.y1 - 9),
      class: "box-label",
    });
    label.textContent = labelText;
    group.appendChild(labelRect);
    group.appendChild(label);
    return group;
  }

  function renderHandles(selected) {
    const handleCoords = [
      [selected.x1, selected.y1, "nw"],
      [selected.x2, selected.y1, "ne"],
      [selected.x2, selected.y2, "se"],
      [selected.x1, selected.y2, "sw"],
    ];
    handleCoords.forEach(([x, y, corner]) => {
      const handle = svgEl("circle", {
        cx: x,
        cy: y,
        r: 6,
        class: "resize-handle",
        "data-corner": corner,
        "data-object-id": selected.id,
      });
      els.overlay.appendChild(handle);
    });
  }

  function applyStageTransform() {
    els.stageTransform.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  }

  function workspaceHeightBudget() {
    const panelHeight = els.centerPanel?.clientHeight || 0;
    const resizerHeight = els.workspaceResizer?.offsetHeight || 0;
    return {
      panelHeight,
      resizerHeight,
      available: Math.max(0, panelHeight - resizerHeight),
    };
  }

  function clampViewerRatio(ratio) {
    const { available } = workspaceHeightBudget();
    if (!available) {
      return Math.max(0.35, Math.min(0.78, ratio || 0.6));
    }
    const minRatio = Math.min(0.82, Math.max(0.25, MIN_VIEWER_HEIGHT / available));
    const maxRatio = Math.max(minRatio, Math.min(0.86, 1 - MIN_FOCUS_HEIGHT / available));
    return Math.max(minRatio, Math.min(maxRatio, ratio));
  }

  function applyWorkspaceSplit(options) {
    const settings = options || {};
    const { available } = workspaceHeightBudget();
    if (!available) return;
    state.viewerSplitRatio = clampViewerRatio(state.viewerSplitRatio);
    const viewerHeight = Math.round(available * state.viewerSplitRatio);
    const focusHeight = Math.max(MIN_FOCUS_HEIGHT, available - viewerHeight);
    els.viewerShell.style.flex = `0 0 ${viewerHeight}px`;
    els.focusPanel.style.flex = `0 0 ${focusHeight}px`;
    if (settings.skipRender === true) return;
    requestAnimationFrame(() => {
      if (state.current) {
        fitMainStageToImage();
      }
      if (selectedIssue()) {
        renderFocusLens();
      }
    });
  }

  function fitMainStageToImage() {
    if (!state.current) return;
    const rect = els.viewerStage.getBoundingClientRect();
    const imageSize = state.current.image_size || { width: 1, height: 1 };
    const padding = 28;
    const availableWidth = Math.max(160, rect.width - padding * 2);
    const availableHeight = Math.max(160, rect.height - padding * 2);
    const scale = Math.min(
      availableWidth / imageSize.width,
      availableHeight / imageSize.height,
      1,
    );
    state.zoom = Math.max(0.2, scale || 1);
    state.panX = (rect.width - imageSize.width * state.zoom) / 2;
    state.panY = (rect.height - imageSize.height * state.zoom) / 2;
    applyStageTransform();
  }

  function imagePointFromEvent(event) {
    const rect = els.overlay.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / state.zoom,
      y: (event.clientY - rect.top) / state.zoom,
    };
  }

  function beginWorkspaceResize(event) {
    event.preventDefault();
    const panelRect = els.centerPanel.getBoundingClientRect();
    state.layoutResize = {
      panelTop: panelRect.top,
      panelHeight: panelRect.height,
      resizerHeight: els.workspaceResizer.offsetHeight || 0,
    };
    els.workspaceResizer.classList.add("is-dragging");
  }

  function updateWorkspaceResize(clientY) {
    if (!state.layoutResize) return;
    const available = Math.max(0, state.layoutResize.panelHeight - state.layoutResize.resizerHeight);
    if (!available) return;
    const viewerHeight = clientY - state.layoutResize.panelTop - state.layoutResize.resizerHeight / 2;
    state.viewerSplitRatio = clampViewerRatio(viewerHeight / available);
    applyWorkspaceSplit();
  }

  function finishWorkspaceResize() {
    if (!state.layoutResize) return;
    state.layoutResize = null;
    els.workspaceResizer.classList.remove("is-dragging");
    saveViewerSplitRatio();
  }

  function emptyCounts() {
    return { fp: 0, fn: 0, cls: 0 };
  }

  function issueSummaryFromCounts(counts) {
    const summary = {
      fp: counts.fp || 0,
      fn: counts.fn || 0,
      cls: counts.cls || 0,
      severity: 2 * (counts.cls || 0) + (counts.fp || 0) + (counts.fn || 0),
    };
    summary.types = [];
    ["cls", "fn", "fp"].forEach((key) => {
      if (summary[key] > 0) summary.types.push(key);
    });
    if (!summary.types.length) {
      summary.types.push("no_issue");
    }
    return summary;
  }

  function boxIou(left, right) {
    const interX1 = Math.max(left.x1, right.x1);
    const interY1 = Math.max(left.y1, right.y1);
    const interX2 = Math.min(left.x2, right.x2);
    const interY2 = Math.min(left.y2, right.y2);
    const interW = Math.max(0, interX2 - interX1);
    const interH = Math.max(0, interY2 - interY1);
    const intersection = interW * interH;
    const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
    const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
    const union = leftArea + rightArea - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function greedyMatch(predictions, labels, sameClass, usedPredictions, usedLabels) {
    const predictionSet = new Set(usedPredictions || []);
    const labelSet = new Set(usedLabels || []);
    const candidates = [];
    predictions.forEach((prediction, predIndex) => {
      if (predictionSet.has(predIndex)) return;
      labels.forEach((label, labelIndex) => {
        if (labelSet.has(labelIndex)) return;
        if (sameClass && prediction.class_id !== label.class_id) return;
        if (!sameClass && prediction.class_id === label.class_id) return;
        const iou = boxIou(prediction, label);
        if (iou >= 0.5) {
          candidates.push({ iou, predIndex, labelIndex });
        }
      });
    });
    candidates.sort((left, right) => right.iou - left.iou);
    const matches = [];
    candidates.forEach((candidate) => {
      if (predictionSet.has(candidate.predIndex) || labelSet.has(candidate.labelIndex)) return;
      predictionSet.add(candidate.predIndex);
      labelSet.add(candidate.labelIndex);
      matches.push(candidate);
    });
    return { matches, usedPredictions: predictionSet, usedLabels: labelSet };
  }

  function bestMatchingObjectId(source, candidates, sameClass, minIou) {
    if (!source) return null;
    let bestId = null;
    let bestIou = 0;
    candidates.forEach((candidate) => {
      if (sameClass === true && candidate.class_id !== source.class_id) return;
      if (sameClass === false && candidate.class_id === source.class_id) return;
      const iou = boxIou(source, candidate);
      if (iou < (minIou || 0.3) || iou <= bestIou) return;
      bestIou = iou;
      bestId = candidate.id;
    });
    return bestId;
  }

  function expandedFocusBox(issue, gtObjects, draftObjects, predictionObjects, imageSize) {
    const related = [];
    [issue.gt_object_id, issue.draft_object_id, issue.pred_object_id].forEach((objectId) => {
      if (!objectId) return;
      const box = gtObjects.find((item) => item.id === objectId)
        || draftObjects.find((item) => item.id === objectId)
        || predictionObjects.find((item) => item.id === objectId);
      if (box) related.push(box);
    });
    if (!related.length) {
      return { x1: 0, y1: 0, x2: imageSize.width, y2: imageSize.height };
    }
    const x1 = Math.min(...related.map((box) => box.x1));
    const y1 = Math.min(...related.map((box) => box.y1));
    const x2 = Math.max(...related.map((box) => box.x2));
    const y2 = Math.max(...related.map((box) => box.y2));
    return {
      x1: Math.max(0, x1 - 24),
      y1: Math.max(0, y1 - 24),
      x2: Math.min(imageSize.width, x2 + 24),
      y2: Math.min(imageSize.height, y2 + 24),
    };
  }

  function compareObjectsToPredictions(labelObjects, predictionObjects, config) {
    const labelRole = config.labelRole || "draft";
    const imageSize = config.imageSize || { width: 1, height: 1 };
    const gtObjects = config.gtObjects || [];
    const draftObjects = config.draftObjects || [];
    const sameMatches = greedyMatch(predictionObjects, labelObjects, true);
    const confusionMatches = greedyMatch(
      predictionObjects,
      labelObjects,
      false,
      sameMatches.usedPredictions,
      sameMatches.usedLabels,
    );
    const issueItems = [];
    let nextId = 1;
    confusionMatches.matches.forEach((match) => {
      const prediction = predictionObjects[match.predIndex];
      const label = labelObjects[match.labelIndex];
      issueItems.push({
        id: nextId,
        type: "cls",
        gt_label: labelRole === "gt" ? label.class_name : null,
        pred_label: prediction.class_name,
        confidence: prediction.confidence,
        iou: match.iou,
        gt_object_id: labelRole === "gt" ? label.id : null,
        draft_object_id: labelRole === "draft" ? label.id : null,
        pred_object_id: prediction.id,
      });
      nextId += 1;
    });
    labelObjects.forEach((label, labelIndex) => {
      if (confusionMatches.usedLabels.has(labelIndex)) return;
      issueItems.push({
        id: nextId,
        type: "fn",
        gt_label: labelRole === "gt" ? label.class_name : null,
        pred_label: null,
        confidence: null,
        iou: null,
        gt_object_id: labelRole === "gt" ? label.id : null,
        draft_object_id: labelRole === "draft" ? label.id : null,
        pred_object_id: null,
      });
      nextId += 1;
    });
    predictionObjects.forEach((prediction, predIndex) => {
      if (confusionMatches.usedPredictions.has(predIndex)) return;
      issueItems.push({
        id: nextId,
        type: "fp",
        gt_label: null,
        pred_label: prediction.class_name,
        confidence: prediction.confidence,
        iou: null,
        gt_object_id: null,
        draft_object_id: null,
        pred_object_id: prediction.id,
      });
      nextId += 1;
    });
    const counts = emptyCounts();
    issueItems.forEach((issue) => {
      counts[issue.type] += 1;
    });
    const enriched = issueItems.map((issue) => {
      const nextIssue = { ...issue };
      let gtObject = nextIssue.gt_object_id ? gtObjects.find((item) => item.id === nextIssue.gt_object_id) : null;
      let draftObject = nextIssue.draft_object_id ? draftObjects.find((item) => item.id === nextIssue.draft_object_id) : null;
      let predObject = nextIssue.pred_object_id ? predictionObjects.find((item) => item.id === nextIssue.pred_object_id) : null;
      if (gtObject && !draftObject) {
        nextIssue.draft_object_id = bestMatchingObjectId(gtObject, draftObjects, true, 0.3);
        draftObject = nextIssue.draft_object_id ? draftObjects.find((item) => item.id === nextIssue.draft_object_id) : null;
      }
      if (draftObject && !gtObject) {
        nextIssue.gt_object_id = bestMatchingObjectId(draftObject, gtObjects, true, 0.3);
        gtObject = nextIssue.gt_object_id ? gtObjects.find((item) => item.id === nextIssue.gt_object_id) : null;
      }
      if (predObject && !gtObject) {
        nextIssue.gt_object_id = bestMatchingObjectId(predObject, gtObjects, null, 0.3);
        gtObject = nextIssue.gt_object_id ? gtObjects.find((item) => item.id === nextIssue.gt_object_id) : null;
      }
      if (predObject && !draftObject) {
        nextIssue.draft_object_id = bestMatchingObjectId(predObject, draftObjects, null, 0.3);
        draftObject = nextIssue.draft_object_id ? draftObjects.find((item) => item.id === nextIssue.draft_object_id) : null;
      }
      if (!predObject && gtObject) {
        nextIssue.pred_object_id = bestMatchingObjectId(gtObject, predictionObjects, null, 0.3);
        predObject = nextIssue.pred_object_id ? predictionObjects.find((item) => item.id === nextIssue.pred_object_id) : null;
      }
      if (!predObject && draftObject) {
        nextIssue.pred_object_id = bestMatchingObjectId(draftObject, predictionObjects, null, 0.3);
        predObject = nextIssue.pred_object_id ? predictionObjects.find((item) => item.id === nextIssue.pred_object_id) : null;
      }
      if (!nextIssue.gt_label && gtObject) {
        nextIssue.gt_label = gtObject.class_name;
      }
      if (!nextIssue.gt_label && draftObject) {
        nextIssue.gt_label = draftObject.class_name;
      }
      if (!nextIssue.pred_label && predObject) {
        nextIssue.pred_label = predObject.class_name;
      }
      if (nextIssue.confidence == null && predObject) {
        nextIssue.confidence = predObject.confidence;
      }
      if (nextIssue.iou == null) {
        const referenceObject = draftObject || gtObject;
        if (referenceObject && predObject) {
          nextIssue.iou = boxIou(referenceObject, predObject);
        }
      }
      nextIssue.focus_box = expandedFocusBox(nextIssue, gtObjects, draftObjects, predictionObjects, imageSize);
      return nextIssue;
    });
    return { summary: issueSummaryFromCounts(counts), items: enriched };
  }

  function rebuildDraftDiagnostics() {
    if (!state.current || !(state.current.prediction_objects || []).length) {
      state.liveDraftDiagnostics = {
        summary: state.current?.draft_issue_summary || state.current?.baseline_issue_summary || emptyIssueSummary(),
        items: state.current?.draft_issue_items || [],
      };
      return;
    }
    state.liveDraftDiagnostics = compareObjectsToPredictions(
      state.current.draft_objects || [],
      state.current.prediction_objects || [],
      {
        labelRole: "draft",
        gtObjects: state.current.gt_objects || [],
        draftObjects: state.current.draft_objects || [],
        imageSize: state.current.image_size || { width: 1, height: 1 },
      },
    );
  }

  function queueItemHtml(item) {
    return `
      <img class="queue-thumb" loading="lazy" src="/api/thumbnail/${item.image_id}" alt="${item.image_path.split("/").pop()}">
      <div class="queue-body">
        <div class="queue-item-title">${item.image_path.split("/").pop()}</div>
        <div class="queue-item-meta">${splitLabel(item.split)} · ${statusLabel(item.status)}</div>
        <div class="queue-item-meta">${issueCounts(item.issue_summary || emptyIssueSummary())}</div>
        <div class="tag-row">
          <span class="tag">问题分 ${item.issue_summary?.severity || 0}</span>
          ${(item.has_edits ? '<span class="tag warn">已编辑</span>' : "")}
        </div>
      </div>
    `;
  }

  function renderQueue() {
    ensureQueueRenderCount(currentItemIndex());
    const previousScrollTop = els.queueList.scrollTop;
    els.queueList.innerHTML = "";
    const visibleItems = visibleQueueItems();
    els.queueMeta.textContent = `${visibleItems.length} / ${state.items.length} 项`;
    visibleItems.forEach((item) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "queue-item" + (item.image_id === state.currentImageId ? " active" : "");
      card.innerHTML = queueItemHtml(item);
      card.addEventListener("click", async () => {
        if (!(await ensureSafeToNavigate())) return;
        await loadItem(item.image_id);
      });
      els.queueList.appendChild(card);
    });
    if (visibleItems.length < state.items.length) {
      const tail = document.createElement("div");
      tail.className = "queue-tail";
      tail.textContent = `继续向下滚动加载更多（剩余 ${state.items.length - visibleItems.length} 项）`;
      els.queueList.appendChild(tail);
    }
    els.queueList.scrollTop = previousScrollTop;
  }

  function renderSummary(summary) {
    const counts = summary.counts || summary.summary || summary;
    els.sessionSummary.innerHTML = `
      <div>总数 ${counts.total || 0}</div>
      <div>已编辑 ${counts.edited || 0}</div>
      <div>待审核 ${(counts.statuses && counts.statuses.todo) || 0}</div>
      <div>问题分 ${counts.total_severity || 0}</div>
    `;
  }

  function renderSelectedDraftEditor() {
    const selected = selectedDraft();
    if (!selected) {
      els.selectedDraftEmpty.classList.remove("hidden");
      els.selectedDraftPanel.classList.add("hidden");
      return;
    }
    els.selectedDraftEmpty.classList.add("hidden");
    els.selectedDraftPanel.classList.remove("hidden");
    populateSelect(els.selectedDraftClass, state.classNames, false);
    els.selectedDraftClass.value = selected.class_name;
  }

  function updateItemHeader() {
    if (!state.current) {
      els.itemHeader.textContent = "尚未选择图片";
      els.itemProgress.textContent = "0 / 0";
      return;
    }
    const item = state.current.item;
    const currentIndex = currentItemIndex();
    els.itemHeader.textContent = `${item.image_path.split("/").pop()} · ${splitLabel(item.split)}`;
    els.itemProgress.textContent = `${currentIndex + 1} / ${state.items.length}`;
    els.itemStatus.value = item.status;
    els.itemNote.value = item.note || "";
  }

  function issueResidualStatus(issue) {
    if (!state.current || !(state.current.prediction_objects || []).length) {
      return { resolved: false, label: "尚未加载预测，无法判定草稿状态" };
    }
    const liveIssues = state.liveDraftDiagnostics.items || [];
    const unresolved = liveIssues.some((candidate) => {
      if (issue.pred_object_id && candidate.pred_object_id && issue.pred_object_id === candidate.pred_object_id) {
        return true;
      }
      if (issue.draft_object_id && candidate.draft_object_id && issue.draft_object_id === candidate.draft_object_id) {
        return true;
      }
      if (issue.gt_object_id && candidate.gt_object_id && issue.gt_object_id === candidate.gt_object_id) {
        return true;
      }
      if (!issue.focus_box || !candidate.focus_box) return false;
      return boxIou(issue.focus_box, candidate.focus_box) >= 0.2;
    });
    if (unresolved) {
      return { resolved: false, label: "当前草稿在该区域仍存在模型冲突" };
    }
    return { resolved: true, label: "当前草稿已不再保留该问题冲突" };
  }

  function renderIssueChips() {
    const issues = currentIssueCollection();
    els.issueChipRow.innerHTML = "";
    issues.forEach((issue) => {
      const status = issueResidualStatus(issue);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "issue-chip" + (issue.id === state.selectedIssueId ? " active" : "") + (status.resolved ? " resolved" : "");
      button.innerHTML = `
        <div>#${issue.id} · ${issueLabel(issue.type)}</div>
        <span class="issue-chip-meta">${issue.gt_label || "-"} / ${issue.pred_label || "-"}</span>
      `;
      button.addEventListener("click", () => {
        selectIssue(issue.id, false);
      });
      els.issueChipRow.appendChild(button);
    });
  }

  function renderIssueList() {
    els.issueList.innerHTML = "";
    const issues = state.current?.baseline_issue_items || currentIssueCollection();
    if (!issues.length) {
      els.issueList.innerHTML = '<div class="muted">当前还没有详细问题列表。</div>';
      return;
    }
    issues.forEach((issue) => {
      const status = issueResidualStatus(issue);
      const row = document.createElement("div");
      row.className = "issue-row" + (issue.id === state.selectedIssueId ? " active" : "") + (status.resolved ? " resolved" : "");
      row.innerHTML = `
        <div><strong>#${issue.id}</strong> · ${issueLabel(issue.type)}</div>
        <div class="issue-row-meta">标注 GT ${issue.gt_label || "-"} · 预测 Pred ${issue.pred_label || "-"}</div>
        <div class="issue-row-meta">置信度 ${issue.confidence != null ? issue.confidence.toFixed(2) : "-"} · IoU ${issue.iou != null ? issue.iou.toFixed(2) : "-"}</div>
        <div class="issue-row-meta">${status.label}</div>
      `;
      row.addEventListener("click", () => {
        state.activeInspectorTab = "diagnostic";
        selectIssue(issue.id, false);
      });
      els.issueList.appendChild(row);
    });
  }

  function renderDraftList() {
    els.draftList.innerHTML = "";
    const drafts = state.current?.draft_objects || [];
    if (!drafts.length) {
      els.draftList.innerHTML = '<div class="muted">这张图当前还没有草稿框。</div>';
      return;
    }
    drafts.forEach((draft) => {
      const row = document.createElement("div");
      row.className = "object-row" + (draft.id === state.selectedDraftId ? " active" : "");
      row.innerHTML = `
        <div><strong>${draft.class_name}</strong></div>
        <div class="object-row-meta">${Math.round(draft.x1)}, ${Math.round(draft.y1)} → ${Math.round(draft.x2)}, ${Math.round(draft.y2)}</div>
      `;
      row.addEventListener("click", () => {
        state.selectedDraftId = draft.id;
        state.activeInspectorTab = "edit";
        renderCurrent();
      });
      els.draftList.appendChild(row);
    });
  }

  function renderPredictionList() {
    els.predictionList.innerHTML = "";
    const predictions = state.current?.prediction_objects || [];
    if (!predictions.length) {
      els.predictionList.innerHTML = '<div class="muted">这张图还没有加载预测结果。</div>';
      return;
    }
    predictions.forEach((prediction) => {
      const row = document.createElement("div");
      row.className = "object-row" + (prediction.id === state.highlightedPredictionId ? " active" : "");
      row.innerHTML = `
        <div><strong>${prediction.class_name}</strong></div>
        <div class="object-row-meta">置信度 ${prediction.confidence != null ? prediction.confidence.toFixed(2) : "-"}</div>
        <div class="row-actions">
          <button class="inline-btn">采纳为新框</button>
        </div>
      `;
      row.addEventListener("click", () => {
        state.highlightedPredictionId = prediction.id;
        renderCurrent();
      });
      row.querySelector("button").addEventListener("click", async (event) => {
        event.stopPropagation();
        await applyPredictionMutation(prediction.id, "append", null);
      });
      els.predictionList.appendChild(row);
    });
  }

  function renderIssueDetail() {
    const issue = selectedIssue();
    if (!issue) {
      els.selectedIssueSummary.textContent = "当前图片没有可聚焦的问题";
      els.selectedIssueEvidence.innerHTML = '<div class="muted">焦点区会展示当前问题的证据说明。</div>';
      els.selectedIssueDetail.innerHTML = '<div class="muted">先从焦点区或问题列表选择一个问题。</div>';
      els.issueActions.innerHTML = "";
      els.baselineState.textContent = "基线诊断：等待选择问题";
      els.baselineState.className = "state-card";
      els.draftState.textContent = "当前草稿：等待选择问题";
      els.draftState.className = "state-card";
      return;
    }
    const residual = issueResidualStatus(issue);
    els.selectedIssueSummary.textContent = `#${issue.id} · ${issueLabel(issue.type)} · GT ${issue.gt_label || "-"} / Pred ${issue.pred_label || "-"}`;
    els.selectedIssueEvidence.innerHTML = `
      <div><strong>${issueLabel(issue.type)}</strong></div>
      <div class="issue-detail-meta">原始标注 ${issue.gt_label || "-"} · 模型预测 ${issue.pred_label || "-"}</div>
      <div class="issue-detail-meta">置信度 ${issue.confidence != null ? issue.confidence.toFixed(2) : "-"} · IoU ${issue.iou != null ? issue.iou.toFixed(2) : "-"}</div>
    `;
    els.selectedIssueDetail.innerHTML = `
      <div><strong>问题说明</strong></div>
      <div class="issue-detail-meta">这条问题来自基线诊断，始终代表“原始标注 vs 模型预测”的证据，不会随着草稿修改而消失。</div>
      <div class="issue-detail-meta">当前草稿状态：${residual.label}</div>
    `;
    els.baselineState.textContent = `基线诊断：${issueLabel(issue.type)}，GT ${issue.gt_label || "-"}，Pred ${issue.pred_label || "-"}`;
    els.baselineState.className = "state-card warn";
    els.draftState.textContent = `当前草稿：${residual.label}`;
    els.draftState.className = `state-card ${residual.resolved ? "ok" : "warn"}`;
    renderIssueActions(issue);
  }

  function createActionButton(label, handler, disabled) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "inline-btn";
    button.textContent = label;
    if (disabled) {
      button.disabled = true;
    }
    button.addEventListener("click", handler);
    return button;
  }

  function renderIssueActions(issue) {
    els.issueActions.innerHTML = "";
    if (!issue) return;
    const targetDraftId = issue.draft_object_id || state.selectedDraftId || null;
    els.issueActions.appendChild(createActionButton("定位主画布", () => {
      focusMainStageOnBox(issue.focus_box);
    }, false));
    if (issue.type === "fp" && issue.pred_object_id) {
      els.issueActions.appendChild(createActionButton("采纳预测为新框", async () => {
        await applyPredictionMutation(issue.pred_object_id, "append", null);
      }, false));
    }
    if (issue.type === "cls" && issue.pred_object_id) {
      els.issueActions.appendChild(createActionButton("用预测覆盖当前草稿", async () => {
        await applyPredictionMutation(issue.pred_object_id, "replace_selected", targetDraftId);
      }, !targetDraftId));
      els.issueActions.appendChild(createActionButton("只替换类别", async () => {
        await applyPredictionMutation(issue.pred_object_id, "class_only", targetDraftId);
      }, !targetDraftId));
      els.issueActions.appendChild(createActionButton("只替换位置", async () => {
        await applyPredictionMutation(issue.pred_object_id, "box_only", targetDraftId);
      }, !targetDraftId));
    }
    if (issue.type === "fn") {
      els.issueActions.appendChild(createActionButton("定位并补框", () => {
        state.mode = "draw";
        updateToolbarModes();
        focusMainStageOnBox(issue.focus_box);
      }, false));
    }
  }

  function renderSummaryCards() {
    const baselineSummary = state.current?.baseline_issue_summary || emptyIssueSummary();
    const draftSummary = state.liveDraftDiagnostics.summary || emptyIssueSummary();
    els.baselineSummaryCard.textContent = `基线诊断 ${baselineSummary.severity || 0} 分 · ${issueCounts(baselineSummary)}`;
    els.draftSummaryCard.textContent = `当前草稿 ${draftSummary.severity || 0} 分 · ${issueCounts(draftSummary)}`;
  }

  function renderOverlay() {
    if (!state.current) return;
    const size = state.current.image_size;
    const baselineIssues = state.current.baseline_issue_items || [];
    const liveDraftIssues = state.liveDraftDiagnostics.items || [];
    const selectedIssueValue = selectedIssue();
    els.overlay.innerHTML = "";
    els.overlay.setAttribute("viewBox", `0 0 ${size.width} ${size.height}`);
    els.overlay.setAttribute("width", String(size.width));
    els.overlay.setAttribute("height", String(size.height));
    els.overlay.setAttribute("preserveAspectRatio", "none");
    els.stageTransform.style.width = `${size.width}px`;
    els.stageTransform.style.height = `${size.height}px`;
    els.mainImage.width = size.width;
    els.mainImage.height = size.height;
    els.mainImage.style.width = `${size.width}px`;
    els.mainImage.style.height = `${size.height}px`;
    els.overlay.style.width = `${size.width}px`;
    els.overlay.style.height = `${size.height}px`;

    if (showGtLayer()) {
      state.current.gt_objects.forEach((box) => {
        const focus = Boolean(selectedIssueValue && selectedIssueValue.gt_object_id === box.id);
        const group = overlayRect(box, "box-gt", { focus });
        renderIssueBadges(group, box, issueBadgesForObject(box.id, baselineIssues, "gt_object_id"), "#ebb23b");
        els.overlay.appendChild(group);
      });
    }

    if (showPredLayer()) {
      state.current.prediction_objects.forEach((box) => {
        const selected = box.id === state.highlightedPredictionId;
        const focus = Boolean(selectedIssueValue && selectedIssueValue.pred_object_id === box.id);
        const group = overlayRect(box, "box-pred", { selected, focus });
        renderIssueBadges(group, box, issueBadgesForObject(box.id, baselineIssues, "pred_object_id"), "#d84f3a");
        els.overlay.appendChild(group);
      });
    }

    if (showDraftLayer()) {
      state.current.draft_objects.forEach((box) => {
        const selected = box.id === state.selectedDraftId;
        const focus = Boolean(selectedIssueValue && selectedIssueValue.draft_object_id === box.id);
        const group = overlayRect(box, "box-draft", { selected, focus });
        renderIssueBadges(group, box, issueBadgesForObject(box.id, liveDraftIssues, "draft_object_id"), "#1f8ea2");
        const rect = group.querySelector("rect");
        rect.addEventListener("pointerdown", (event) => {
          if (state.mode !== "select") return;
          event.stopPropagation();
          state.selectedDraftId = box.id;
          state.activeInspectorTab = "edit";
          state.highlightedPredictionId = null;
          state.pointerAction = {
            type: "move",
            objectId: box.id,
            startPoint: imagePointFromEvent(event),
            original: { ...box },
          };
          renderCurrent();
        });
        els.overlay.appendChild(group);
      });
      const selected = selectedDraft();
      if (selected) renderHandles(selected);
    }

    if (state.drawPreview) {
      els.overlay.appendChild(
        svgEl("rect", {
          x: Math.min(state.drawPreview.x1, state.drawPreview.x2),
          y: Math.min(state.drawPreview.y1, state.drawPreview.y2),
          width: Math.abs(state.drawPreview.x2 - state.drawPreview.x1),
          height: Math.abs(state.drawPreview.y2 - state.drawPreview.y1),
          class: "box-draft",
          "stroke-dasharray": "6 4",
          "stroke-width": 2,
        }),
      );
    }
  }

  function focusBoxesForSelectedIssue() {
    const issue = selectedIssue();
    if (!issue || !state.current) return [];
    const boxes = [];
    if ((state.selectedDiagnosticLayer === "gt" || state.selectedDiagnosticLayer === "combined") && issue.gt_object_id) {
      const gtObject = getGtById(issue.gt_object_id);
      if (gtObject) boxes.push({ box: gtObject, className: "box-gt", selected: false, focus: true });
    }
    if ((state.selectedDiagnosticLayer === "pred" || state.selectedDiagnosticLayer === "combined") && issue.pred_object_id) {
      const prediction = getPredictionById(issue.pred_object_id);
      if (prediction) boxes.push({ box: prediction, className: "box-pred", selected: true, focus: true });
    }
    if ((state.selectedDiagnosticLayer === "draft" || state.selectedDiagnosticLayer === "combined") && issue.draft_object_id) {
      const draftObject = getDraftById(issue.draft_object_id);
      if (draftObject) boxes.push({ box: draftObject, className: "box-draft", selected: draftObject.id === state.selectedDraftId, focus: true });
    }
    return boxes;
  }

  function renderFocusLens() {
    const issue = selectedIssue();
    if (!state.current || !issue) {
      els.focusEmpty.classList.remove("hidden");
      els.focusStage.classList.add("hidden");
      return;
    }
    els.focusEmpty.classList.add("hidden");
    els.focusStage.classList.remove("hidden");
    const imageSize = state.current.image_size;
    const focusBox = issue.focus_box || { x1: 0, y1: 0, x2: imageSize.width, y2: imageSize.height };
    const width = Math.max(320, els.focusViewport.clientWidth || 640);
    const height = Math.max(280, els.focusViewport.clientHeight || 320);
    const boxWidth = Math.max(1, focusBox.x2 - focusBox.x1);
    const boxHeight = Math.max(1, focusBox.y2 - focusBox.y1);
    const scale = Math.min(width / boxWidth, height / boxHeight);
    const offsetX = (width - boxWidth * scale) / 2;
    const offsetY = (height - boxHeight * scale) / 2;
    const transform = `translate(${offsetX - focusBox.x1 * scale}px, ${offsetY - focusBox.y1 * scale}px) scale(${scale})`;

    els.focusImage.width = imageSize.width;
    els.focusImage.height = imageSize.height;
    els.focusImage.style.width = `${imageSize.width}px`;
    els.focusImage.style.height = `${imageSize.height}px`;
    els.focusImage.style.transform = transform;
    els.focusOverlay.innerHTML = "";
    els.focusOverlay.setAttribute("viewBox", `0 0 ${imageSize.width} ${imageSize.height}`);
    els.focusOverlay.setAttribute("width", String(imageSize.width));
    els.focusOverlay.setAttribute("height", String(imageSize.height));
    els.focusOverlay.setAttribute("preserveAspectRatio", "none");
    els.focusOverlay.style.width = `${imageSize.width}px`;
    els.focusOverlay.style.height = `${imageSize.height}px`;
    els.focusOverlay.style.transform = transform;

    focusBoxesForSelectedIssue().forEach((entry) => {
      els.focusOverlay.appendChild(overlayRect(entry.box, entry.className, entry));
    });
  }

  function renderCurrent() {
    updateInspectorTabs();
    updateObjectTabs();
    updateLayerButtons();
    if (!state.current) {
      els.viewerEmpty.classList.remove("hidden");
      els.viewerStage.classList.add("hidden");
      els.focusEmpty.classList.remove("hidden");
      els.focusStage.classList.add("hidden");
      updateItemHeader();
      renderQueue();
      return;
    }
    ensureSelectedIssue();
    els.viewerEmpty.classList.add("hidden");
    els.viewerStage.classList.remove("hidden");
    updateItemHeader();
    renderSummaryCards();
    renderIssueChips();
    renderIssueDetail();
    renderIssueList();
    renderDraftList();
    renderPredictionList();
    renderSelectedDraftEditor();
    renderOverlay();
    renderFocusLens();
    applyStageTransform();
    renderQueue();
  }

  async function refreshSummary() {
    const payload = await fetchJSON("/api/session/summary");
    renderSummary(payload);
  }

  async function refreshItems() {
    const payload = await fetchJSON(`/api/items?${queryString(currentFilters())}`);
    state.items = payload.items || [];
    state.queueRenderCount = Math.min(state.items.length, state.queueBatchSize);
    renderQueue();
    if (!state.currentImageId && state.items.length) {
      await loadItem(state.items[0].image_id);
      return;
    }
    if (state.currentImageId && !state.items.some((item) => item.image_id === state.currentImageId)) {
      state.current = null;
      state.currentImageId = null;
      if (state.items.length) {
        await loadItem(state.items[0].image_id);
      } else {
        renderCurrent();
      }
    }
  }

  async function loadItem(imageId) {
    const payload = await fetchJSON(`/api/items/${imageId}`);
    state.current = payload;
    state.currentImageId = imageId;
    state.selectedDraftId = payload.draft_objects[0]?.id || null;
    state.highlightedPredictionId = null;
    state.selectedIssueId = payload.baseline_issue_items?.[0]?.id || payload.draft_issue_items?.[0]?.id || null;
    state.zoom = 1;
    state.panX = 0;
    state.panY = 0;
    state.drawPreview = null;
    els.mainImage.src = `/api/image/${imageId}`;
    els.focusImage.src = `/api/image/${imageId}`;
    rebuildDraftDiagnostics();
    setDirty(false);
    ensureQueueRenderCount(state.items.findIndex((item) => item.image_id === imageId));
    renderCurrent();
    requestAnimationFrame(() => fitMainStageToImage());
  }

  function deleteSelectedDraft() {
    if (!state.current || !state.selectedDraftId) return;
    state.current.draft_objects = state.current.draft_objects.filter((obj) => obj.id !== state.selectedDraftId);
    state.selectedDraftId = state.current.draft_objects[0]?.id || null;
    rebuildDraftDiagnostics();
    setDirty(true);
    renderCurrent();
  }

  async function saveCurrentItem() {
    if (!state.current) return;
    const payload = await fetchJSON(`/api/items/${state.current.item.image_id}/save`, {
      method: "POST",
      body: JSON.stringify({
        draft_objects: state.current.draft_objects,
        status: els.itemStatus.value,
        note: els.itemNote.value,
      }),
    });
    state.current = payload;
    state.selectedDraftId = payload.draft_objects[0]?.id || state.selectedDraftId;
    rebuildDraftDiagnostics();
    setDirty(false);
    await refreshSummary();
    await refreshItems();
    renderCurrent();
  }

  async function loadPredictions() {
    if (!state.current) return;
    const payload = await fetchJSON(`/api/items/${state.current.item.image_id}/predict`, {
      method: "POST",
    });
    state.current = payload;
    state.highlightedPredictionId = null;
    rebuildDraftDiagnostics();
    ensureSelectedIssue();
    renderCurrent();
    await refreshSummary();
    await refreshItems();
  }

  async function useAllPredictions() {
    if (!state.current) return;
    if (!window.confirm("整张采用预测会替换当前所有草稿框，确认继续吗？")) {
      return;
    }
    const payload = await fetchJSON(`/api/items/${state.current.item.image_id}/replace-with-predictions`, {
      method: "POST",
    });
    state.current.draft_objects = payload.draft_objects;
    state.selectedDraftId = payload.draft_objects[0]?.id || null;
    rebuildDraftDiagnostics();
    setDirty(true);
    renderCurrent();
  }

  async function applyPredictionMutation(predictionId, mode, targetDraftId) {
    if (!state.current) return;
    const payload = await fetchJSON(`/api/items/${state.current.item.image_id}/apply-prediction`, {
      method: "POST",
      body: JSON.stringify({
        prediction_id: predictionId,
        mode,
        target_draft_id: targetDraftId,
      }),
    });
    state.current.draft_objects = payload.draft_objects;
    state.selectedDraftId = targetDraftId || payload.draft_objects[payload.draft_objects.length - 1]?.id || null;
    rebuildDraftDiagnostics();
    setDirty(true);
    renderCurrent();
  }

  function setMode(mode) {
    state.mode = mode;
    updateToolbarModes();
  }

  function selectedDraftIndex() {
    return state.current?.draft_objects.findIndex((item) => item.id === state.selectedDraftId) ?? -1;
  }

  function startDraw(event) {
    const point = imagePointFromEvent(event);
    state.drawPreview = { x1: point.x, y1: point.y, x2: point.x, y2: point.y };
    state.pointerAction = { type: "draw", startPoint: point };
    renderCurrent();
  }

  function startPan(event) {
    state.pointerAction = {
      type: "pan",
      clientX: event.clientX,
      clientY: event.clientY,
      panX: state.panX,
      panY: state.panY,
    };
  }

  function maybeStartResize(event) {
    const target = event.target;
    if (!(target instanceof SVGCircleElement)) return false;
    const corner = target.dataset.corner;
    const objectId = target.dataset.objectId;
    const draft = getDraftById(objectId);
    if (!corner || !draft) return false;
    state.selectedDraftId = objectId;
    state.pointerAction = {
      type: "resize",
      objectId,
      corner,
      startPoint: imagePointFromEvent(event),
      original: { ...draft },
    };
    return true;
  }

  function handleOverlayPointerDown(event) {
    if (!state.current) return;
    if (maybeStartResize(event)) {
      renderCurrent();
      return;
    }
    if (state.mode === "draw") {
      startDraw(event);
      return;
    }
    if (state.mode === "pan") {
      startPan(event);
      return;
    }
    if (event.target === els.overlay) {
      state.selectedDraftId = null;
      renderCurrent();
    }
  }

  function updateMoveAction(point) {
    const action = state.pointerAction;
    const index = selectedDraftIndex();
    if (!action || index < 0) return;
    const dx = point.x - action.startPoint.x;
    const dy = point.y - action.startPoint.y;
    state.current.draft_objects[index] = {
      ...state.current.draft_objects[index],
      x1: action.original.x1 + dx,
      y1: action.original.y1 + dy,
      x2: action.original.x2 + dx,
      y2: action.original.y2 + dy,
    };
  }

  function updateResizeAction(point) {
    const action = state.pointerAction;
    const index = selectedDraftIndex();
    if (!action || index < 0) return;
    const next = { ...state.current.draft_objects[index] };
    if (action.corner.includes("n")) next.y1 = point.y;
    if (action.corner.includes("s")) next.y2 = point.y;
    if (action.corner.includes("w")) next.x1 = point.x;
    if (action.corner.includes("e")) next.x2 = point.x;
    state.current.draft_objects[index] = next;
  }

  function handlePointerMove(event) {
    if (state.layoutResize) {
      updateWorkspaceResize(event.clientY);
      return;
    }
    if (!state.pointerAction || !state.current) return;
    if (state.pointerAction.type === "draw") {
      const point = imagePointFromEvent(event);
      state.drawPreview = {
        x1: state.pointerAction.startPoint.x,
        y1: state.pointerAction.startPoint.y,
        x2: point.x,
        y2: point.y,
      };
      renderCurrent();
      return;
    }
    if (state.pointerAction.type === "pan") {
      state.panX = state.pointerAction.panX + (event.clientX - state.pointerAction.clientX);
      state.panY = state.pointerAction.panY + (event.clientY - state.pointerAction.clientY);
      applyStageTransform();
      return;
    }
    const point = imagePointFromEvent(event);
    if (state.pointerAction.type === "move") {
      updateMoveAction(point);
      setDirty(true);
      renderCurrent();
    }
    if (state.pointerAction.type === "resize") {
      updateResizeAction(point);
      setDirty(true);
      renderCurrent();
    }
  }

  function finishPointerAction() {
    if (state.layoutResize) {
      finishWorkspaceResize();
      return;
    }
    if (!state.current || !state.pointerAction) return;
    if (state.pointerAction.type === "draw" && state.drawPreview) {
      const draft = {
        id: `draft-${Date.now()}`,
        class_id: 0,
        class_name: state.classNames[0],
        x1: Math.min(state.drawPreview.x1, state.drawPreview.x2),
        y1: Math.min(state.drawPreview.y1, state.drawPreview.y2),
        x2: Math.max(state.drawPreview.x1, state.drawPreview.x2),
        y2: Math.max(state.drawPreview.y1, state.drawPreview.y2),
        source: "draft",
      };
      if (draft.x2 - draft.x1 > 4 && draft.y2 - draft.y1 > 4) {
        state.current.draft_objects.push(draft);
        state.selectedDraftId = draft.id;
        setDirty(true);
      }
      state.drawPreview = null;
    }
    if (state.pointerAction.type === "move" || state.pointerAction.type === "resize" || state.pointerAction.type === "draw") {
      rebuildDraftDiagnostics();
      renderCurrent();
    }
    state.pointerAction = null;
  }

  function changeSelectedDraftClass(value) {
    const selected = selectedDraft();
    if (!selected) return;
    const index = selectedDraftIndex();
    state.current.draft_objects[index] = {
      ...selected,
      class_id: state.classNames.indexOf(value),
      class_name: value,
    };
    rebuildDraftDiagnostics();
    setDirty(true);
    renderCurrent();
  }

  function focusMainStageOnBox(box) {
    if (!state.current || !box) return;
    const stageRect = els.viewerStage.getBoundingClientRect();
    const width = Math.max(240, stageRect.width - 40);
    const height = Math.max(240, stageRect.height - 40);
    const boxWidth = Math.max(20, box.x2 - box.x1);
    const boxHeight = Math.max(20, box.y2 - box.y1);
    const scale = Math.max(0.35, Math.min(4, Math.min(width / boxWidth, height / boxHeight)));
    state.zoom = scale;
    state.panX = (stageRect.width - boxWidth * scale) / 2 - box.x1 * scale;
    state.panY = (stageRect.height - boxHeight * scale) / 2 - box.y1 * scale;
    applyStageTransform();
  }

  function selectIssue(issueId, autoFocus) {
    state.selectedIssueId = issueId;
    const issue = selectedIssue();
    if (issue) {
      state.selectedDraftId = issue.draft_object_id || state.selectedDraftId;
      state.highlightedPredictionId = issue.pred_object_id || null;
      if (autoFocus) {
        requestAnimationFrame(() => focusMainStageOnBox(issue.focus_box));
      }
    }
    renderCurrent();
  }

  function selectAdjacentIssue(delta) {
    const issues = currentIssueCollection();
    if (!issues.length) return;
    const index = issues.findIndex((issue) => issue.id === state.selectedIssueId);
    const nextIndex = Math.max(0, Math.min(issues.length - 1, index + delta));
    selectIssue(issues[nextIndex].id, true);
  }

  async function ensureSafeToNavigate() {
    if (!state.dirty) return true;
    return window.confirm("当前图片还有未保存修改，确认丢弃并切换吗？");
  }

  async function loadAdjacentItem(delta) {
    if (!(await ensureSafeToNavigate())) return;
    const index = currentItemIndex();
    if (index < 0) return;
    const nextIndex = Math.max(0, Math.min(state.items.length - 1, index + delta));
    if (nextIndex === index) return;
    await loadItem(state.items[nextIndex].image_id);
  }

  function markStatus(nextStatus) {
    els.itemStatus.value = nextStatus;
    setDirty(true);
  }

  function maybeLoadMoreQueueItems() {
    if (!state.items.length || state.queueRenderCount >= state.items.length) {
      return;
    }
    const threshold = 240;
    const distanceToBottom = els.queueList.scrollHeight - els.queueList.scrollTop - els.queueList.clientHeight;
    if (distanceToBottom > threshold) {
      return;
    }
    state.queueRenderCount = Math.min(state.items.length, state.queueRenderCount + state.queueBatchSize);
    renderQueue();
  }

  function isTypingTarget(target) {
    return target instanceof HTMLInputElement
      || target instanceof HTMLTextAreaElement
      || target instanceof HTMLSelectElement
      || target?.isContentEditable;
  }

  function wireFilters() {
    [
      els.filterSplit,
      els.filterStatus,
      els.filterIssue,
      els.filterGtClass,
      els.filterPredClass,
      els.filterEditedOnly,
      els.filterOrder,
    ].forEach((element) => {
      element.addEventListener("change", refreshItems);
    });
    els.filterQuery.addEventListener("input", () => {
      window.clearTimeout(els.filterQuery._timer);
      els.filterQuery._timer = window.setTimeout(refreshItems, 200);
    });
  }

  function wireActions() {
    els.modeSelect.addEventListener("click", () => setMode("select"));
    els.modeDraw.addEventListener("click", () => setMode("draw"));
    els.modePan.addEventListener("click", () => setMode("pan"));
    els.deleteObject.addEventListener("click", deleteSelectedDraft);
    els.saveItem.addEventListener("click", saveCurrentItem);
    els.loadPredictions.addEventListener("click", loadPredictions);
    els.useAllPredictions.addEventListener("click", useAllPredictions);
    els.zoomIn.addEventListener("click", () => {
      state.zoom = Math.min(4, state.zoom + 0.12);
      applyStageTransform();
    });
    els.zoomOut.addEventListener("click", () => {
      state.zoom = Math.max(0.25, state.zoom - 0.12);
      applyStageTransform();
    });
    els.zoomReset.addEventListener("click", () => {
      fitMainStageToImage();
    });
    els.workspaceResizer.addEventListener("pointerdown", beginWorkspaceResize);
    [els.toggleGt, els.togglePred, els.toggleDraft].forEach((element) => {
      element.addEventListener("change", () => {
        if (!state.current) return;
        renderOverlay();
        applyStageTransform();
      });
    });
    els.selectedDraftClass.addEventListener("change", (event) => changeSelectedDraftClass(event.target.value));
    els.itemStatus.addEventListener("change", () => setDirty(true));
    els.itemNote.addEventListener("input", () => setDirty(true));
    els.queueList.addEventListener("scroll", maybeLoadMoreQueueItems);
    els.overlay.addEventListener("pointerdown", handleOverlayPointerDown);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", finishPointerAction);
    els.prevItem.addEventListener("click", () => { loadAdjacentItem(-1); });
    els.nextItem.addEventListener("click", () => { loadAdjacentItem(1); });
    els.tabDiagnostic.addEventListener("click", () => {
      state.activeInspectorTab = "diagnostic";
      renderCurrent();
    });
    els.tabEdit.addEventListener("click", () => {
      state.activeInspectorTab = "edit";
      renderCurrent();
    });
    els.tabDraftObjects.addEventListener("click", () => {
      state.activeObjectTab = "draft";
      renderCurrent();
    });
    els.tabPredictionObjects.addEventListener("click", () => {
      state.activeObjectTab = "prediction";
      renderCurrent();
    });
    els.layerButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedDiagnosticLayer = button.dataset.layer;
        renderCurrent();
      });
    });
    window.addEventListener("keydown", (event) => {
      if (isTypingTarget(event.target)) return;
      const key = event.key.toLowerCase();
      if (key === "j") {
        event.preventDefault();
        loadAdjacentItem(-1);
      } else if (key === "k") {
        event.preventDefault();
        loadAdjacentItem(1);
      } else if (key === "n") {
        event.preventDefault();
        selectAdjacentIssue(-1);
      } else if (key === "m") {
        event.preventDefault();
        selectAdjacentIssue(1);
      } else if (key === "s") {
        event.preventDefault();
        saveCurrentItem();
      } else if (key === "r") {
        event.preventDefault();
        markStatus("reviewed_ok");
      } else if (key === "f") {
        event.preventDefault();
        markStatus("fixed");
      } else if (key === "delete" || key === "backspace") {
        deleteSelectedDraft();
      }
    });
    window.addEventListener("beforeunload", (event) => {
      if (!state.dirty) return;
      event.preventDefault();
      event.returnValue = "";
    });
    window.addEventListener("resize", () => {
      applyWorkspaceSplit();
    });
  }

  async function init() {
    state.viewerSplitRatio = loadViewerSplitRatio();
    populateSelect(els.filterSplit, state.splits, true, splitLabel);
    populateSelect(els.filterStatus, state.statuses, true, statusLabel);
    populateSelect(els.itemStatus, state.statuses, false, statusLabel);
    populateSelect(els.filterGtClass, state.classNames, true);
    populateSelect(els.filterPredClass, state.classNames, true);
    populateSelect(els.selectedDraftClass, state.classNames, false);
    updateToolbarModes();
    updateInspectorTabs();
    updateObjectTabs();
    updateLayerButtons();
    wireFilters();
    wireActions();
    applyWorkspaceSplit({ skipRender: true });
    await refreshSummary();
    await refreshItems();
  }

  init().catch((error) => {
    console.error(error);
    window.alert(error.message || "标注走查工具初始化失败");
  });
})();
