const DATA = window.DSV4_GRAPH;
const I18N = window.DSV4_I18N || {};
// Development guard: group narratives are hand-written prose, so assert that
// every detailed group member still has an inline node link in its description.
const VERIFY_GROUP_NARRATIVE_LINKS = true;

function initialStoredChoice(key, fallback) {
  const saved = window.localStorage?.getItem(key);
  if (saved) return saved;
  window.localStorage?.setItem(key, fallback);
  return fallback;
}

function initialTheme() {
  const saved = window.localStorage?.getItem("dsv4-theme");
  if (saved === "dark" || saved === "light") return saved;
  const detected = window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  window.localStorage?.setItem("dsv4-theme", detected);
  return detected;
}

const state = {
  model: "pro",
  layer: 0,
  scene: "overview",
  graphDetail: "detailed",
  selected: "input-ids",
  selectedGroup: null,
  groupHighlightActive: false,
  detailOpen: true,
  detailPanelPosition: null,
  showNodeFormula: true,
  embedNodeDescription: false,
  theme: initialTheme(),
  lang: initialStoredChoice("dsv4-lang", "en") === "ko" ? "ko" : "en",
};

const elk = new ELK();
const ZOOM_EXTENT = [0.08, 14];
let zoomBehavior;
let lastLayout = { width: 1280, height: 1280 };
let layoutOffset = { x: 0, y: 0 };
let renderVersion = 0;
let shouldFitAfterLayout = true;
let minimapState = null;
let measureHost = null;
let detailPanelDrag = null;

function model() {
  return DATA.models[state.model];
}

function scene() {
  return DATA.scenes[state.scene];
}

function sceneView() {
  const current = scene();
  // Keep simple/detailed graph topology isolated: nodes, edges, and groups
  // are selected as one view bundle instead of filtering detailed topology.
  return current.views?.[state.graphDetail] || current.views?.detailed || current;
}

function t(key) {
  return I18N.ui?.[state.lang]?.[key] || I18N.ui?.ko?.[key] || key;
}

function isEnglish() {
  return state.lang === "en";
}

function nodeTitle(doc) {
  return doc?.title || "";
}

function nodeText(doc, lang = state.lang) {
  return doc?.id ? I18N.nodeText?.[lang]?.[doc.id] || null : null;
}

function nodeSummary(doc) {
  if (!doc) return "";
  const text = nodeText(doc);
  const localized = text?.details?.why || text?.summary;
  if (localized) return resolve(localized);
  if (!isEnglish()) return resolve(doc.details?.why || doc.summary || "");
  return englishNodeDescription(doc);
}

function englishNodeDescription(doc) {
  const input = resolve(doc.input || "");
  const output = resolve(doc.output || "");
  const category = doc.category || "node";
  return `${doc.title} is a ${category} node that maps ${input || "its input"} to ${output || "its output"} in the DeepSeek V4 graph. It keeps the tensor transformation explicit while the surrounding group explains why this step exists in the model.`;
}

function nodeDetails(doc) {
  const text = nodeText(doc);
  const details = { ...(text?.details || {}) };
  if (isEnglish() && !details.why) details.why = englishNodeDescription(doc);
  if (isEnglish() && !details.runtime) details.runtime = englishNodeRuntime(doc);
  const formula = localizedFormula(doc);
  if (formula.length) details.formula = formula;
  return details;
}

function englishNodeRuntime(doc) {
  const params = Object.entries(doc.params || {});
  if (!params.length) return "Runtime behavior follows the displayed input and output tensor shapes for this node.";
  const renderedParams = params.map(([key, value]) => `${key}=${resolve(value)}`).join(", ");
  return `Key runtime parameters are ${renderedParams}.`;
}

function ratioValue() {
  return model().schedule[state.layer] ?? 0;
}

function attentionMode() {
  const ratio = ratioValue();
  if (ratio === 4) return "csa";
  if (ratio === 128) return "hca";
  return "swa";
}

function ratioLabel() {
  return `R=${ratioValue()}`;
}

function resolve(value) {
  if (typeof value !== "string") return typeof value === "function" ? value() : value;
  return value
    .replaceAll("$D", String(model().D))
    .replaceAll("$H", String(model().H))
    .replaceAll("$G", String(model().G))
    .replaceAll("$Qr", String(model().Qr))
    .replaceAll("$Or", String(model().Or))
    .replaceAll("$E", String(model().E))
    .replaceAll("$I", String(model().I))
    .replaceAll("$indexTopK", String(model().indexTopK))
    .replaceAll("$routeScale", String(model().routeScale))
    .replaceAll("$R", ratioLabel());
}

function visibleNode(id) {
  const node = DATA.nodes[id];
  if (!node) return false;
  return visibleWhen(node.visibleWhen);
}

function visibleWhen(condition) {
  if (!condition) return true;
  if (condition.ratio !== undefined) {
    const ratios = Array.isArray(condition.ratio) ? condition.ratio : [condition.ratio];
    if (!ratios.includes(ratioValue())) return false;
  }
  if (condition.mode !== undefined) {
    const modes = Array.isArray(condition.mode) ? condition.mode : [condition.mode];
    if (!modes.includes(attentionMode())) return false;
  }
  return true;
}

function visibleNodeIds() {
  return sceneView().nodeIds.filter(visibleNode);
}

function visibleEdges() {
  const ids = new Set(visibleNodeIds());
  return sceneView().edges.filter((edge) => ids.has(edge.from) && ids.has(edge.to) && visibleWhen(edge.visibleWhen));
}

function render(fitAfterLayout = false) {
  shouldFitAfterLayout ||= fitAfterLayout;
  ensureSelection();
  applyTheme();
  applyLanguage();
  renderModePicker();
  renderGraph();
  renderDetail();
  renderPanelState();
}

function renderModePicker() {
  document.querySelectorAll("[data-layer-preset]").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.layerPreset) === state.layer);
  });
  document.querySelectorAll("[data-scene-select]").forEach((button) => {
    button.classList.toggle("active", button.dataset.sceneSelect === state.scene);
  });
  document.querySelectorAll("[data-detail-select]").forEach((button) => {
    button.classList.toggle("active", button.dataset.detailSelect === state.graphDetail);
  });
  document.querySelectorAll("[data-lang-select]").forEach((button) => {
    button.classList.toggle("active", button.dataset.langSelect === state.lang);
  });
  const descToggle = document.querySelector("#embedDescToggle");
  if (descToggle) descToggle.checked = state.embedNodeDescription;
  const formulaToggle = document.querySelector("#formulaToggle");
  if (formulaToggle) formulaToggle.checked = state.showNodeFormula;
  const darkToggle = document.querySelector("#darkModeToggle");
  if (darkToggle) darkToggle.checked = state.theme === "dark";
  document.querySelectorAll("[data-ui-key]").forEach((item) => {
    item.textContent = t(item.dataset.uiKey);
  });
  const overviewButton = document.querySelector('[data-scene-select="overview"]');
  if (overviewButton) overviewButton.textContent = t("overview");
  const formulaLabel = document.querySelector("#formulaToggle + span");
  if (formulaLabel) formulaLabel.textContent = t("nodeFormulas");
  const descLabel = document.querySelector("#embedDescToggle + span");
  if (descLabel) descLabel.textContent = t("embedDesc");
  const darkLabel = document.querySelector("#darkModeToggle + span");
  if (darkLabel) darkLabel.textContent = t("dark");
  const drill = document.querySelector("#drillButton");
  if (drill) drill.textContent = t("openSubgraph");
}

function applyTheme() {
  document.body.classList.toggle("dark", state.theme === "dark");
  document.body.classList.toggle("light", state.theme !== "dark");
}

function applyLanguage() {
  document.documentElement.lang = state.lang;
}

function renderStats() {
  if (!document.querySelector("#variantStats")) return;
  const m = model();
  document.querySelector("#variantStats").innerHTML = [
    ["model", m.label],
    ["total", m.total],
    ["active", m.active],
    ["D", m.D],
    ["layers", m.layers],
    ["heads", m.H],
    ["experts", m.E],
  ]
    .map(([key, value]) => `<span>${key}<b>${value}</b></span>`)
    .join("");
}

function renderLayerControls() {
  if (!document.querySelector("#layerSlider")) return;
  const m = model();
  const maxLayer = m.schedule.length - 1;
  if (state.layer > maxLayer) state.layer = maxLayer;

  const slider = document.querySelector("#layerSlider");
  slider.max = String(maxLayer);
  slider.value = String(state.layer);

  document.querySelector("#layerLabel").textContent =
    state.layer === m.layers ? `MTP ${state.layer}` : `Layer ${state.layer}`;
  document.querySelector("#ratioLabel").textContent = ratioLabel();
  document.querySelector("#layerStrip").innerHTML = m.schedule
    .map((ratio, index) => {
      const klass = ratio === 4 ? "r4" : ratio === 128 ? "r128" : "r0";
      return `<button class="layer-cell ${klass} ${index === state.layer ? "active" : ""}" data-layer="${index}" title="Layer ${index}, R=${ratio}"></button>`;
    })
    .join("");
}

function renderSceneHeader() {
  if (!document.querySelector("#sceneTitle")) return;
  document.querySelector("#sceneTitle").textContent = scene().title;
  document.querySelector("#sceneSubtitle").textContent = scene().subtitle;
  document.querySelector("#backScene").disabled = state.scene === "overview";
}

function ensureSelection() {
  if (state.graphDetail !== "detailed") {
    state.selectedGroup = null;
    state.groupHighlightActive = false;
  }
  if (state.selectedGroup && !currentGroup(state.selectedGroup)) {
    state.selectedGroup = null;
    state.groupHighlightActive = false;
  }
  if (state.selectedGroup) return;
  if (!visibleNode(state.selected) || !visibleNodeIds().includes(state.selected)) {
    state.selected = visibleNodeIds()[0] || "input-ids";
  }
}

function currentGroup(label) {
  return (sceneView().groups || []).find((group) => group.label === label) || null;
}

function isGroupActive(group) {
  if (state.selectedGroup) return state.groupHighlightActive && group.label === state.selectedGroup;
  return group.nodeIds?.includes(state.selected);
}

function isNodeInSelectedGroup(id) {
  return state.groupHighlightActive && Boolean(currentGroup(state.selectedGroup)?.nodeIds?.includes(id));
}

function nodeClass(item) {
  return [
    "node",
    item.doc.category,
    state.selected === item.id ? "selected" : "",
    isNodeInSelectedGroup(item.id) ? "group-selected" : "",
    item.doc.drill ? "drillable" : "",
  ]
    .filter(Boolean)
    .join(" ");
}

function isEdgeActive(edge) {
  if (state.selected === edge.from || state.selected === edge.to) return true;
  if (!state.groupHighlightActive) return false;
  const group = currentGroup(state.selectedGroup);
  return Boolean(group?.nodeIds?.includes(edge.from) || group?.nodeIds?.includes(edge.to));
}

async function buildLayout() {
  if (state.scene === "overview") return buildElkOverviewLayout();
  return buildDagreLayout();
}

function buildDagreLayout() {
  const g = new dagre.graphlib.Graph({ multigraph: true });
  g.setGraph({
    rankdir: "TB",
    align: "UL",
    ranksep: state.scene === "overview" ? 70 : 86,
    nodesep: state.scene === "moe" ? 74 : 50,
    edgesep: 18,
    marginx: 44,
    marginy: 44,
    ranker: "network-simplex",
  });
  g.setDefaultEdgeLabel(() => ({}));

  visibleNodeIds().forEach((id) => {
    const doc = DATA.nodes[id];
    g.setNode(id, { width: nodeWidth(doc), height: nodeHeight(doc) });
  });

  visibleEdges().forEach((edge, index) => {
    g.setEdge(edge.from, edge.to, { type: edge.type, label: edge.label }, `${edge.from}-${edge.to}-${index}`);
  });

  dagre.layout(g);

  const bounds = g.nodes().reduce(
    (acc, id) => {
      const item = g.node(id);
      acc.minX = Math.min(acc.minX, item.x - item.width / 2);
      acc.maxX = Math.max(acc.maxX, item.x + item.width / 2);
      acc.minY = Math.min(acc.minY, item.y - item.height / 2);
      acc.maxY = Math.max(acc.maxY, item.y + item.height / 2);
      return acc;
    },
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity },
  );

  layoutOffset = { x: bounds.minX - 90, y: bounds.minY - 70 };
  lastLayout = {
    width: Math.max(980, bounds.maxX - bounds.minX + 180),
    height: Math.max(900, bounds.maxY - bounds.minY + 140),
  };
  return g;
}

async function buildElkOverviewLayout() {
  const ids = visibleNodeIds();
  const idSet = new Set(ids);
  const edges = visibleEdges().filter((edge) => idSet.has(edge.from) && idSet.has(edge.to));
  const cycles = findCycles(ids, edges);
  if (cycles.length) console.warn("DeepSeek V4 graph contains dependency cycles", cycles);

  const children = ids.map((id) => {
    const doc = DATA.nodes[id];
    return { id, width: nodeWidth(doc), height: nodeHeight(doc) };
  });

  const elkGraph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.spacing.nodeNode": "44",
      "elk.layered.spacing.nodeNodeBetweenLayers": "84",
      "elk.layered.spacing.edgeNodeBetweenLayers": "34",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.padding": "[top=28,left=28,bottom=28,right=28]",
    },
    children,
    edges: edges.map((edge, index) => ({
      id: `${edge.from}-${edge.to}-${index}`,
      sources: [edge.from],
      targets: [edge.to],
      type: edge.type,
      label: edge.label,
    })),
  };

  const layout = await elk.layout(elkGraph);
  const nodeMap = Object.fromEntries(
    (layout.children || []).map((child) => [
      child.id,
      {
        x: (child.x || 0) + child.width / 2,
        y: (child.y || 0) + child.height / 2,
        width: child.width,
        height: child.height,
      },
    ]),
  );
  const edgeRefs = (layout.edges || []).map((edge) => ({
    v: edge.sources[0],
    w: edge.targets[0],
    name: edge.id,
  }));
  const edgeByName = Object.fromEntries(
    (layout.edges || []).map((edge) => [
      edge.id,
      {
        type: edge.type,
        label: edge.label,
        points: edge.sections?.flatMap((section) => [
          section.startPoint,
          ...(section.bendPoints || []),
          section.endPoint,
        ]) || [],
      },
    ]),
  );
  const groupData = (sceneView().groups || []).map((group) => groupBounds(group, nodeMap)).filter(Boolean);

  layoutOffset = { x: 0, y: 0 };
  lastLayout = {
    width: Math.max(980, Math.ceil((layout.width || 1280) + 56)),
    height: Math.max(900, Math.ceil((layout.height || 1280) + 56)),
  };

  return {
    nodes: () => ids,
    node: (id) => nodeMap[id],
    edges: () => edgeRefs,
    edge: (ref) => edgeByName[ref.name],
    groups: groupData,
  };
}

function findCycles(ids, edges) {
  const graph = Object.fromEntries(ids.map((id) => [id, []]));
  edges.forEach((edge) => graph[edge.from]?.push(edge.to));
  const visiting = new Set();
  const visited = new Set();
  const stack = [];
  const cycles = [];

  function visit(id) {
    if (visiting.has(id)) {
      const start = stack.indexOf(id);
      cycles.push([...stack.slice(start), id]);
      return;
    }
    if (visited.has(id)) return;
    visiting.add(id);
    stack.push(id);
    (graph[id] || []).forEach(visit);
    stack.pop();
    visiting.delete(id);
    visited.add(id);
  }

  ids.forEach(visit);
  return cycles;
}

function nodeWidth(doc) {
  const titleLen = String(doc.title || "").length;
  const shapeLen = Math.max(String(resolve(doc.input)).length, String(resolve(doc.output)).length);
  const detailBoost = doc.details ? 12 : 0;
  const formulaBoost = state.showNodeFormula ? 54 : 0;
  const descLen = state.embedNodeDescription ? String(nodeSummary(doc)).length : 0;
  const descBoost = state.embedNodeDescription ? Math.min(190, 72 + descLen * 0.18) : 0;
  const raw = 168 + titleLen * 3.2 + Math.min(shapeLen, 52) * 1.8 + detailBoost + formulaBoost + descBoost;
  const min = state.embedNodeDescription ? 430 : state.showNodeFormula ? 286 : state.scene === "overview" ? 198 : 228;
  const max = state.embedNodeDescription ? 640 : state.showNodeFormula ? 420 : state.scene === "overview" ? 286 : state.scene === "moe" ? 342 : 326;
  return Math.round(Math.max(min, Math.min(max, raw)));
}

function nodeHeight(doc) {
  const width = nodeWidth(doc);
  return measureNodeHeight(doc, width);
}

function measureNodeHeight(doc, width) {
  if (!measureHost) {
    measureHost = document.createElement("div");
    measureHost.className = "node-measure";
    document.body.appendChild(measureHost);
  }
  measureHost.style.width = `${width}px`;
  measureHost.innerHTML = `<div class="node-html">${nodeHtml({ doc })}</div>`;
  return Math.ceil(measureHost.scrollHeight + 2);
}

async function renderGraph() {
  const version = ++renderVersion;
  const g = await buildLayout();
  if (version !== renderVersion) return;
  const svg = d3.select("#graph");
  const zoomLayer = d3.select("#zoomLayer");
  const groupLayer = d3.select("#groupLayer");
  const edgeLayer = d3.select("#edgeLayer");
  const edgeLabelLayer = d3.select("#edgeLabelLayer");
  const nodeLayer = d3.select("#nodeLayer");

  svg.attr("viewBox", `0 0 ${lastLayout.width} ${lastLayout.height}`);

  const edgeData = [];
  g.edges().forEach((edgeRef) => {
    const edgeInfo = g.edge(edgeRef);
    edgeData.push({
      id: edgeRef.name,
      from: edgeRef.v,
      to: edgeRef.w,
      type: edgeInfo.type,
      label: edgeLabel(edgeRef.v, edgeRef.w, edgeInfo),
      points: edgeInfo.points.map((point) => ({ x: point.x - layoutOffset.x, y: point.y - layoutOffset.y })),
      active: isEdgeActive({ from: edgeRef.v, to: edgeRef.w }),
    });
  });

  edgeLayer
    .selectAll("path.edge")
    .data(edgeData, (item) => item.id)
    .join(
      (enter) => enter.append("path").attr("class", "edge"),
      (update) => update,
      (exit) => exit.remove(),
    )
    .attr("class", (item) => `edge ${item.type === "branch" ? "branch" : "main"} ${item.active ? "active" : ""}`)
    .attr("d", (item) => lineForPoints(item.points));

  const edgeLabelJoin = edgeLabelLayer
    .selectAll("g.edge-label")
    .data(edgeData.filter((item) => item.label), (item) => item.id)
    .join(
      (enter) => {
        const group = enter.append("g").attr("class", "edge-label");
        group.append("rect");
        group.append("text").attr("text-anchor", "middle").attr("dominant-baseline", "central");
        return group;
      },
      (update) => update,
      (exit) => exit.remove(),
    )
    .attr("class", (item) => `edge-label ${item.type === "branch" ? "branch" : "main"} ${item.active ? "active" : ""}`)
    .attr("transform", (item) => {
      const point = pathMidpoint(item.points);
      return `translate(${point.x},${point.y})`;
    });

  edgeLabelJoin.select("text").text((item) => item.label);
  edgeLabelJoin.each(function () {
    const group = d3.select(this);
    const box = group.select("text").node().getBBox();
    group
      .select("rect")
      .attr("x", box.x - 7)
      .attr("y", box.y - 3)
      .attr("width", box.width + 14)
      .attr("height", box.height + 6)
      .attr("rx", 5);
  });

  const nodeData = visibleNodeIds().map((id) => {
    const laidOut = g.node(id);
    const doc = DATA.nodes[id];
    return { id, ...laidOut, x: laidOut.x - layoutOffset.x, y: laidOut.y - layoutOffset.y, doc };
  });

  const nodeById = Object.fromEntries(nodeData.map((item) => [item.id, item]));
  const groupData = g.groups || (sceneView().groups || []).map((group) => groupBounds(group, nodeById)).filter(Boolean);

  const groupJoin = groupLayer
    .selectAll("g.graph-group")
    .data(groupData, (item) => item.label)
    .join(
      (enter) => {
        const group = enter.append("g").attr("class", "graph-group");
        group.append("rect");
        group.append("text");
        return group;
      },
      (update) => update,
      (exit) => exit.remove(),
    )
    .attr("class", (item) => `graph-group ${item.category} ${isGroupActive(item) ? "active" : ""} ${state.graphDetail === "detailed" ? "clickable" : ""}`)
    .attr("data-group", (item) => item.label);

  groupJoin
    .select("rect")
    .attr("x", (item) => item.x)
    .attr("y", (item) => item.y)
    .attr("width", (item) => item.width)
    .attr("height", (item) => item.height);

  groupJoin
    .select("text")
    .attr("x", (item) => item.x + 16)
    .attr("y", (item) => item.y + 25)
    .text((item) => item.label);

  prioritizeGraphGroups();

  nodeLayer
    .selectAll("g.node")
    .data(nodeData, (item) => item.id)
    .join(
      (enter) => {
        const group = enter.append("g").attr("class", "node").attr("data-node", (item) => item.id);
        group.append("rect");
        group.append("foreignObject").append("xhtml:div").attr("class", "node-html");
        return group;
      },
      (update) => update,
      (exit) => exit.remove(),
    )
    .attr("class", nodeClass)
    .attr("data-node", (item) => item.id)
    .attr("transform", (item) => `translate(${item.x - item.width / 2},${item.y - item.height / 2})`)
    .each(function (item) {
      const group = d3.select(this);
      group.select("rect").attr("width", item.width).attr("height", item.height);
      group.select("foreignObject").attr("width", item.width).attr("height", item.height);
      group.select(".node-html").html(nodeHtml(item));
    });

  if (!zoomBehavior) {
    zoomBehavior = d3
      .zoom()
      .scaleExtent(ZOOM_EXTENT)
      .wheelDelta((event) => {
        const modeFactor = event.deltaMode === 1 ? 0.06 : event.deltaMode ? 1 : 0.0028;
        return -event.deltaY * modeFactor * (event.ctrlKey ? 7 : 1);
      })
      .on("zoom", (event) => {
        zoomLayer.attr("transform", event.transform);
        updateMinimapViewport();
      });
    svg.call(zoomBehavior);
  }

  renderMinimap(nodeData, edgeData, groupData);

  if (shouldFitAfterLayout) {
    shouldFitAfterLayout = false;
    requestAnimationFrame(fitGraph);
  }
}

function renderMinimap(nodes, edges, groups) {
  const mini = d3.select("#minimapSvg");
  if (mini.empty()) return;

  const width = 180;
  const height = 128;
  const pad = 10;
  const scale = Math.min((width - pad * 2) / lastLayout.width, (height - pad * 2) / lastLayout.height);
  const offsetX = (width - lastLayout.width * scale) / 2;
  const offsetY = (height - lastLayout.height * scale) / 2;
  minimapState = { width, height, scale, offsetX, offsetY };

  const x = (value) => offsetX + value * scale;
  const y = (value) => offsetY + value * scale;

  d3.select("#minimapGroups")
    .selectAll("rect")
    .data(groups, (item) => item.label)
    .join("rect")
    .attr("class", (item) => `minimap-group ${isGroupActive(item) ? "active" : ""}`)
    .attr("x", (item) => x(item.x))
    .attr("y", (item) => y(item.y))
    .attr("width", (item) => Math.max(1, item.width * scale))
    .attr("height", (item) => Math.max(1, item.height * scale));

  d3.select("#minimapEdges")
    .selectAll("path")
    .data(edges, (item) => item.id)
    .join("path")
    .attr("class", "minimap-edge")
    .attr("d", (item) =>
      lineForPoints(item.points.map((point) => ({ x: x(point.x), y: y(point.y) }))),
    );

  d3.select("#minimapNodes")
    .selectAll("rect")
    .data(nodes, (item) => item.id)
    .join("rect")
    .attr("class", (item) => `minimap-node ${item.doc.category}`)
    .attr("x", (item) => x(item.x - item.width / 2))
    .attr("y", (item) => y(item.y - item.height / 2))
    .attr("width", (item) => Math.max(2.4, item.width * scale))
    .attr("height", (item) => Math.max(2.4, item.height * scale))
    .attr("rx", 1.3);

  mini.on("pointerdown", (event) => {
    event.preventDefault();
    panToMinimapEvent(event);
    mini.node().setPointerCapture?.(event.pointerId);
    mini.on("pointermove.minimap", panToMinimapEvent);
  });
  mini.on("pointerup pointerleave pointercancel", (event) => {
    mini.on("pointermove.minimap", null);
    mini.node().releasePointerCapture?.(event.pointerId);
  });

  updateMinimapViewport();
}

function panToMinimapEvent(event) {
  if (!minimapState || !zoomBehavior) return;
  const [mx, my] = d3.pointer(event, d3.select("#minimapSvg").node());
  const graphX = (mx - minimapState.offsetX) / minimapState.scale;
  const graphY = (my - minimapState.offsetY) / minimapState.scale;
  const svgNode = document.querySelector("#graph");
  const rect = svgNode.getBoundingClientRect();
  const transform = d3.zoomTransform(svgNode);
  const viewScaleX = lastLayout.width / rect.width;
  const viewScaleY = lastLayout.height / rect.height;
  const centerX = (rect.width * viewScaleX) / 2;
  const centerY = (rect.height * viewScaleY) / 2;
  d3.select(svgNode).call(
    zoomBehavior.transform,
    d3.zoomIdentity.translate(centerX - graphX * transform.k, centerY - graphY * transform.k).scale(transform.k),
  );
}

function updateMinimapViewport() {
  if (!minimapState) return;
  const svgNode = document.querySelector("#graph");
  if (!svgNode) return;
  const rect = svgNode.getBoundingClientRect();
  const transform = d3.zoomTransform(svgNode);
  const viewScaleX = lastLayout.width / rect.width;
  const viewScaleY = lastLayout.height / rect.height;
  const viewW = (rect.width * viewScaleX) / transform.k;
  const viewH = (rect.height * viewScaleY) / transform.k;
  const viewX = -transform.x / transform.k;
  const viewY = -transform.y / transform.k;

  d3.select("#minimapViewport")
    .attr("x", minimapState.offsetX + viewX * minimapState.scale)
    .attr("y", minimapState.offsetY + viewY * minimapState.scale)
    .attr("width", viewW * minimapState.scale)
    .attr("height", viewH * minimapState.scale);
}

function lineForPoints(points) {
  if (!points || points.length < 2) return "";
  return d3
    .line()
    .x((point) => point.x)
    .y((point) => point.y)
    .curve(d3.curveLinear)(points);
}

function edgeLabel(from, to, edgeInfo = {}) {
  if (edgeInfo.label) return resolve(edgeInfo.label);
  const source = DATA.nodes[from];
  const target = DATA.nodes[to];
  const sourceOutput = source ? resolve(source.output) : "";
  const targetInput = target ? resolve(target.input) : "";
  const label = sourceOutput || targetInput;
  if (!label || label === "undefined") return "";
  if (sourceOutput && targetInput && sourceOutput !== targetInput && sourceOutput.length > 34) {
    return truncate(sourceOutput, 34);
  }
  return truncate(label, 38);
}

function pathMidpoint(points) {
  if (!points || points.length === 0) return { x: 0, y: 0 };
  if (points.length === 1) return points[0];

  const segments = [];
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    const from = points[index - 1];
    const to = points[index];
    const length = Math.hypot(to.x - from.x, to.y - from.y);
    segments.push({ from, to, length });
    total += length;
  }

  let walked = 0;
  const half = total / 2;
  for (const segment of segments) {
    if (walked + segment.length >= half) {
      const ratio = segment.length === 0 ? 0 : (half - walked) / segment.length;
      return {
        x: segment.from.x + (segment.to.x - segment.from.x) * ratio,
        y: segment.from.y + (segment.to.y - segment.from.y) * ratio,
      };
    }
    walked += segment.length;
  }
  return points[Math.floor(points.length / 2)];
}

function nodeHtml(item) {
  const input = truncate(resolve(item.doc.input), state.scene === "overview" ? 34 : 46);
  const output = truncate(resolve(item.doc.output), state.scene === "overview" ? 34 : 46);
  const formula = state.showNodeFormula ? nodeFormulaHtml(item.doc) : "";
  const description = state.embedNodeDescription ? nodeDescriptionHtml(item.doc) : "";
  return `
    <div class="node-title-row">
      <span>${escapeHtml(item.doc.category)}</span>
      <strong>${escapeHtml(nodeTitle(item.doc))}</strong>
    </div>
    ${formula}
    ${description}
    <div class="node-shape">${escapeHtml(input)} -> ${escapeHtml(output)}</div>
  `;
}

function nodeDescriptionHtml(doc) {
  const source = nodeSummary(doc);
  if (!source) return "";
  const text = String(source).replace(/\s+/g, " ").trim();
  return `<p class="node-description">${escapeHtml(text)}</p>`;
}

function nodeFormulaHtml(doc) {
  const formula = firstFormula(localizedFormula(doc));
  if (!formula?.latex) return "";
  const title = formula.title ? `<b>${escapeHtml(resolve(formula.title))}</b>` : "";
  return `<div class="node-formula">${title}${renderLatex(compactLatex(resolve(formula.latex)), { displayMode: false })}</div>`;
}

function formulaTitle(title) {
  const resolved = resolve(title);
  if (!isEnglish()) return resolved;
  return I18N.formulaTitle?.en?.[resolved] || resolved;
}

function localizedFormula(doc) {
  const formulas = doc?.details?.formula;
  const items = formulas ? (Array.isArray(formulas) ? formulas : [formulas]) : [];
  if (!items.length) return [];
  const ownMeta = nodeText(doc)?.formula || [];
  const fallbackMeta =
    isEnglish() && !ownMeta.length
      ? (nodeText(doc, "ko")?.formula || []).map((item) => ({
          title: item.title ? formulaTitle(item.title) : undefined,
        }))
      : [];
  const meta = ownMeta.length ? ownMeta : fallbackMeta;
  return items.map((item, index) => {
    const formula = typeof item === "string" ? { latex: item } : item;
    const extra = meta[index] || {};
    return {
      ...formula,
      title: extra.title ?? formula.title,
      note: extra.note ?? formula.note,
    };
  });
}

function compactLatex(source) {
  return String(source)
    .replaceAll(String.raw`\qquad`, String.raw`\,`)
    .replaceAll(String.raw`\quad`, String.raw`\,`)
    .replaceAll(String.raw`\;`, String.raw`\,`);
}

function firstFormula(value) {
  if (!value) return null;
  const item = Array.isArray(value) ? value[0] : value;
  return typeof item === "string" ? { latex: item } : item;
}

function groupBounds(group, nodeById) {
  const members = group.nodeIds.map((id) => nodeById[id]).filter(Boolean);
  if (!members.length) return null;
  const padX = 34;
  const padTop = 44;
  const padBottom = 24;
  const minX = Math.min(...members.map((item) => item.x - item.width / 2));
  const maxX = Math.max(...members.map((item) => item.x + item.width / 2));
  const minY = Math.min(...members.map((item) => item.y - item.height / 2));
  const maxY = Math.max(...members.map((item) => item.y + item.height / 2));
  return {
    label: group.label,
    category: group.category,
    nodeIds: group.nodeIds,
    x: minX - padX,
    y: minY - padTop,
    width: maxX - minX + padX * 2,
    height: maxY - minY + padTop + padBottom,
  };
}

function renderDetail() {
  if (state.selectedGroup) {
    renderGroupDetail(currentGroup(state.selectedGroup));
    return;
  }
  const doc = DATA.nodes[state.selected];
  if (!doc) return;
  document.querySelector(".detail-shapes").hidden = false;
  document.querySelector("#detailCategory").textContent = doc.category;
  document.querySelector("#detailTitle").textContent = nodeTitle(doc);
  document.querySelector("#detailCards").innerHTML = renderDetailCards(nodeDetails(doc));
  document.querySelector("#inputShape").textContent = resolve(doc.input);
  document.querySelector("#outputShape").textContent = resolve(doc.output);
  document.querySelector("#paramList").innerHTML = Object.entries(doc.params || {})
    .map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(resolve(value))}</dd></div>`)
    .join("");
  document.querySelector("#noteList").innerHTML = "";
  document.querySelector("#noteList").hidden = true;
  document.querySelector("#sourceList").innerHTML = doc.sources
    .map((source) => `<a href="${source.url}" target="_blank" rel="noreferrer">${source.label}</a>`)
    .join("");

  const drill = document.querySelector("#drillButton");
  drill.hidden = true;
  delete drill.dataset.scene;
}

function renderGroupDetail(group) {
  if (!group) return;
  const docs = group.nodeIds.map((id) => DATA.nodes[id]).filter(Boolean);
  document.querySelector(".detail-shapes").hidden = true;
  document.querySelector("#detailCategory").textContent = `${group.category} ${t("group")}`;
  document.querySelector("#detailTitle").textContent = group.label;
  document.querySelector("#detailCards").innerHTML = renderGroupCards(group, docs);
  document.querySelector("#inputShape").textContent = docs.length ? resolve(docs[0].input) : "";
  document.querySelector("#outputShape").textContent = docs.length ? resolve(docs[docs.length - 1].output) : "";
  document.querySelector("#paramList").innerHTML = `<div><dt>${escapeHtml(t("nodes"))}</dt><dd>${docs.length}</dd></div>`;
  document.querySelector("#noteList").innerHTML = "";
  document.querySelector("#noteList").hidden = true;
  document.querySelector("#sourceList").innerHTML = collectSources(docs)
    .map((source) => `<a href="${source.url}" target="_blank" rel="noreferrer">${source.label}</a>`)
    .join("");

  const drill = document.querySelector("#drillButton");
  drill.hidden = true;
  delete drill.dataset.scene;
}

function renderGroupCards(group, docs) {
  const prose = [...groupPurpose(group), ...groupNarrative(group, docs)];
  return `
    <section class="group-detail-card">
      <h3>${escapeHtml(t("description"))}</h3>
      ${prose.map((paragraph) => `<p>${paragraph}</p>`).join("")}
    </section>`;
}

function groupPurpose(group) {
  const purpose = I18N.groupPurpose?.[state.lang]?.[group.label] || I18N.groupPurpose?.[state.lang]?.default;
  if (!purpose) return [];
  const items = Array.isArray(purpose) ? purpose : [purpose];
  return items.map((item) => String(item).replaceAll("{group}", escapeHtml(group.label)));
}
function groupNarrative(group, docs) {
  const n = (id, label) => nodeInline(id, DATA.nodes[id], label);
  const narratives = I18N.groupNarrative?.[state.lang] || {};
  const narrative = narratives[group.label] || narratives.default;
  if (typeof narrative === "function") return narrative(group, docs, n, { escapeHtml, resolve });
  if (Array.isArray(narrative)) return narrative;
  return [];
}
function nodeInline(id, doc = DATA.nodes[id], label = null) {
  const text = label || doc?.title || id;
  const title = doc?.title || id;
  return `<button class="node-ref" data-group-node="${escapeHtml(id)}" title="${escapeHtml(title)}">${escapeHtml(text)}</button>`;
}

function verifyGroupNarrativeLinks() {
  if (!VERIFY_GROUP_NARRATIVE_LINKS) return;
  const failures = [];
  const originalLang = state.lang;
  ["ko", "en"].forEach((lang) => {
    state.lang = lang;
    Object.entries(DATA.scenes).forEach(([sceneId, item]) => {
      const detailedView = item.views?.detailed || item;
      (detailedView.groups || []).forEach((group) => {
        // group.nodeIds is the source of truth; every prose paragraph must link
        // every member exactly once so group explanations stay navigable.
        const docs = group.nodeIds.map((id) => DATA.nodes[id]).filter(Boolean);
        const html = renderGroupCards(group, docs);
        const linkedIds = new Set([...html.matchAll(/data-group-node="([^"]+)"/g)].map((match) => match[1]));
        const missingDocs = group.nodeIds.filter((id) => !DATA.nodes[id]);
        const missingLinks = group.nodeIds.filter((id) => DATA.nodes[id] && !linkedIds.has(id));
        const strayLinks = [...linkedIds].filter((id) => !group.nodeIds.includes(id));
        if (missingDocs.length || missingLinks.length || strayLinks.length) {
          failures.push({
            scene: `${lang}:${sceneId}`,
            group: group.label,
            missingDocs,
            missingLinks,
            strayLinks,
          });
        }
      });
    });
  });
  state.lang = originalLang;

  if (!failures.length) return;
  const report = failures
    .map((item) => {
      const parts = [`${item.scene} / ${item.group}`];
      if (item.missingDocs.length) parts.push(`unknown nodes: ${item.missingDocs.join(", ")}`);
      if (item.missingLinks.length) parts.push(`missing links: ${item.missingLinks.join(", ")}`);
      if (item.strayLinks.length) parts.push(`stray links: ${item.strayLinks.join(", ")}`);
      return parts.join(" | ");
    })
    .join("\n");
  throw new Error(`Group narrative link verification failed:\n${report}`);
}

function collectSources(docs) {
  const seen = new Set();
  return docs.flatMap((doc) => doc.sources || []).filter((source) => {
    if (!source?.url || seen.has(source.url)) return false;
    seen.add(source.url);
    return true;
  });
}

function renderPanelState() {
  const panel = document.querySelector(".info-panel");
  if (!panel) return;
  panel.classList.toggle("closed", !state.detailOpen);
  if (state.detailOpen) requestAnimationFrame(applyDetailPanelPosition);
}

function detailPanelParent(panel) {
  return panel.offsetParent || document.querySelector(".graph-shell") || document.documentElement;
}

function detailPanelCurrentPosition(panel) {
  const parentRect = detailPanelParent(panel).getBoundingClientRect();
  const panelRect = panel.getBoundingClientRect();
  return {
    x: panelRect.left - parentRect.left,
    y: panelRect.top - parentRect.top,
  };
}

function constrainDetailPanelPosition(panel, position) {
  const parent = detailPanelParent(panel);
  const margin = 10;
  const maxX = Math.max(margin, parent.clientWidth - panel.offsetWidth - margin);
  const maxY = Math.max(margin, parent.clientHeight - panel.offsetHeight - margin);
  return {
    x: Math.min(Math.max(position.x, margin), maxX),
    y: Math.min(Math.max(position.y, margin), maxY),
  };
}

function applyDetailPanelPosition() {
  const panel = document.querySelector(".info-panel");
  if (!panel || !state.detailPanelPosition || !state.detailOpen) return;
  state.detailPanelPosition = constrainDetailPanelPosition(panel, state.detailPanelPosition);
  panel.style.left = `${state.detailPanelPosition.x}px`;
  panel.style.top = `${state.detailPanelPosition.y}px`;
  panel.style.right = "auto";
  panel.style.bottom = "auto";
}

function beginDetailPanelDrag(event) {
  if (event.button !== 0 || event.target.closest("button, a, input, textarea, select")) return;
  const panel = event.currentTarget.closest(".info-panel");
  if (!panel) return;
  const startPosition = constrainDetailPanelPosition(panel, state.detailPanelPosition || detailPanelCurrentPosition(panel));
  state.detailPanelPosition = startPosition;
  applyDetailPanelPosition();

  detailPanelDrag = {
    pointerId: event.pointerId,
    startClientX: event.clientX,
    startClientY: event.clientY,
    startX: startPosition.x,
    startY: startPosition.y,
  };
  panel.classList.add("dragging");
  event.currentTarget.setPointerCapture?.(event.pointerId);
  event.preventDefault();
}

function moveDetailPanelDrag(event) {
  if (!detailPanelDrag || event.pointerId !== detailPanelDrag.pointerId) return;
  const panel = document.querySelector(".info-panel");
  if (!panel) return;
  state.detailPanelPosition = constrainDetailPanelPosition(panel, {
    x: detailPanelDrag.startX + event.clientX - detailPanelDrag.startClientX,
    y: detailPanelDrag.startY + event.clientY - detailPanelDrag.startClientY,
  });
  applyDetailPanelPosition();
}

function endDetailPanelDrag(event) {
  if (!detailPanelDrag || event.pointerId !== detailPanelDrag.pointerId) return;
  document.querySelector(".info-panel")?.classList.remove("dragging");
  event.currentTarget.releasePointerCapture?.(event.pointerId);
  detailPanelDrag = null;
}

function renderSelectionState() {
  d3.selectAll("path.edge").attr(
    "class",
    (item) => `edge ${item.type === "branch" ? "branch" : "main"} ${isEdgeActive(item) ? "active" : ""}`,
  );
  d3.selectAll("g.edge-label").attr(
    "class",
    (item) => `edge-label ${item.type === "branch" ? "branch" : "main"} ${isEdgeActive(item) ? "active" : ""}`,
  );
  d3.selectAll("g.graph-group").attr(
    "class",
    (item) => `graph-group ${item.category} ${isGroupActive(item) ? "active" : ""} ${state.graphDetail === "detailed" ? "clickable" : ""}`,
  );
  prioritizeGraphGroups();
  d3.selectAll("#minimapGroups rect").attr("class", (item) => `minimap-group ${isGroupActive(item) ? "active" : ""}`);
  d3.selectAll("g.node").attr(
    "class",
    nodeClass,
  );
}

function prioritizeGraphGroups() {
  d3.selectAll("g.graph-group").sort((a, b) => groupStackRank(a) - groupStackRank(b));
}

function groupStackRank(group) {
  const selectedNodeRank = group.nodeIds?.includes(state.selected) ? 100 : 0;
  const selectedGroupRank = state.groupHighlightActive && state.selectedGroup === group.label ? 200 : 0;
  const area = Math.max(1, (group.width || 0) * (group.height || 0));
  const specificityRank = 1 / area;
  return selectedGroupRank + selectedNodeRank + specificityRank;
}

function renderDetailCards(details) {
  if (!details) return "";
  return Object.entries(details)
    .filter(([, value]) => value)
    .map(([label, value]) => {
      if (label === "formula") {
        return `<section class="formula-card"><h3>${escapeHtml(t(label))}</h3>${renderFormulaList(value)}</section>`;
      }
      const items = Array.isArray(value) ? value : [value];
      const body =
        items.length === 1
          ? `<p>${escapeHtml(resolve(items[0]))}</p>`
          : `<ul>${items.map((item) => `<li>${escapeHtml(resolve(item))}</li>`).join("")}</ul>`;
      return `<section><h3>${escapeHtml(t(label) || label)}</h3>${body}</section>`;
    })
    .join("");
}

function renderFormulaList(value) {
  const items = Array.isArray(value) ? value : [{ latex: value }];
  return items
    .map((item) => {
      const formula = typeof item === "string" ? { latex: item } : item;
      const title = formula.title ? `<h4>${escapeHtml(resolve(formula.title))}</h4>` : "";
      const note = formula.note ? `<p>${escapeHtml(resolve(formula.note))}</p>` : "";
      return `<div class="formula-block">${title}<div class="formula">${renderLatex(resolve(formula.latex || ""))}</div>${note}</div>`;
    })
    .join("");
}

function renderLatex(source, options = {}) {
  if (!source) return "";
  if (window.katex?.renderToString) {
    return window.katex.renderToString(source, {
      displayMode: options.displayMode ?? true,
      throwOnError: false,
      strict: false,
      trust: false,
    });
  }
  return `<code>${escapeHtml(source)}</code>`;
}

function openScene(nextScene) {
  if (!DATA.scenes[nextScene]) return;
  state.scene = nextScene;
  state.selectedGroup = null;
  state.groupHighlightActive = false;
  state.selected = sceneView().nodeIds.find(visibleNode) || "input-ids";
  render(true);
}

function fitGraph() {
  const svgNode = document.querySelector("#graph");
  if (!svgNode || !zoomBehavior) return;
  const rect = svgNode.getBoundingClientRect();
  const userScaleX = lastLayout.width / rect.width;
  const userScaleY = lastLayout.height / rect.height;
  const visibleWidth = rect.width * userScaleX;
  const visibleHeight = rect.height * userScaleY;
  const widthFillScale = (rect.width * lastLayout.height) / (lastLayout.width * rect.height);
  const scale =
    state.scene === "overview"
      ? Math.max(0.7, Math.min(3.6, widthFillScale * 0.82))
      : Math.min(1.05, (visibleWidth / lastLayout.width) * 0.9);
  const tx = (visibleWidth - lastLayout.width * scale) / 2;
  const ty = state.scene === "overview" ? 58 : Math.max(24, (visibleHeight - lastLayout.height * scale) / 2);
  d3.select(svgNode)
    .transition()
    .duration(220)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

function zoomBy(factor) {
  if (!zoomBehavior) return;
  d3.select("#graph").transition().duration(160).call(zoomBehavior.scaleBy, factor);
}

function centerNode(id) {
  if (!zoomBehavior) return;
  let target = null;
  d3.selectAll("g.node").each((item) => {
    if (item.id === id) target = item;
  });
  if (!target) return;
  const svgNode = document.querySelector("#graph");
  const rect = svgNode.getBoundingClientRect();
  const transform = d3.zoomTransform(svgNode);
  const viewScaleX = lastLayout.width / rect.width;
  const viewScaleY = lastLayout.height / rect.height;
  const centerX = (rect.width * viewScaleX) / 2;
  const centerY = (rect.height * viewScaleY) / 2;
  d3.select(svgNode)
    .transition()
    .duration(180)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(centerX - target.x * transform.k, centerY - target.y * transform.k).scale(transform.k));
}

function focusGroupNarrativeNode(id) {
  state.selected = id;
  state.groupHighlightActive = false;
  renderSelectionState();
  centerNode(id);
}

function truncate(value, max) {
  const text = String(value);
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

document.addEventListener("click", (event) => {
  const modelButton = event.target.closest("[data-model]");
  if (modelButton) {
    state.model = modelButton.dataset.model;
    state.layer = Math.min(state.layer, model().schedule.length - 1);
    document.querySelectorAll("[data-model]").forEach((button) => {
      button.classList.toggle("active", button.dataset.model === state.model);
    });
    render(true);
    return;
  }

  const langButton = event.target.closest("[data-lang-select]");
  if (langButton) {
    state.lang = langButton.dataset.langSelect === "en" ? "en" : "ko";
    window.localStorage?.setItem("dsv4-lang", state.lang);
    render(true);
    return;
  }

  const layerButton = event.target.closest("[data-layer]");
  if (layerButton) {
    state.layer = Number(layerButton.dataset.layer);
    render(true);
    return;
  }

  const presetButton = event.target.closest("[data-layer-preset]");
  if (presetButton) {
    state.layer = Number(presetButton.dataset.layerPreset);
    render(true);
    return;
  }

  const sceneButton = event.target.closest("[data-scene-select]");
  if (sceneButton) {
    if (sceneButton.dataset.sceneSelect === "indexer" && attentionMode() !== "csa") {
      state.layer = 2;
    }
    openScene(sceneButton.dataset.sceneSelect);
    return;
  }

  const detailButton = event.target.closest("[data-detail-select]");
  if (detailButton) {
    state.graphDetail = detailButton.dataset.detailSelect;
    if (state.graphDetail !== "detailed") {
      state.selectedGroup = null;
      state.groupHighlightActive = false;
    }
    render(true);
    return;
  }

  const groupNodeButton = event.target.closest("[data-group-node]");
  if (groupNodeButton) {
    focusGroupNarrativeNode(groupNodeButton.dataset.groupNode);
    return;
  }

  const graphNode = event.target.closest("[data-node]");
  if (graphNode) {
    state.selected = graphNode.dataset.node;
    state.selectedGroup = null;
    state.groupHighlightActive = false;
    state.detailOpen = true;
    renderDetail();
    renderPanelState();
    renderSelectionState();
    return;
  }

  const graphGroup = event.target.closest("[data-group]");
  if (graphGroup && state.graphDetail === "detailed") {
    state.selected = null;
    state.selectedGroup = graphGroup.dataset.group;
    state.groupHighlightActive = true;
    state.detailOpen = true;
    renderDetail();
    renderPanelState();
    renderSelectionState();
  }
});

document.addEventListener("pointerover", (event) => {
  const groupNodeButton = event.target.closest("[data-group-node]");
  if (!groupNodeButton || groupNodeButton.contains(event.relatedTarget)) return;
  focusGroupNarrativeNode(groupNodeButton.dataset.groupNode);
});

document.querySelector("#layerSlider")?.addEventListener("input", (event) => {
  state.layer = Number(event.target.value);
  render(true);
});

document.querySelector("#formulaToggle")?.addEventListener("change", (event) => {
  state.showNodeFormula = event.currentTarget.checked;
  shouldFitAfterLayout = true;
  render(true);
});

document.querySelector("#embedDescToggle")?.addEventListener("change", (event) => {
  state.embedNodeDescription = event.currentTarget.checked;
  shouldFitAfterLayout = true;
  render(true);
});

document.querySelector("#darkModeToggle")?.addEventListener("change", (event) => {
  state.theme = event.currentTarget.checked ? "dark" : "light";
  window.localStorage?.setItem("dsv4-theme", state.theme);
  applyTheme();
  renderModePicker();
});

document.querySelector("#zoomIn")?.addEventListener("click", () => zoomBy(1.2));
document.querySelector("#zoomOut")?.addEventListener("click", () => zoomBy(0.84));
document.querySelector("#zoomFit")?.addEventListener("click", fitGraph);
document.querySelector(".info-head")?.addEventListener("pointerdown", beginDetailPanelDrag);
document.querySelector(".info-head")?.addEventListener("pointermove", moveDetailPanelDrag);
document.querySelector(".info-head")?.addEventListener("pointerup", endDetailPanelDrag);
document.querySelector(".info-head")?.addEventListener("pointercancel", endDetailPanelDrag);
document.querySelector("#closeDetail")?.addEventListener("click", () => {
  state.detailOpen = false;
  renderPanelState();
});
document.querySelector("#backScene")?.addEventListener("click", () => openScene("overview"));
document.querySelector("#drillButton")?.addEventListener("click", (event) => {
  if (event.currentTarget.dataset.scene) openScene(event.currentTarget.dataset.scene);
});
window.addEventListener("resize", applyDetailPanelPosition);

verifyGroupNarrativeLinks();
render(true);
document.fonts?.ready.then(() => render(true));
