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

function nodeSummary(doc) {
  if (!doc) return "";
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
  if (!isEnglish()) return doc.details;
  return {
    why: englishNodeDescription(doc),
    runtime: englishNodeRuntime(doc),
    formula: doc.details?.formula,
  };
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
  const formula = firstFormula(doc.details?.formula);
  if (!formula?.latex) return "";
  const title = formula.title ? `<b>${escapeHtml(resolve(formula.title))}</b>` : "";
  return `<div class="node-formula">${title}${renderLatex(compactLatex(resolve(formula.latex)), { displayMode: false })}</div>`;
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
  if (isEnglish()) {
    return [I18N.groupPurpose?.en?.[group.label] || `${escapeHtml(group.label)} collects related graph nodes that work together as one model subsystem.`];
  }
  const purpose = {
    "Model entry (once)": "Decoder layer 반복에 들어가기 전, token id를 dense hidden state로 바꾸고 DeepSeek V4의 기본 운반 형식인 4-lane residual stream을 여는 진입 구간입니다. 여기서 mHC(manifold / hyper-connection 계열의 4-lane residual 구조)가 처음 등장하며, 이후 모든 attention과 MoE sublayer는 단일 `[B,S,D]` stream이 아니라 `[B,S,4,D]` lane state를 읽고 다시 쓰는 방식으로 동작합니다. 이 구간의 의도는 입력 token 정보를 일반 embedding으로 끝내지 않고, layer 사이를 더 안정적으로 이동할 수 있는 다중 residual lane 상태로 올려놓는 것입니다.",
    "mHC controller + read path": "Attention sublayer 앞에서 mHC(manifold / hyper-connection 계열 residual lane controller)가 4개 lane 전체를 보고 read/write/mix coefficient를 만드는 제어 구간입니다. attention 자체를 lane마다 4번 실행하면 compute가 커지므로, controller가 먼저 어떤 lane 조합을 읽을지 정하고 data path는 그 결과를 단일 `[B,S,D]` hidden stream으로 접습니다. 동시에 post coefficient와 doubly-stochastic comb matrix도 준비해 두기 때문에, 이 구간은 attention 입력 생성뿐 아니라 attention 이후 residual lane 안정화까지 미리 설계하는 역할을 합니다.",
    "Attention Q/KV paths": "Long-context attention의 핵심 경로로, query low-rank projection, shared KV, SWA(Sliding Window Attention) local window, compressed cache, sparse attention, grouped output projection이 한 흐름 안에서 연결됩니다. 의도는 1M token context를 dense KV attention처럼 모두 읽지 않고, 최근 token은 정확하게 보존하고 오래된 token은 compressed entry와 선택된 block만 읽게 만드는 것입니다. 그래서 이 그룹은 단순한 Q/K/V 생성이 아니라 memory 절감, cache layout, sparse candidate selection, output 복원을 한 번에 보여주는 attention 본체에 가깝습니다.",
    "SWA cache write path": "SWA(Sliding Window Attention) cache는 최근 token을 압축하지 않은 KV 형태로 보존하는 local memory path입니다. compressed attention이 오래된 문맥을 저렴하게 다루는 동안, 최근 128 token 근처의 세부 정보는 손실 없이 남겨야 다음 decode step에서 local coherence가 유지됩니다. 이 구간은 모델 수식만으로 끝나는 부분이 아니라 runtime이 ring buffer처럼 최신 KV를 갱신하고 오래된 local entry를 밀어내야 하는 cache 관리 책임까지 드러냅니다.",
    "KV compressor + tail state": "긴 문맥을 compressed KV entry로 바꿔 memory footprint를 줄이는 경로입니다. C4A류 compression은 일정 token block을 모아 하나의 cache entry로 만들지만, streaming decode에서는 항상 block boundary가 딱 맞지 않으므로 tail state가 아직 완성되지 않은 remainder token의 projection과 score state를 보관합니다. 이 그룹의 의도는 오래된 문맥을 버리는 것이 아니라, attention이 나중에 다시 읽을 수 있는 더 작은 KV 단위로 재표현하고 runtime chunking 문제까지 함께 처리하는 것입니다.",
    "Lightning indexer": "CSA(Compressed Sparse Attention 계열) layer에서 많은 compressed KV block 중 실제 attention이 볼 후보만 빠르게 고르는 retrieval gate입니다. Lightning Indexer는 value sum을 직접 수행하는 attention이 아니라, compressed block ranking을 위해 별도 query와 별도 index cache를 사용하는 경량 검색 경로입니다. long-context에서 compute 병목은 cache를 저장하는 것뿐 아니라 어떤 block을 읽을지 고르는 데서도 생기므로, 이 그룹은 attention 전에 candidate set을 줄여 sparse attention의 비용을 제한하는 역할을 합니다.",
    "mHC attention residual mixing": "Attention output을 4-lane residual state에 다시 쓰는 mHC writeback 구간입니다. mHC에서는 기존 residual lane을 그대로 다음 layer로 넘기는 항과 attention output을 새로 주입하는 항이 분리되어 있고, comb matrix가 기존 lane 사이의 residual transport를 담당합니다. 이 설계의 의도는 sublayer output이 residual stream을 덮어쓰는 느낌이 아니라, 기존 lane 신호를 안정적으로 운반하면서 필요한 만큼만 attention update를 각 lane에 분배하는 것입니다.",
    "mHC MoE controller + read path": "MoE(Mixture-of-Experts) sublayer 앞에서 4-lane residual state를 단일 FFN input stream으로 읽는 FFN-side mHC controller 구간입니다. attention 쪽과 같은 read/write/mix 구조를 쓰지만 parameter set은 분리되어 있어, attention에 적합한 lane 읽기 정책과 sparse FFN에 적합한 lane 읽기 정책을 따로 학습할 수 있습니다. 이 그룹의 의도는 MoE compute를 lane마다 반복하지 않으면서도, MoE가 현재 residual lane 전체를 보고 필요한 정보를 골라 들어가게 만드는 것입니다.",
    "MoE routing + SwiGLU experts": "DeepSeek V4의 sparse capacity를 담당하는 MoE(Mixture-of-Experts) 본체입니다. token마다 일부 routed expert만 실행해 token당 compute를 제한하고, shared expert는 항상 더해 공통 변환 경로를 유지합니다. 이 그룹은 router score와 expert id 선택, selected token dispatch, routed SwiGLU expert 계산, shared expert 결합까지 이어지며, 큰 parameter capacity를 실제 실행 비용과 분리하려는 MoE의 핵심 의도를 보여줍니다.",
    "mHC MoE residual mixing": "MoE 결과를 다시 4-lane residual stream으로 되돌리는 FFN-side writeback 구간입니다. routed/shared expert가 만든 update는 post coefficient로 각 lane에 주입되고, 기존 residual lane은 comb matrix를 통해 별도로 이동합니다. attention writeback과 같은 구조를 MoE 뒤에도 반복하는 이유는 sparse expert update가 강하게 들어와도 residual lane gradient와 signal propagation이 갑자기 불안정해지지 않도록 하기 위해서입니다.",
    "Final output + MTP": "반복 decoder stack이 모두 끝난 뒤 main LM head와 MTP(Multi-Token Prediction) branch로 나뉘는 최종 output 구간입니다. LM head는 매 layer마다 실행되는 것이 아니라 최종 state 뒤에서 last token에 대해서만 vocabulary logits를 계산하고, MTP branch는 별도의 auxiliary prediction path로 final hidden과 token embedding 정보를 다시 결합합니다. 이 그룹은 모델 내부 반복 계산과 최종 prediction head의 경계를 분명히 하며, main autoregressive decode와 보조 next-token prediction이 어디서 갈라지는지 보여줍니다.",
    "mHC attention controller + read path": "Attention 상세 scene에서 mHC controller와 read data path만 확대해 보여주는 구간입니다. mHC는 여기서 manifold / hyper-connection residual lane 구조를 뜻하며, controller path는 coefficient를 만들고 data path는 실제 tensor를 읽는 식으로 역할이 분리됩니다. attention 계산 자체보다 중요한 점은 attention이 어떤 residual lane 조합을 입력으로 받는지, 그리고 이후 writeback을 위해 post/comb coefficient가 어떻게 함께 준비되는지입니다.",
    "mHC attention entry/exit": "Attention sublayer를 mHC wrapper 관점에서 감싸 보여주는 입구와 출구입니다. wrapper의 입구에서는 `[B,S,4,D]` lane state를 attention용 `[B,S,D]` stream으로 읽고, 출구에서는 attention output과 기존 residual lane transport를 합쳐 다시 `[B,S,4,D]`로 복원합니다. 이 그룹은 attention kernel의 세부보다 sublayer boundary에서 hidden state 형식이 어떻게 바뀌고 다시 돌아오는지 이해하는 데 초점을 둡니다.",
    "Query LoRA + RoPE": "Attention query를 저비용으로 만들면서 position 정보를 필요한 slice에만 주입하는 query 생성 구간입니다. LoRA(low-rank adaptation식 저랭크 projection) 형태의 query latent는 projection 비용과 parameter 부담을 줄이고, RMSNorm은 main attention과 indexer가 공유할 query latent의 scale을 안정화합니다. RoPE(Rotary Position Embedding)는 query의 position-sensitive slice에만 적용되어, content dimension과 position phase를 구분한 채 long-context attention score를 만들 수 있게 합니다.",
    "Shared KV + SWA": "Head별 KV cache를 모두 들고 가지 않고 shared KV representation과 SWA(Sliding Window Attention) local window를 함께 쓰는 memory 절감 구간입니다. shared KV는 multi-head query가 하나의 compact KV entry를 공유하게 만들어 cache 크기를 줄이고, SWA window는 최근 128 token을 uncompressed로 남겨 가까운 문맥의 정확도를 보존합니다. 이 그룹의 의도는 오래된 context는 압축/선택으로 싸게 다루되, 방금 생성된 local context는 손실 없이 attention 후보에 유지하는 균형입니다.",
    "Compressed selection": "Attention이 읽을 compressed memory 후보를 정하는 선택 구간입니다. CSA 계열에서는 Lightning Indexer가 compressed block을 sparse retrieval하고, HCA 계열에서는 더 강하게 압축된 block 전체를 후보로 넣는 식으로 layer schedule에 따라 선택 방식이 달라집니다. 이 그룹은 compressed cache가 저장되어 있다는 사실보다, 각 layer가 그 compressed memory를 어떤 정책으로 attention candidate set에 포함시키는지를 보여주는 데 의미가 있습니다.",
    "Core attention kernel": "이미 선택된 KV 후보 위에서 실제 sparse attention을 수행하는 kernel 구간입니다. dense context 전체가 아니라 selected cache entry만 gather한 뒤, QK score, causal/window mask, attention sink, softmax, value sum을 순서대로 적용합니다. 이 그룹의 의도는 long-context attention이 결국 같은 attention 수식을 쓰더라도, 수식이 적용되는 domain이 전체 context가 아니라 선택된 cache subset이라는 점을 명확히 보여주는 것입니다.",
    "KV sharing output fix": "Shared KV 설계 때문에 output 쪽에서 필요한 value phase 보정을 처리하는 구간입니다. key score를 만들기 위해 KV representation 일부에는 RoPE phase가 들어가지만, value sum 결과는 position phase가 그대로 섞이면 의미가 어색해질 수 있으므로 inverse RoPE 계열 보정이 필요합니다. 이후 grouped low-rank output projection이 attention head 결과를 residual hidden size로 되돌려 mHC writeback이 받을 수 있는 `[B,S,D]` stream으로 정리합니다.",
    "SWA window cache": "Cache 상세 scene에서 SWA(Sliding Window Attention) local uncompressed memory를 담당하는 구간입니다. compressed cache가 오래된 문맥을 줄여 저장하는 동안, SWA window는 최근 token의 정확한 KV를 유지해 short-range dependency를 안정적으로 처리합니다. 이 그룹은 모델 graph와 runtime cache layout이 만나는 부분으로, prefill에서는 마지막 window를 남기고 decode에서는 ring buffer처럼 새 token KV를 쓰는 책임을 보여줍니다.",
    "Compressor projections": "Compression이 단순 평균이 아니라 learned projection과 learned gate score 위에서 일어난다는 점을 보여주는 구간입니다. compressor KV projection은 compressed entry의 내용이 될 vector를 만들고, gate projection은 여러 token 중 어떤 정보를 더 강하게 남길지 정하는 pooling score를 만듭니다. APE 계열 block-local position signal까지 더해지므로, 이 구간의 의도는 압축 전부터 content path와 selection weight path를 분리해 더 정보량 있는 compressed memory를 만드는 것입니다.",
    "Tail / cutoff runtime state": "Streaming 입력을 compression block 단위로 자를 때 필요한 runtime state 구간입니다. compressor는 window 8, stride 4 같은 block 규칙을 쓰기 때문에 현재 chunk 끝에 남은 token이 아직 완성 block을 이루지 못할 수 있고, tail state는 그 remainder token의 projection과 score를 다음 호출까지 보관합니다. 이 그룹은 모델 수식만이 아니라 dynamic decode runtime에서 if/cutoff/state carry가 왜 필요한지 보여줍니다.",
    "Block pooling": "여러 token representation을 하나의 compressed KV entry로 합치는 실제 pooling 구간입니다. c4a식 overlap transform은 stride보다 넓은 token span을 보게 해 boundary 손실을 줄이고, softmax-gated pooling은 learned score로 중요한 token/channel 정보를 더 크게 반영합니다. 이 그룹의 의도는 오래된 문맥을 균등하게 뭉개는 것이 아니라, block 내부에서 중요도가 높은 정보를 weighted sum으로 남겨 attention cache entry로 재표현하는 것입니다.",
    "Compressed entry write": "Pooling된 compressed representation을 attention cache에서 읽을 수 있는 정식 cache entry로 마무리하는 구간입니다. compressed block은 내부 token 각각의 position을 모두 유지하지 않고 anchor position을 대표 위치로 쓰며, RoPE와 normalization을 거쳐 key로 읽힐 수 있는 형태가 됩니다. slot mapping과 cache write는 SWA prefix 뒤에 compressed suffix를 배치하므로, 이 그룹은 compressed memory가 실제 attention id space에 편입되는 마지막 단계입니다.",
    "Attention consumer": "Cache/compressor가 만든 결과가 attention kernel로 들어가는 소비 지점입니다. SWA cache, compressed KV cache, indexer가 고른 selected block이 서로 다른 방식으로 만들어졌더라도, attention kernel 입장에서는 selected KV gather라는 하나의 입력 인터페이스로 합쳐집니다. 이 그룹은 생산자별 세부 경로를 지나 최종적으로 sparse attention이 실제로 읽는 cache entry가 무엇인지 연결해 줍니다.",
    "Indexer query path": "Lightning Indexer가 main attention과 별도의 cheap retrieval query를 만드는 구간입니다. main attention query는 value sum까지 이어지는 고품질 attention score를 위한 것이고, indexer query는 compressed block ranking을 빠르게 만들기 위한 별도 표현입니다. RoPE, Hadamard rotation, FP4 activation을 거치며 retrieval용 근사 표현으로 바뀌기 때문에, 이 그룹은 정확한 attention output보다 top-k 후보를 싸게 고르는 데 초점이 있습니다.",
    "Indexer compressed KV cache": "Main KV cache와 분리된 indexer 전용 compressed cache를 보여주는 구간입니다. 이 cache는 value sum에 직접 쓰이지 않고, Lightning Indexer가 query와 dot score를 계산해 어떤 compressed block을 볼지 정하는 검색 메모리로 쓰입니다. 분리된 작은 retrieval memory를 두는 의도는 main compressed KV cache를 매번 전부 attention score 대상으로 삼지 않고, 먼저 더 싼 공간에서 후보를 줄이는 것입니다.",
    "Score + head weighting": "Indexer query와 index cache를 비교해 compressed block ranking score를 만드는 구간입니다. 여러 index head는 서로 다른 retrieval 관점을 제공하고, token별 head weighting은 이 점수들을 하나의 block score로 합쳐 top-k selection에 넘깁니다. 이 그룹은 sparse attention의 품질이 단순 dot product 하나가 아니라, head별 후보 평가와 query-dependent weighting을 거쳐 결정된다는 점을 보여줍니다.",
    "Masked TopK selected blocks": "Indexer score에서 causal correctness와 top-k budget을 적용하는 선택 마무리 구간입니다. future block을 먼저 제거해 decode causality를 지키고, 남은 compressed block 중 제한된 수만 골라 attention compute를 예산 안에 묶습니다. 마지막 offset 변환은 compressed block id를 실제 attention cache id로 바꾸므로, 이 그룹은 retrieval score가 실행 가능한 KV gather 목록으로 바뀌는 경계입니다.",
    "mHC MoE entry/exit": "MoE scene에서 sparse FFN을 감싸는 mHC wrapper의 경계를 보여줍니다. mHC는 manifold / hyper-connection residual lane 구조로, MoE가 독립적인 FFN처럼 보이더라도 실제로는 `[B,S,4,D]` lane state를 읽고 다시 쓰는 wrapper 안에서 실행됩니다. 이 그룹의 의도는 routing과 expert 계산만 보면 놓치기 쉬운 residual lane read/write boundary를 MoE 앞뒤에 명확히 드러내는 것입니다.",
    "Router scores + ids": "MoE(Mixture-of-Experts)에서 어떤 expert를 실행할지 결정하는 routing control 구간입니다. 초기 layer의 hash routing은 token id 기반 expert prior를 쓰고, 일반 layer의 top-k routing은 score, bias, original score gather를 통해 expert 선택과 output weighting을 분리합니다. 이 그룹은 token이 어떤 expert로 갈지 정하는 control path이며, compute 절감뿐 아니라 expert load와 token specialization을 좌우하는 핵심 의사결정 지점입니다.",
    "Routed expert dispatch": "선택된 expert별로 token rows를 실제 실행 가능한 batch 형태로 모으는 runtime 구간입니다. routing 결과는 논리적으로는 token마다 expert id 목록이지만, 실제 matmul은 expert별 token 묶음으로 재배치되어야 효율적으로 실행됩니다. 이 그룹은 sparse MoE에서 자주 숨겨지는 dispatch/gather 비용을 드러내며, routing decision이 실제 expert compute layout으로 바뀌는 지점을 보여줍니다.",
    "Routed SwiGLU internals": "Routed expert 하나가 token hidden을 어떻게 FFN 변환하는지 보여주는 expert 내부 구간입니다. SwiGLU(Swish-Gated Linear Unit)는 gate projection과 up projection을 곱해 비선형성을 만들고, down projection은 다시 hidden dimension으로 복원합니다. 이 그룹의 의도는 MoE expert가 단순 linear layer가 아니라 token별로 선택된 작은 FFN이며, routed capacity의 실제 표현력은 이 gated projection 내부에서 나온다는 점을 보여주는 것입니다.",
    "Shared expert + combine": "Sparse routing과 무관하게 모든 token이 거치는 shared expert path와 routed output 결합을 보여줍니다. shared expert는 token별 top-k expert가 놓칠 수 있는 공통 변환을 항상 제공하고, routed experts는 token-specific specialization을 담당합니다. 이 그룹은 MoE가 완전히 sparse expert만 믿는 구조가 아니라, 공통 경로와 선택 경로를 더해 안정적인 기본 변환과 큰 sparse capacity를 함께 쓰는 설계라는 점을 설명합니다.",
    "Final stack state": "Decoder 반복이 끝난 상태를 output branch로 넘기는 경계입니다. 마지막 mHC lane state, 원래 input token identity, output branch가 만나는 위치이므로 LM head와 MTP(Multi-Token Prediction) branch가 어디서 시작되는지 구분해 줍니다. 이 그룹은 layer 내부 반복과 최종 head 계산을 분리해, logits가 매 layer에서 만들어지는 것이 아니라 stack exit 이후에만 계산된다는 흐름을 명확히 합니다.",
    "LM head path": "Main autoregressive prediction을 만드는 최종 head 경로입니다. 4-lane mHC residual state를 단일 hidden stream으로 collapse하고 final RMSNorm으로 scale을 맞춘 뒤, 마지막 token만 vocabulary projection해 logits를 만듭니다. 이 그룹의 의도는 output head가 전체 sequence 전체나 매 layer에서 반복되는 무거운 계산이 아니라, decode path에서 최종 last-token state를 vocab score로 바꾸는 좁은 단계임을 보여주는 것입니다.",
    "MTP branch": "MTP(Multi-Token Prediction) branch는 main logits와 별도로 final hidden state와 token embedding 정보를 결합해 auxiliary next-token prediction을 수행하는 보조 경로입니다. 공개 graph에서는 MTP block이 SWA-only attention mode(R=0)로 표시되며, main decoder stack 뒤에 붙어 추가 prediction signal을 제공합니다. 이 그룹은 main autoregressive head와 별도의 보조 prediction branch가 어떤 입력을 받아 어떻게 logits로 이어지는지 보여주는 데 목적이 있습니다.",
  };
  return [purpose[group.label] || `${escapeHtml(group.label)} 관련 노드들을 하나의 기능 단위로 묶어 전체 detailed graph 안에서의 역할을 보여줍니다.`];
}

function groupNarrative(group, docs) {
  const n = (id, label) => nodeInline(id, DATA.nodes[id], label);
  if (isEnglish()) return englishGroupNarrative(group, docs, n);
  const narratives = {
    "Model entry (once)": [
      `${n("input-ids", "토큰 id")}는 모델에 한 번 들어오는 이산 입력이고, ${n("embedding", "embedding lookup")}이 이를 hidden vector로 바꾼 뒤 ${n("hc-expand", "4-lane residual state")}로 확장합니다. 이 상태가 ${n("stack-entry", "대표 decoder layer")}의 시작점이 되므로, 반복 layer 내부가 아니라 전체 stack 진입부를 보여줍니다.`,
    ],
    "mHC controller + read path": [
      `${n("hc-flatten", "lane flatten")}은 4개 residual lane을 controller가 볼 수 있는 하나의 vector로 펼치고, ${n("hc-controller", "controller linear")}가 read/write/mix 정책을 한 번에 예측합니다. 그 출력은 ${n("hc-split", "pre, post, comb split")}으로 갈라지고, ${n("hc-pre-sigmoid", "read coefficient")}와 ${n("hc-post-sigmoid", "write coefficient")}는 bounded scalar로 정리됩니다.`,
      `Residual lane mixing 쪽은 ${n("hc-comb-softmax", "row-normalized comb seed")}를 만든 뒤 ${n("hc-comb-sinkhorn", "Sinkhorn-normalized comb")}로 안정화됩니다. 최종적으로 ${n("hc-read", "read data path")}가 pre coefficient를 이용해 4-lane state를 attention이 받을 단일 hidden stream으로 읽습니다.`,
    ],
    "Attention Q/KV paths": [
      `${n("q-wqa", "query low-rank projection")}과 ${n("q-norm", "query normalization")}은 attention query와 indexer query가 공유할 안정적인 latent를 만들고, ${n("q-wqb", "head expansion")}, ${n("q-reshape", "head reshape")}, ${n("q-rope", "query RoPE")}가 이를 실제 multi-head query로 바꿉니다.`,
      `KV 쪽에서는 ${n("kv-wkv", "shared KV projection")}과 ${n("kv-norm", "KV normalization")}이 하나의 512-dim shared cache entry를 만들고, ${n("kv-slice", "content/RoPE split")}와 ${n("kv-rope-quant", "RoPE/quantized KV")}가 key score와 value semantics를 분리합니다. 최근 token 후보는 ${n("window-topk", "SWA window")}와 ${n("cache-layout", "logical cache layout")}에서 오고, HCA에서는 ${n("hca-all-compressed", "all compressed blocks")}가 indexer 없이 들어갑니다.`,
      `선택된 후보는 ${n("attn-selected", "selected ids")}와 ${n("attn-gather", "KV gather")}를 거쳐 ${n("attn-score", "QK score")}, ${n("attn-mask-sink", "mask/sink")}, ${n("attn-softmax", "selected softmax")}, ${n("attn-value-sum", "value sum")}으로 처리됩니다. 마지막으로 ${n("attn-inv-rope", "inverse RoPE value fix")}가 shared-KV value phase를 보정하고 ${n("o-woa", "grouped low-rank output A")}, ${n("o-wob", "output projection B")}가 residual stream 크기로 복원합니다.`,
    ],
    "SWA cache write path": [
      `${n("swa-prefill-write", "prefill write")}는 긴 prompt 중 최근 local window만 uncompressed cache에 남기고, ${n("swa-decode-write", "decode ring write")}는 새 token KV를 128-slot ring buffer에 갱신합니다. ${n("cache-layout", "logical cache layout")}은 이 SWA prefix를 compressed suffix와 같은 id 공간에 놓고, ${n("window-topk", "window ids")}가 score top-k와 무관하게 최근 local token을 항상 후보로 보존합니다.`,
    ],
    "KV compressor + tail state": [
      `${n("comp-wkv", "compressor KV projection")}과 ${n("comp-wgate", "pooling score projection")}은 compressed entry를 만들 재료와 weight를 따로 만들고, ${n("comp-ape", "compressor APE")}가 block 내부 위치 정보를 gate score에 더합니다. Streaming decode에서는 ${n("tail-state", "tail state")}, ${n("comp-cutoff", "cutoff/remainder split")}, ${n("tail-append", "tail append")}가 아직 block을 이루지 못한 token projection을 다음 호출까지 이어 줍니다.`,
      `완성된 block은 ${n("comp-block-view", "block view")}와 ${n("overlap-transform", "c4a overlap transform")}을 거쳐 pooling 축으로 재배치되고, ${n("gated-pool", "softmax-gated pooling")}이 여러 token을 하나의 compressed KV entry로 모읍니다. 그 entry는 ${n("comp-anchor", "anchor position")}, ${n("comp-norm-rope", "compressed norm/RoPE")}, ${n("comp-cache-slot", "compressed slot map")}, ${n("comp-cache-write", "compressed cache write")}를 통해 attention cache에서 읽을 수 있는 형태로 저장됩니다.`,
    ],
    "Lightning indexer": [
      `${n("idx-q", "indexer query")}는 main attention Q가 아니라 q latent에서 cheap retrieval query를 만들고, ${n("idx-rope", "indexer RoPE")}, ${n("idx-hadamard", "Hadamard rotation")}, ${n("idx-fp4", "FP4 activation")}가 ranking용 표현을 가볍게 만듭니다. 별도의 ${n("idx-cache-compress", "index cache compressor")}, ${n("idx-cache-write", "index cache write")}, ${n("idx-cache", "index cache")}는 main KV와 분리된 작은 retrieval memory를 유지합니다.`,
      `${n("idx-einsum", "ReLU dot score")}가 query와 compressed index cache를 비교하고, ${n("idx-weight", "head weighting")}가 여러 index head를 하나의 block score로 합칩니다. 이후 ${n("idx-mask", "causal mask")}, ${n("idx-topk", "compressed topK")}, ${n("idx-offset", "cache offset")}이 future block을 제거하고 SWA prefix 뒤 cache id로 변환합니다.`,
    ],
    "mHC attention residual mixing": [
      `${n("attn-residual-mix", "residual lane mixing")}은 attention output과 별개로 기존 4-lane residual state를 comb matrix로 운반하고, ${n("attn-post-inject", "attention output injection")}은 단일 attention result를 post coefficient로 각 lane에 분배합니다. ${n("hc-write", "attention writeback")}은 이 두 항을 더해 다음 sublayer가 받을 residual lane state를 만듭니다.`,
    ],
    "mHC MoE controller + read path": [
      `${n("ffn-hc-flatten", "MoE lane flatten")}과 ${n("ffn-hc-controller", "MoE controller")}는 attention과 별도 parameter set으로 MoE sublayer 전용 coefficient를 만듭니다. ${n("ffn-hc-split", "MoE split")} 후 ${n("ffn-hc-pre-sigmoid", "MoE read coefficient")}와 ${n("ffn-hc-post-sigmoid", "MoE write coefficient")}가 bounded scale로 정리되고, ${n("ffn-hc-comb-softmax", "MoE comb seed")}와 ${n("ffn-hc-comb-sinkhorn", "MoE Sinkhorn comb")}이 residual lane transport matrix를 안정화합니다.`,
      `마지막으로 ${n("hc-pre-moe", "MoE read path")}가 4-lane residual을 router와 experts가 처리할 단일 hidden stream으로 읽어 MoE compute가 lane마다 반복되지 않게 합니다.`,
    ],
    "MoE routing + SwiGLU experts": [
      `${n("gate-score", "router score")}는 expert affinity를 만들고, 초기 layer에서는 ${n("hash-route", "hash routing")}가 token id 기반 expert prior를 사용합니다. 후반 layer에서는 ${n("route-bias", "selection bias")}와 ${n("topk-route", "top-k routing")}가 expert id를 고르며, ${n("route-score-gather", "original score gather")}와 ${n("route-weights", "route weights")}가 output magnitude를 위한 mixture weight를 다시 계산합니다.`,
      `${n("expert-counts", "expert counts")}와 ${n("expert-dispatch", "dispatch")}는 token rows를 expert별로 모으고, routed path는 ${n("expert-w1w3", "expert gate/up projection")}, ${n("swiglu", "SwiGLU")}, ${n("expert-w2", "expert down projection")}, ${n("routed-accum", "routed accumulation")}으로 sparse FFN 결과를 원래 token stream에 되돌립니다. 병렬 always-on path에서는 ${n("shared-w1w3", "shared gate/up")}, ${n("shared-swiglu", "shared SwiGLU")}, ${n("shared-w2", "shared down")}가 공통 transform을 만들고, ${n("expert-combine", "expert combine")}과 ${n("moe-allreduce", "MoE all-reduce")}가 routed/shared 결과를 하나로 맞춥니다.`,
    ],
    "mHC MoE residual mixing": [
      `${n("ffn-residual-mix", "MoE residual lane mixing")}은 sparse expert update와 별개로 기존 residual lane을 comb matrix로 운반하고, ${n("ffn-post-inject", "MoE output injection")}은 MoE result를 각 lane에 주입합니다. ${n("hc-post-moe", "MoE writeback")}은 두 항을 합쳐 다음 decoder block이 받을 4-lane residual state를 만듭니다.`,
    ],
    "Final output + MTP": [
      `${n("stack-exit", "final stack state")} 이후에만 output path가 시작되고, ${n("hc-head-collapse", "HC head collapse")}와 ${n("final-rmsnorm", "final RMSNorm")}이 4-lane state를 logits용 hidden stream으로 정리합니다. ${n("last-token", "last-token slice")}와 ${n("lm-project", "LM projection")}는 모든 token이 아니라 마지막 token의 vocabulary score를 만들고, ${n("logits", "logits")}가 main decode score가 됩니다.`,
      `보조 branch에서는 ${n("mtp-embed", "MTP embedding path")}, ${n("mtp-hidden-proj", "MTP hidden projection")}, ${n("mtp-combine", "MTP combine")}, ${n("mtp-block", "MTP block")}, ${n("mtp-head", "MTP head")}가 final state 뒤에서 auxiliary next-token prediction을 구성합니다.`,
    ],
    "mHC attention controller + read path": [
      `${n("hc-flatten", "flattened residual lanes")}를 기준으로 ${n("hc-controller", "attention mHC controller")}가 attention 앞뒤의 coefficient를 예측합니다. ${n("hc-split", "split")}은 그 결과를 read/write/mix 계열로 나누고, ${n("hc-pre-sigmoid", "pre coefficient")}와 ${n("hc-post-sigmoid", "post coefficient")}는 lane read와 output injection의 scale을 안정화합니다.`,
      `${n("hc-comb-softmax", "comb seed")}와 ${n("hc-comb-sinkhorn", "Sinkhorn comb")}는 residual lane transport matrix를 만들며, ${n("hc-read", "read path")}가 이 controller 결과 중 read coefficient를 실제 attention input stream에 적용합니다.`,
    ],
    "mHC attention entry/exit": [
      `${n("mhc-attn", "attention mHC wrapper")}는 attention sublayer를 mHC read/write 구조로 감싸는 개념적 경계이고, ${n("hc-write", "HC writeback")}는 attention 결과와 residual lane transport를 합쳐 wrapper 밖으로 내보내는 출구입니다.`,
    ],
    "Query LoRA + RoPE": [
      `${n("q-wqa", "low-rank query A projection")}이 hidden stream을 compact latent로 낮추고 ${n("q-norm", "query RMSNorm")}이 main attention과 indexer가 공유할 scale을 맞춥니다. 이후 ${n("q-wqb", "query B projection")}와 ${n("q-reshape", "head reshape")}가 multi-head query를 만들고, ${n("q-rope", "RoPE slice")}가 position phase를 query의 일부 dimension에만 넣습니다.`,
    ],
    "Shared KV + SWA": [
      `${n("kv-wkv", "shared KV projection")}과 ${n("kv-norm", "KV RMSNorm")}은 head별 KV가 아니라 하나의 shared 512-dim cache entry를 만듭니다. ${n("kv-slice", "content/RoPE split")}와 ${n("kv-rope-quant", "KV RoPE/quantization")}는 key score와 value semantics를 분리하고, ${n("cache-layout", "cache layout")}과 ${n("window-topk", "SWA window")}가 최근 128개 local token을 항상 attention 후보로 유지합니다.`,
    ],
    "Compressed selection": [
      `${n("compressor", "KV compressor")}가 오래된 context를 compressed memory로 바꾸고, CSA에서는 ${n("indexer", "Lightning indexer")}와 ${n("idx-offset", "compressed cache offset")}이 중요한 compressed block만 고릅니다. HCA에서는 ${n("hca-all-compressed", "all compressed blocks")}가 indexer 없이 compressed memory 전체를 후보로 넣고, 최종 후보 집합은 ${n("attn-selected", "selected KV ids")}에 모입니다.`,
    ],
    "Core attention kernel": [
      `${n("attn-gather", "selected KV gather")}가 cache에서 필요한 entry만 모으고, ${n("attn-score", "QK score")}가 sparse candidate set 위에서 logit을 계산합니다. ${n("attn-mask-sink", "mask and sink")}가 causal/window 제약과 attention sink를 더한 뒤 ${n("attn-softmax", "selected softmax")}가 확률을 만들고, ${n("attn-value-sum", "value sum")}이 selected value만 가중합합니다.`,
    ],
    "KV sharing output fix": [
      `${n("attn-inv-rope", "inverse RoPE")}는 shared KV가 key로 쓰일 때 들어간 position phase를 value output에서 보정합니다. 그 뒤 ${n("o-woa", "grouped output A projection")}와 ${n("o-wob", "output B projection")}가 attention head 결과를 residual hidden dimension으로 복원합니다.`,
    ],
    "SWA window cache": [
      `${n("kv-path", "shared KV path")}와 ${n("kv-cache", "KV cache")}가 window/compressed cache의 기본 표현을 만들고, ${n("swa-prefill-write", "prefill window write")}와 ${n("swa-decode-write", "decode ring write")}가 최근 local KV를 uncompressed 상태로 유지합니다. ${n("cache-layout", "logical cache layout")}과 ${n("window-topk", "window ids")}는 이 SWA region을 compressed suffix와 같은 attention id 체계에 올립니다.`,
    ],
    "Compressor projections": [
      `${n("comp-wkv", "compressor KV projection")}은 pooling될 value 후보를 만들고, ${n("comp-wgate", "gate projection")}은 어떤 token/channel을 더 남길지 정하는 score를 만듭니다. ${n("comp-ape", "compressor APE")}는 이 score에 block-local 위치 정보를 더해 compressed entry가 내부 순서를 완전히 잃지 않게 합니다.`,
    ],
    "Tail / cutoff runtime state": [
      `${n("tail-state", "tail state")}는 아직 compression block을 채우지 못한 remainder를 request state로 보관합니다. ${n("comp-cutoff", "cutoff split")}은 완성 block과 남은 tail을 나누고, ${n("tail-append", "tail append")}는 다음 decode/prefill chunk에서 이어 쓸 state를 갱신합니다.`,
    ],
    "Block pooling": [
      `${n("comp-block-view", "block view")}가 projection channel을 pooling kernel이 읽을 block/token/value 축으로 바꾸고, c4a에서는 ${n("overlap-transform", "overlap transform")}이 stride 4 entry가 8-token span을 보게 만듭니다. ${n("gated-pool", "softmax-gated pool")}은 learned weight로 여러 token을 하나의 compressed KV entry로 합칩니다.`,
    ],
    "Compressed entry write": [
      `${n("comp-anchor", "anchor position")}는 compressed block 전체에 대표 position을 붙이고, ${n("comp-norm-rope", "compressed norm/RoPE")}가 attention key로 쓸 수 있게 scale과 phase를 맞춥니다. ${n("comp-cache-slot", "slot map")}은 SWA prefix 뒤 logical slot으로 옮기고, ${n("comp-cache-write", "cache write")}가 compressed memory에 기록합니다.`,
    ],
    "Attention consumer": [
      `${n("attn-gather", "attention gather")}는 compressor나 SWA가 만든 cache entry를 실제 attention kernel 입력으로 소비하는 접점입니다.`,
    ],
    "Indexer query path": [
      `${n("q-norm", "normalized query latent")}에서 ${n("idx-q", "indexer query projection")}가 retrieval 전용 query를 만들고, ${n("idx-rope", "indexer RoPE")}가 query position을 반영합니다. ${n("idx-hadamard", "Hadamard rotation")}와 ${n("idx-fp4", "FP4 activation")}는 top-k ranking을 싸게 만들기 위한 approximate representation을 구성합니다.`,
    ],
    "Indexer compressed KV cache": [
      `${n("kv-path", "shared KV path")}와 별개로 ${n("idx-cache-compress", "index cache compressor")}가 retrieval score 전용 128-dim entry를 만들고, ${n("idx-cache-write", "index cache write")}가 이를 저장합니다. 이후 ${n("idx-cache", "index cache")}는 main value sum이 아니라 compressed block ranking에만 쓰입니다.`,
    ],
    "Score + head weighting": [
      `${n("idx-einsum", "index score")}이 query와 compressed index cache 사이의 ReLU dot score를 만들고, ${n("idx-weight", "head weighting")}가 64개 index head의 점수를 query-dependent weight로 합쳐 block score를 만듭니다.`,
    ],
    "Masked TopK selected blocks": [
      `${n("idx-mask", "causal block mask")}가 아직 볼 수 없는 compressed block을 제거하고, ${n("idx-topk", "compressed topK")}가 제한된 수의 block만 남깁니다. ${n("idx-offset", "cache offset")}은 그 block id를 SWA prefix 뒤 cache id로 바꾸고, ${n("attn-selected", "selected ids")}가 window 후보와 합칩니다.`,
    ],
    "mHC MoE entry/exit": [
      `${n("mhc-ffn", "MoE mHC wrapper")}는 sparse FFN 주변의 read/write 구조를 나타내고, ${n("hc-post-moe", "post-MoE writeback")}는 MoE result와 residual lane transport를 합친 다음 block input state를 나타냅니다.`,
    ],
    "Router scores + ids": [
      `${n("gate-score", "router score")}가 expert affinity를 만들고, 초기 layer에서는 ${n("hash-route", "hash route")}가 token id 기반 expert ids를 사용합니다. 일반 layer에서는 ${n("route-bias", "selection bias")}와 ${n("topk-route", "top-k route")}가 expert ids를 고르고, ${n("route-score-gather", "original score gather")}와 ${n("route-weights", "normalized route weights")}가 실제 mixture weight를 만듭니다.`,
    ],
    "Routed expert dispatch": [
      `${n("expert-counts", "expert counts")}는 expert별 token row 수를 세고, ${n("expert-dispatch", "expert dispatch")}는 선택된 rows를 해당 expert 연산으로 모아 sparse FFN을 batch 연산처럼 실행할 수 있게 합니다.`,
    ],
    "Routed SwiGLU internals": [
      `${n("expert-w1w3", "expert gate/up projection")}이 routed expert의 intermediate features를 만들고, ${n("swiglu", "SwiGLU")}가 gate와 up activation을 곱해 비선형성을 줍니다. ${n("expert-w2", "expert down projection")}는 hidden dimension으로 복원하고, ${n("routed-accum", "routed accumulation")}은 route weight를 곱해 원 token 위치로 되돌립니다.`,
    ],
    "Shared expert + combine": [
      `${n("shared-w1w3", "shared gate/up projection")}, ${n("shared-swiglu", "shared SwiGLU")}, ${n("shared-w2", "shared down projection")}는 모든 token이 항상 거치는 common FFN path를 구성합니다. ${n("expert-combine", "expert combine")}은 shared output과 routed output을 합치고, ${n("moe-allreduce", "MoE all-reduce")}는 parallel shard 결과를 동일한 hidden stream으로 맞춥니다.`,
    ],
    "Final stack state": [
      `${n("input-ids", "input ids")}는 output/MTP branch에서도 token identity를 참조할 수 있는 원천이고, ${n("hc-post-moe", "post-MoE state")}는 decoder stack 마지막 residual lane state를 제공합니다. ${n("stack-exit", "stack exit")}는 반복 layer가 끝나고 output head로 넘어가는 경계를 표시합니다.`,
    ],
    "LM head path": [
      `${n("hc-head-collapse", "HC head collapse")}가 4-lane residual을 단일 hidden stream으로 접고, ${n("final-rmsnorm", "final RMSNorm")}이 vocab projection 전 scale을 맞춥니다. ${n("last-token", "last-token slice")}는 마지막 token hidden만 고르고, ${n("lm-project", "LM projection")}와 ${n("logits", "logits")}가 main vocabulary score를 만듭니다.`,
    ],
    "MTP branch": [
      `${n("mtp-embed", "MTP embedding")}은 token id 기반 branch 입력을 만들고, ${n("mtp-hidden-proj", "hidden projection")}는 final hidden state를 auxiliary block에 맞춥니다. ${n("mtp-combine", "MTP combine")}이 두 입력을 합친 뒤 ${n("mtp-block", "MTP block")}과 ${n("mtp-head", "MTP head")}가 auxiliary next-token logits를 만듭니다.`,
    ],
  };
  if (narratives[group.label]) return narratives[group.label];
  return fallbackGroupNarrative(group, docs, n);
}

function fallbackGroupNarrative(group, docs, n) {
  const refs = docs.map((doc) => n(doc.id)).join(", ");
  return [`${escapeHtml(group.label)}는 ${refs}를 하나의 처리 구간으로 묶어 보여줍니다. 각 링크를 누르면 해당 노드의 상세 설명과 graph 위치로 이동합니다.`];
}

function englishGroupNarrative(group, docs, n) {
  if (!docs.length) return [`${escapeHtml(group.label)} has no visible nodes in the current graph view.`];
  const chunks = [];
  for (let index = 0; index < docs.length; index += 5) {
    const part = docs
      .slice(index, index + 5)
      .map((doc) => {
        const input = escapeHtml(resolve(doc.input || ""));
        const output = escapeHtml(resolve(doc.output || ""));
        return `${n(doc.id)} (${input} -> ${output})`;
      })
      .join(", ");
    const prefix = index === 0 ? "The linked flow starts with " : "It then continues through ";
    chunks.push(`${prefix}${part}.`);
  }
  return chunks;
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
      const note = formula.note && !isEnglish() ? `<p>${escapeHtml(resolve(formula.note))}</p>` : "";
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
