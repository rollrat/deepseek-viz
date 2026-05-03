const DATA = window.DSV4_GRAPH;

const state = {
  model: "pro",
  layer: 0,
  scene: "overview",
  graphDetail: "detailed",
  selected: "input-ids",
  selectedGroup: null,
  detailOpen: true,
  detailPanelPosition: null,
  showNodeFormula: true,
  embedNodeDescription: false,
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
  const descToggle = document.querySelector("#embedDescToggle");
  if (descToggle) descToggle.checked = state.embedNodeDescription;
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
  if (state.graphDetail !== "detailed") state.selectedGroup = null;
  if (state.selectedGroup && !currentGroup(state.selectedGroup)) state.selectedGroup = null;
  if (state.selectedGroup) return;
  if (!visibleNode(state.selected) || !visibleNodeIds().includes(state.selected)) {
    state.selected = visibleNodeIds()[0] || "input-ids";
  }
}

function currentGroup(label) {
  return (sceneView().groups || []).find((group) => group.label === label) || null;
}

function isGroupActive(group) {
  if (state.selectedGroup) return group.label === state.selectedGroup;
  return group.nodeIds?.includes(state.selected);
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
  const descLen = state.embedNodeDescription ? String(resolve(doc.details?.why || doc.summary || "")).length : 0;
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
      active: state.selected === edgeRef.v || state.selected === edgeRef.w,
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
    .attr("class", (item) => `node ${item.doc.category} ${state.selected === item.id ? "selected" : ""} ${item.doc.drill ? "drillable" : ""}`)
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
      <strong>${escapeHtml(item.doc.title)}</strong>
    </div>
    ${formula}
    ${description}
    <div class="node-shape">${escapeHtml(input)} -> ${escapeHtml(output)}</div>
  `;
}

function nodeDescriptionHtml(doc) {
  const source = resolve(doc.details?.why || doc.summary || "");
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
  document.querySelector("#detailTitle").textContent = doc.title;
  document.querySelector("#detailCards").innerHTML = renderDetailCards(doc.details);
  document.querySelector("#inputShape").textContent = resolve(doc.input);
  document.querySelector("#outputShape").textContent = resolve(doc.output);
  document.querySelector("#paramList").innerHTML = Object.entries(doc.params)
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
  document.querySelector("#detailCategory").textContent = `${group.category} group`;
  document.querySelector("#detailTitle").textContent = group.label;
  document.querySelector("#detailCards").innerHTML = renderGroupCards(group, docs);
  document.querySelector("#inputShape").textContent = docs.length ? resolve(docs[0].input) : "";
  document.querySelector("#outputShape").textContent = docs.length ? resolve(docs[docs.length - 1].output) : "";
  document.querySelector("#paramList").innerHTML = `<div><dt>nodes</dt><dd>${docs.length}</dd></div>`;
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
      <h3>설명</h3>
      ${prose.map((paragraph) => `<p>${paragraph}</p>`).join("")}
    </section>`;
}

function groupPurpose(group) {
  const purpose = {
    "Model entry (once)": "Decoder layer 반복에 들어가기 전, 모델 입력을 hidden state로 바꾸고 DeepSeek V4의 핵심 상태 형식인 4-lane residual stream을 시작하는 진입 구간입니다. token id 기반 정보와 mHC residual 표현이 전체 모델에서 처음 만나는 지점입니다.",
    "mHC controller + read path": "Attention sublayer가 4개 residual lane 전체를 직접 계산하지 않고 필요한 단일 hidden stream만 읽도록 만드는 mHC 제어 구간입니다. DeepSeek V4의 residual 안정화 특징인 read/write/mix coefficient가 여기서 만들어집니다.",
    "Attention Q/KV paths": "Long-context attention의 본체입니다. query low-rank path, shared KV, SWA window, compressed cache, sparse attention, grouped output projection이 한데 묶여 memory와 compute를 줄이면서 attention 품질을 유지합니다.",
    "SWA cache write path": "최근 token 정보를 압축하지 않고 보존하는 local memory path입니다. compressed attention이 오래된 문맥을 싸게 다루는 동안, SWA cache는 가까운 문맥의 세부 정보를 잃지 않게 해 줍니다.",
    "KV compressor + tail state": "긴 문맥을 compressed KV entry로 바꾸는 memory 절감 구간입니다. streaming decode에서 block boundary가 맞지 않는 상황까지 처리하므로, 모델 구조와 runtime cache 관리가 맞물립니다.",
    "Lightning indexer": "CSA layer에서 많은 compressed KV block 중 실제로 볼 block만 고르는 retrieval gate입니다. attention을 직접 줄이는 것이 아니라 attention 전에 candidate set을 줄여 long-context compute를 제한합니다.",
    "mHC attention residual mixing": "Attention 결과를 residual lane state에 다시 쓰는 attention writeback 구간입니다. 기존 residual lane 자체를 운반하는 항과 attention output을 주입하는 항이 분리되어 있다는 점이 mHC의 중요한 특징입니다.",
    "mHC MoE controller + read path": "MoE sublayer가 4-lane residual state에서 필요한 hidden stream을 읽도록 만드는 FFN-side mHC controller입니다. attention 쪽과 같은 구조를 쓰지만 별도 parameter set으로 MoE의 read/write 정책을 따로 제어합니다.",
    "MoE routing + SwiGLU experts": "DeepSeek V4의 sparse capacity를 담당하는 MoE 본체입니다. token마다 일부 routed expert만 활성화하고 shared expert를 항상 더해, 큰 parameter capacity와 제한된 token당 compute를 동시에 노립니다.",
    "mHC MoE residual mixing": "MoE 결과를 4-lane residual stream으로 되돌리는 writeback 구간입니다. sparse expert update와 기존 residual lane transport를 합쳐 다음 decoder block으로 넘깁니다.",
    "Final output + MTP": "반복 decoder stack이 끝난 뒤 최종 prediction으로 빠지는 output 구간입니다. LM head가 매 layer마다 실행되는 것이 아니라 최종 state 뒤에서 last token에 대해 계산된다는 경계를 명확히 보여줍니다.",
    "mHC attention controller + read path": "Attention scene 안에서 mHC controller 부분만 확대해 보여주는 구간입니다. attention 계산 자체보다 attention에 들어갈 stream을 어떻게 읽고, 이후 writeback coefficient를 어떻게 준비하는지가 핵심입니다.",
    "mHC attention entry/exit": "Attention sublayer를 감싸는 mHC wrapper의 입구와 출구를 요약합니다. 내부 attention 연산보다 residual lane state가 sublayer 경계를 어떻게 통과하는지 이해하는 데 초점을 둡니다.",
    "Query LoRA + RoPE": "Attention query를 싸게 만들면서 position 정보를 필요한 slice에만 넣는 query 생성 구간입니다. low-rank query latent로 projection 부담을 줄이고, RoPE slice로 위치 정보를 분리합니다.",
    "Shared KV + SWA": "Key/value cache memory를 줄이기 위한 shared KV와 최근 local window 보존을 함께 보여줍니다. head별 KV를 피하면서도 최근 128 token은 uncompressed로 남기는 균형점입니다.",
    "Compressed selection": "Attention이 읽을 compressed memory 후보를 정하는 선택 구간입니다. CSA에서는 indexer가 sparse retrieval을 하고, HCA에서는 강하게 압축된 block 전체를 읽는 차이가 여기서 드러납니다.",
    "Core attention kernel": "이미 선택된 KV 후보 위에서 실제 sparse attention을 수행하는 kernel 구간입니다. dense context 전체가 아니라 selected cache entry만 gather, score, softmax, value sum으로 처리합니다.",
    "KV sharing output fix": "Shared KV 설계의 부작용을 output 쪽에서 보정하는 구간입니다. key로 쓰기 위해 들어간 RoPE phase가 value output에 남지 않게 처리하고, attention output을 residual hidden 크기로 되돌립니다.",
    "SWA window cache": "Cache scene에서 local uncompressed memory를 담당합니다. compressed cache가 오래된 문맥을 줄이는 동안, SWA window는 최근 token의 정확한 KV를 유지하는 runtime 책임을 보여줍니다.",
    "Compressor projections": "Compression이 단순 평균이 아니라 learned projection과 learned gate score 위에서 일어난다는 점을 보여줍니다. compressed entry의 내용과 pooling weight가 각각 별도 path에서 만들어집니다.",
    "Tail / cutoff runtime state": "Compressor가 streaming 입력을 block 단위로 끊어 처리할 때 필요한 runtime state입니다. 아직 block이 완성되지 않은 token들을 잃지 않고 다음 호출로 넘깁니다.",
    "Block pooling": "여러 token representation을 하나의 compressed entry로 합치는 실제 pooling 구간입니다. c4a에서는 overlap을 통해 boundary 손실을 줄이고, gate score로 중요한 정보를 더 남깁니다.",
    "Compressed entry write": "Pooling된 compressed representation을 attention cache에서 읽을 수 있는 정식 cache entry로 마무리하는 구간입니다. anchor position, RoPE, slot mapping이 여기서 정리됩니다.",
    "Attention consumer": "Cache/compressor가 만든 결과가 attention kernel로 들어가는 소비 지점입니다. 여러 cache path의 산출물이 최종적으로 selected KV gather라는 하나의 kernel 입력으로 합쳐집니다.",
    "Indexer query path": "Lightning Indexer가 main attention과 별도의 cheap retrieval query를 만드는 구간입니다. 정확한 attention output보다 compressed block ranking을 빠르게 만드는 데 초점이 있습니다.",
    "Indexer compressed KV cache": "Main KV cache와 분리된 indexer 전용 compressed cache를 보여줍니다. value sum에 쓰는 cache가 아니라 top-k 검색을 위한 작은 retrieval memory입니다.",
    "Score + head weighting": "Indexer query와 index cache를 비교해 compressed block ranking score를 만드는 구간입니다. 여러 index head의 점수를 token별 weight로 합쳐 하나의 block score로 줄입니다.",
    "Masked TopK selected blocks": "Indexer score에서 causal correctness와 top-k budget을 적용하는 선택 마무리 구간입니다. future block을 제거하고, 남은 compressed block id를 attention cache id로 바꿉니다.",
    "mHC MoE entry/exit": "MoE scene에서 sparse FFN을 감싸는 mHC wrapper의 경계를 보여줍니다. MoE가 독립 sublayer처럼 보이지만 실제로는 residual lane read/write 구조 안에서 실행됩니다.",
    "Router scores + ids": "MoE에서 어떤 expert를 실행할지 결정하는 routing control 구간입니다. hash routing, top-k routing, bias, original score gather가 expert 선택과 output weighting을 분리하기 위해 배치됩니다.",
    "Routed expert dispatch": "선택된 expert별로 token을 실제 실행 가능한 batch 형태로 모으는 runtime 구간입니다. sparse routing의 논리적 선택이 실제 expert compute로 바뀌는 지점입니다.",
    "Routed SwiGLU internals": "Routed expert 하나가 token hidden을 어떻게 FFN 변환하는지 보여주는 expert 내부 구간입니다. SwiGLU gate/up/down projection이 routed capacity의 실제 비선형 계산을 담당합니다.",
    "Shared expert + combine": "Sparse routing과 무관하게 모든 token이 거치는 shared expert path와 routed output 결합을 보여줍니다. token별 specialization과 공통 변환을 동시에 사용하는 MoE 특징이 드러납니다.",
    "Final stack state": "Decoder 반복이 끝난 상태를 output branch로 넘기는 경계입니다. 최종 state, 입력 token identity, output branch가 만나는 지점이라 LM head와 MTP가 어디서 시작되는지 구분해 줍니다.",
    "LM head path": "Main autoregressive prediction을 만드는 최종 head 경로입니다. 4-lane state를 단일 hidden으로 접고, 마지막 token만 vocab projection해 logits를 만듭니다.",
    "MTP branch": "Main logits와 별도의 auxiliary next-token prediction branch입니다. final hidden state와 token embedding 정보를 결합해 MTP block으로 보내는 보조 학습/추론 경로를 보여줍니다.",
  };
  return [purpose[group.label] || `${escapeHtml(group.label)} 관련 노드들을 하나의 기능 단위로 묶어 전체 detailed graph 안에서의 역할을 보여줍니다.`];
}

function groupNarrative(group, docs) {
  const n = (id, label) => nodeInline(id, DATA.nodes[id], label);
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

function nodeInline(id, doc = DATA.nodes[id], label = null) {
  const text = label || doc?.title || id;
  const title = doc?.title || id;
  return `<button class="node-ref" data-group-node="${escapeHtml(id)}" title="${escapeHtml(title)}">${escapeHtml(text)}</button>`;
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
  const selected = state.selected;
  d3.selectAll("path.edge").attr(
    "class",
    (item) => `edge ${item.type === "branch" ? "branch" : "main"} ${selected === item.from || selected === item.to ? "active" : ""}`,
  );
  d3.selectAll("g.edge-label").attr(
    "class",
    (item) => `edge-label ${item.type === "branch" ? "branch" : "main"} ${selected === item.from || selected === item.to ? "active" : ""}`,
  );
  d3.selectAll("g.graph-group").attr(
    "class",
    (item) => `graph-group ${item.category} ${isGroupActive(item) ? "active" : ""} ${state.graphDetail === "detailed" ? "clickable" : ""}`,
  );
  d3.selectAll("#minimapGroups rect").attr("class", (item) => `minimap-group ${isGroupActive(item) ? "active" : ""}`);
  d3.selectAll("g.node").attr(
    "class",
    (item) => `node ${item.doc.category} ${selected === item.id ? "selected" : ""} ${item.doc.drill ? "drillable" : ""}`,
  );
}

function renderDetailCards(details) {
  if (!details) return "";
  const labels = {
    why: "설명",
    runtime: "런타임 동작",
    ui: "시각화 포인트",
    open: "남은 질문",
    formula: "계산식",
  };
  return Object.entries(details)
    .filter(([, value]) => value)
    .map(([label, value]) => {
      if (label === "formula") {
        return `<section class="formula-card"><h3>${escapeHtml(labels[label])}</h3>${renderFormulaList(value)}</section>`;
      }
      const items = Array.isArray(value) ? value : [value];
      const body =
        items.length === 1
          ? `<p>${escapeHtml(resolve(items[0]))}</p>`
          : `<ul>${items.map((item) => `<li>${escapeHtml(resolve(item))}</li>`).join("")}</ul>`;
      return `<section><h3>${escapeHtml(labels[label] || label)}</h3>${body}</section>`;
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
    if (state.graphDetail !== "detailed") state.selectedGroup = null;
    render(true);
    return;
  }

  const groupNodeButton = event.target.closest("[data-group-node]");
  if (groupNodeButton) {
    state.selected = groupNodeButton.dataset.groupNode;
    renderSelectionState();
    centerNode(state.selected);
    return;
  }

  const graphNode = event.target.closest("[data-node]");
  if (graphNode) {
    state.selected = graphNode.dataset.node;
    state.selectedGroup = null;
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
    state.detailOpen = true;
    renderDetail();
    renderPanelState();
    renderSelectionState();
  }
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

render(true);
document.fonts?.ready.then(() => render(true));
