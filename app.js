const DATA = window.DSV4_GRAPH;

const state = {
  model: "pro",
  layer: 0,
  scene: "overview",
  graphDetail: "detailed",
  selected: "input-ids",
  detailOpen: true,
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
  if (!visibleNode(state.selected) || !visibleNodeIds().includes(state.selected)) {
    state.selected = visibleNodeIds()[0] || "input-ids";
  }
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
    .attr("class", (item) => `graph-group ${item.category} ${item.nodeIds?.includes(state.selected) ? "active" : ""}`);

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
    .attr("class", (item) => `minimap-group ${item.nodeIds?.includes(state.selected) ? "active" : ""}`)
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
  const doc = DATA.nodes[state.selected];
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

function renderPanelState() {
  document.querySelector(".info-panel")?.classList.toggle("closed", !state.detailOpen);
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
    (item) => `graph-group ${item.category} ${item.nodeIds?.includes(selected) ? "active" : ""}`,
  );
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
    render(true);
    return;
  }

  const graphNode = event.target.closest("[data-node]");
  if (graphNode) {
    state.selected = graphNode.dataset.node;
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
document.querySelector("#closeDetail")?.addEventListener("click", () => {
  state.detailOpen = false;
  renderPanelState();
});
document.querySelector("#backScene")?.addEventListener("click", () => openScene("overview"));
document.querySelector("#drillButton")?.addEventListener("click", (event) => {
  if (event.currentTarget.dataset.scene) openScene(event.currentTarget.dataset.scene);
});

render(true);
document.fonts?.ready.then(() => render(true));
