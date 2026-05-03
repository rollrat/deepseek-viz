const DATA = window.DSV4_GRAPH;

const state = {
  model: "pro",
  layer: 0,
  scene: "overview",
  selected: "input-ids",
};

const elk = new ELK();
let zoomBehavior;
let lastLayout = { width: 1280, height: 1280 };
let layoutOffset = { x: 0, y: 0 };
let renderVersion = 0;
let shouldFitAfterLayout = true;

function model() {
  return DATA.models[state.model];
}

function scene() {
  return DATA.scenes[state.scene];
}

function ratioValue() {
  return model().schedule[state.layer] ?? 0;
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
  if (!node.visibleWhen) return true;
  if (node.visibleWhen.ratio !== undefined) return ratioValue() === node.visibleWhen.ratio;
  return true;
}

function visibleNodeIds() {
  return scene().nodeIds.filter(visibleNode);
}

function visibleEdges() {
  const ids = new Set(visibleNodeIds());
  return scene().edges.filter((edge) => ids.has(edge.from) && ids.has(edge.to));
}

function render(fitAfterLayout = false) {
  shouldFitAfterLayout ||= fitAfterLayout;
  renderStats();
  renderLayerControls();
  renderSceneHeader();
  ensureSelection();
  renderGraph();
  renderDetail();
}

function renderStats() {
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
    g.setEdge(edge.from, edge.to, { type: edge.type }, `${edge.from}-${edge.to}-${index}`);
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
        points: edge.sections?.flatMap((section) => [
          section.startPoint,
          ...(section.bendPoints || []),
          section.endPoint,
        ]) || [],
      },
    ]),
  );
  const groupData = (scene().groups || []).map((group) => groupBounds(group, nodeMap)).filter(Boolean);

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
  if (state.scene === "moe") return doc.category === "expert" ? 300 : 270;
  if (state.scene !== "overview") return 280;
  if (doc.category === "expert" || doc.category === "cache") return 230;
  return 220;
}

function nodeHeight(doc) {
  if (state.scene !== "overview") return 98;
  return doc.category === "attention" || doc.category === "routing" ? 76 : 72;
}

async function renderGraph() {
  const version = ++renderVersion;
  const g = await buildLayout();
  if (version !== renderVersion) return;
  const svg = d3.select("#graph");
  const zoomLayer = d3.select("#zoomLayer");
  const groupLayer = d3.select("#groupLayer");
  const edgeLayer = d3.select("#edgeLayer");
  const nodeLayer = d3.select("#nodeLayer");

  svg.attr("viewBox", `0 0 ${lastLayout.width} ${lastLayout.height}`);
  d3.select(".canvas-bg")
    .attr("x", 16)
    .attr("y", 16)
    .attr("width", lastLayout.width - 32)
    .attr("height", lastLayout.height - 32);

  const edgeData = [];
  g.edges().forEach((edgeRef) => {
    const edgeInfo = g.edge(edgeRef);
    edgeData.push({
      id: edgeRef.name,
      from: edgeRef.v,
      to: edgeRef.w,
      type: edgeInfo.type,
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

  const nodeData = visibleNodeIds().map((id) => {
    const laidOut = g.node(id);
    const doc = DATA.nodes[id];
    return { id, ...laidOut, x: laidOut.x - layoutOffset.x, y: laidOut.y - layoutOffset.y, doc };
  });

  const nodeById = Object.fromEntries(nodeData.map((item) => [item.id, item]));
  const groupData = g.groups || (scene().groups || []).map((group) => groupBounds(group, nodeById)).filter(Boolean);

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
    .attr("class", (item) => `graph-group ${item.category}`);

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
      .scaleExtent([0.18, 5])
      .on("zoom", (event) => zoomLayer.attr("transform", event.transform));
    svg.call(zoomBehavior);
  }

  if (shouldFitAfterLayout) {
    shouldFitAfterLayout = false;
    requestAnimationFrame(fitGraph);
  }
}

function lineForPoints(points) {
  if (!points || points.length < 2) return "";
  return d3
    .line()
    .x((point) => point.x)
    .y((point) => point.y)
    .curve(d3.curveLinear)(points);
}

function nodeHtml(item) {
  const input = truncate(resolve(item.doc.input), state.scene === "overview" ? 34 : 46);
  const output = truncate(resolve(item.doc.output), state.scene === "overview" ? 34 : 46);
  return `
    <div class="node-title-row">
      <span>${escapeHtml(item.doc.category)}</span>
      <strong>${escapeHtml(item.doc.title)}</strong>
    </div>
    <div class="node-shape">${escapeHtml(input)} -> ${escapeHtml(output)}</div>
  `;
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
  document.querySelector("#detailSummary").textContent = doc.summary;
  document.querySelector("#inputShape").textContent = resolve(doc.input);
  document.querySelector("#outputShape").textContent = resolve(doc.output);
  document.querySelector("#paramList").innerHTML = Object.entries(doc.params)
    .map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(resolve(value))}</dd></div>`)
    .join("");
  document.querySelector("#noteList").innerHTML = doc.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
  document.querySelector("#sourceList").innerHTML = doc.sources
    .map((source) => `<a href="${source.url}" target="_blank" rel="noreferrer">${source.label}</a>`)
    .join("");

  const drill = document.querySelector("#drillButton");
  drill.hidden = true;
  delete drill.dataset.scene;
}

function openScene(nextScene) {
  if (!DATA.scenes[nextScene]) return;
  state.scene = nextScene;
  state.selected = DATA.scenes[nextScene].nodeIds.find(visibleNode) || "input-ids";
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
  const ty = state.scene === "overview" ? 24 : Math.max(24, (visibleHeight - lastLayout.height * scale) / 2);
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

  const graphNode = event.target.closest("[data-node]");
  if (graphNode) {
    state.selected = graphNode.dataset.node;
    render();
  }
});

document.querySelector("#layerSlider").addEventListener("input", (event) => {
  state.layer = Number(event.target.value);
  render(true);
});

document.querySelector("#zoomIn").addEventListener("click", () => zoomBy(1.2));
document.querySelector("#zoomOut").addEventListener("click", () => zoomBy(0.84));
document.querySelector("#zoomFit").addEventListener("click", fitGraph);
document.querySelector("#backScene").addEventListener("click", () => openScene("overview"));
document.querySelector("#drillButton").addEventListener("click", (event) => {
  if (event.currentTarget.dataset.scene) openScene(event.currentTarget.dataset.scene);
});

render(true);
