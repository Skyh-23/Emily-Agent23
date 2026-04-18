window.__nsbBooted = false;

const statsEl = document.getElementById("stats");
const graphContainer = document.getElementById("graph");
const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const nextBtn = document.getElementById("next-btn");
const refreshBtn = document.getElementById("refresh-btn");
const fitBtn = document.getElementById("fit-btn");
const zoomInBtn = document.getElementById("zoom-in-btn");
const zoomOutBtn = document.getElementById("zoom-out-btn");
const nightToggle = document.getElementById("night-toggle");

const inspector = {
  id: document.getElementById("node-id"),
  type: document.getElementById("node-type"),
  importance: document.getElementById("node-importance"),
  created: document.getElementById("node-created"),
  content: document.getElementById("node-content"),
};

const palette = {
  memory: "#374151",
  entity: "#0f766e",
  event: "#c2410c",
  document: "#7c3aed",
  node: "#4b5563",
};

const paletteNight = {
  memory: "#93c5fd",
  entity: "#34d399",
  event: "#fb923c",
  document: "#c4b5fd",
  node: "#cbd5e1",
};

const HIGHLIGHT_COLOR = "#a100ff";
const HIGHLIGHT_EDGE_LIGHT = "rgba(161, 0, 255, 0.96)";
const DIM_EDGE_LIGHT = "rgba(90, 90, 95, 0.42)";
const HIGHLIGHT_EDGE_DARK = "rgba(188, 96, 255, 0.98)";
const DIM_EDGE_DARK = "rgba(200, 210, 235, 0.42)";

let graphData = { nodes: [], links: [] };
let selectedId = null;
let selectedNeighborIds = new Set();
let searchResults = [];
let searchCursor = -1;
let lastGraphVersion = null;
let versionTimer = null;

let svg = null;
let rootG = null;
let linkSel = null;
let nodeSel = null;
let simulation = null;
let zoomBehavior = null;

function setStatus(text) {
  if (statsEl) statsEl.textContent = text;
}

function nodeColor(node) {
  const set = document.body.classList.contains("night") ? paletteNight : palette;
  return set[node.node_type] || set.node;
}

function endpointId(endpoint) {
  return typeof endpoint === "string" ? endpoint : endpoint?.id;
}

function refreshHighlightState() {
  selectedNeighborIds = new Set();
  if (!selectedId || !graphData?.links) return;
  for (const l of graphData.links) {
    const a = endpointId(l.source);
    const b = endpointId(l.target);
    if (a === selectedId && b) selectedNeighborIds.add(b);
    if (b === selectedId && a) selectedNeighborIds.add(a);
  }
}

function isDirectHighlightLink(link) {
  if (!selectedId) return false;
  const a = endpointId(link.source);
  const b = endpointId(link.target);
  return (a === selectedId && selectedNeighborIds.has(b)) || (b === selectedId && selectedNeighborIds.has(a));
}

function showNode(node) {
  selectedId = node.id;
  refreshHighlightState();
  inspector.id.textContent = node.id || "-";
  inspector.type.textContent = node.node_type || "-";
  inspector.importance.textContent = (node.importance ?? "-").toString();
  inspector.created.textContent = node.created_at || "-";
  inspector.content.textContent = node.content || "(empty)";
}

function normalizeGraph(raw) {
  const nodes = (raw.nodes || []).map((n) => ({ ...n }));
  const nodeIds = new Set(nodes.map((n) => n.id));
  const links = (raw.edges || [])
    .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
    .map((e) => ({
      source: e.source,
      target: e.target,
      relationship: e.relationship || "related",
      weight: Number(e.weight || 0.4),
    }));
  return { nodes, links };
}

function ensureCanvas2D() {
  if (svg) return;
  graphContainer.innerHTML = "";

  const rect = graphContainer.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height));

  svg = d3.select(graphContainer)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .style("display", "block");

  rootG = svg.append("g");
  rootG.append("g").attr("class", "links");
  rootG.append("g").attr("class", "nodes");

  zoomBehavior = d3.zoom()
    .scaleExtent([0.01, 80])
    .on("zoom", (event) => {
      rootG.attr("transform", event.transform);
    });

  svg.call(zoomBehavior);
}

function updateStyles() {
  if (!nodeSel || !linkSel) return;

  const night = document.body.classList.contains("night");

  nodeSel
    .attr("fill", (n) => (n.id === selectedId || selectedNeighborIds.has(n.id) ? HIGHLIGHT_COLOR : nodeColor(n)))
    .attr("stroke", (n) => (n.id === selectedId ? HIGHLIGHT_COLOR : "#ffffff"))
    .attr("stroke-width", (n) => (n.id === selectedId ? 3.2 : 2.6));

  linkSel
    .attr("stroke", (l) => {
      if (isDirectHighlightLink(l)) {
        return night ? HIGHLIGHT_EDGE_DARK : HIGHLIGHT_EDGE_LIGHT;
      }
      return night ? DIM_EDGE_DARK : DIM_EDGE_LIGHT;
    })
    .attr("stroke-width", (l) => (isDirectHighlightLink(l) ? 4.2 : 1.3 + Math.min(2.4, (l.weight || 0.2) * 1.4)));
}

function renderGraph() {
  ensureCanvas2D();

  const width = parseFloat(svg.attr("width"));
  const height = parseFloat(svg.attr("height"));

  if (simulation) simulation.stop();

  linkSel = rootG.select(".links")
    .selectAll("line")
    .data(graphData.links, (d) => `${endpointId(d.source)}-${endpointId(d.target)}-${d.relationship}`)
    .join("line");

  nodeSel = rootG.select(".nodes")
    .selectAll("circle")
    .data(graphData.nodes, (d) => d.id)
    .join("circle")
    .attr("r", (d) => (d.id === selectedId ? 28 : 26 + Math.min(22, (d.importance || 1) * 3.8)))
    .style("cursor", "pointer")
    .on("click", (_, d) => {
      showNode(d);
      updateStyles();
      focusNode(d);
    });

  nodeSel.append("title").text((d) => `${d.node_type || "node"}: ${d.content || d.id}`);

  simulation = d3.forceSimulation(graphData.nodes)
    .force("link", d3.forceLink(graphData.links).id((d) => d.id).distance((l) => 240 + (1 - (l.weight || 0.2)) * 280))
    .force("charge", d3.forceManyBody().strength(-900))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide().radius((d) => 42 + Math.min(28, (d.importance || 1) * 5.2)))
    .on("tick", () => {
      linkSel
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      nodeSel
        .attr("cx", (d) => d.x)
        .attr("cy", (d) => d.y);
    });

  nodeSel.call(
    d3.drag()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.2).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      })
  );

  updateStyles();
}

function focusNode(node) {
  if (!svg || !zoomBehavior || !node) return;
  const width = parseFloat(svg.attr("width"));
  const height = parseFloat(svg.attr("height"));
  const current = d3.zoomTransform(svg.node());
  const tx = width / 2 - current.k * (node.x || width / 2);
  const ty = height / 2 - current.k * (node.y || height / 2);
  svg.transition().duration(500).call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(current.k));
}

function fitGraph() {
  if (!svg || !zoomBehavior || !graphData.nodes.length) return;
  const width = parseFloat(svg.attr("width"));
  const height = parseFloat(svg.attr("height"));
  const minX = d3.min(graphData.nodes, (d) => d.x ?? width / 2) ?? 0;
  const maxX = d3.max(graphData.nodes, (d) => d.x ?? width / 2) ?? width;
  const minY = d3.min(graphData.nodes, (d) => d.y ?? height / 2) ?? 0;
  const maxY = d3.max(graphData.nodes, (d) => d.y ?? height / 2) ?? height;
  const dx = Math.max(1, maxX - minX);
  const dy = Math.max(1, maxY - minY);
  const k = Math.max(0.01, Math.min(12, 0.9 / Math.max(dx / width, dy / height)));
  const tx = width / 2 - k * (minX + dx / 2);
  const ty = height / 2 - k * (minY + dy / 2);
  svg.transition().duration(550).call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(k));
}

function zoomMaxIn() {
  if (!svg || !zoomBehavior) return;
  const current = d3.zoomTransform(svg.node());
  const k = 80;
  svg.transition().duration(350).call(zoomBehavior.transform, d3.zoomIdentity.translate(current.x, current.y).scale(k));
}

function zoomMaxOut() {
  if (!svg || !zoomBehavior) return;
  const width = parseFloat(svg.attr("width"));
  const height = parseFloat(svg.attr("height"));
  const k = 0.01;
  svg.transition().duration(450).call(zoomBehavior.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(k));
}

function searchNodes() {
  const q = (searchInput?.value || "").trim().toLowerCase();
  if (!q) {
    searchResults = [];
    searchCursor = -1;
    setStatus(`Nodes ${graphData.nodes.length} • Edges ${graphData.links.length} • 2D`);
    return;
  }

  searchResults = graphData.nodes.filter((n) => {
    const id = (n.id || "").toLowerCase();
    const type = (n.node_type || "").toLowerCase();
    const content = (n.content || "").toLowerCase();
    return id.includes(q) || type.includes(q) || content.includes(q);
  });

  searchCursor = searchResults.length ? 0 : -1;
  jumpToSearchResult();
}

function jumpToSearchResult() {
  if (!searchResults.length) {
    setStatus(`No match • Nodes ${graphData.nodes.length} • Edges ${graphData.links.length}`);
    return;
  }
  const node = searchResults[searchCursor];
  showNode(node);
  updateStyles();
  focusNode(node);
  setStatus(`Match ${searchCursor + 1}/${searchResults.length} • Nodes ${graphData.nodes.length} • Edges ${graphData.links.length} • 2D`);
}

function nextResult() {
  if (!searchResults.length) return;
  searchCursor = (searchCursor + 1) % searchResults.length;
  jumpToSearchResult();
}

function setTheme() {
  updateStyles();
}

function render(graphPayload) {
  graphData = normalizeGraph(graphPayload);
  refreshHighlightState();
  renderGraph();
  setTheme();
  setStatus(`Nodes ${graphData.nodes.length} • Edges ${graphData.links.length} • 2D`);

  if (selectedId) {
    const hit = graphData.nodes.find((n) => n.id === selectedId);
    if (hit) {
      showNode(hit);
      updateStyles();
    }
  }
}

async function loadGraph() {
  try {
    const res = await fetch("/api/graph", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    if (typeof payload?.graph?.version === "number") {
      lastGraphVersion = payload.graph.version;
    }
    render(payload.graph || { nodes: [], edges: [] });
  } catch (err) {
    setStatus(`Graph load failed: ${err.message}`);
    console.error(err);
  }
}

async function pollVersionAndRefresh() {
  try {
    const res = await fetch("/api/version", { cache: "no-store" });
    if (!res.ok) return;
    const payload = await res.json();
    if (typeof payload?.version !== "number") return;
    if (lastGraphVersion === null) {
      lastGraphVersion = payload.version;
      return;
    }
    if (payload.version > lastGraphVersion) {
      await loadGraph();
    }
  } catch (_err) {
    // Silent polling failure.
  }
}

function resizeGraph() {
  if (!svg) return;
  const rect = graphContainer.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(320, Math.floor(rect.height));
  svg.attr("width", width).attr("height", height).attr("viewBox", `0 0 ${width} ${height}`);
}

function bindEvents() {
  if (searchBtn) searchBtn.addEventListener("click", searchNodes);
  if (searchInput) {
    searchInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") searchNodes();
    });
  }
  if (nextBtn) nextBtn.addEventListener("click", nextResult);
  if (refreshBtn) refreshBtn.addEventListener("click", loadGraph);
  if (fitBtn) fitBtn.addEventListener("click", fitGraph);
  if (zoomInBtn) zoomInBtn.addEventListener("click", zoomMaxIn);
  if (zoomOutBtn) zoomOutBtn.addEventListener("click", zoomMaxOut);
  if (nightToggle) {
    nightToggle.addEventListener("change", () => {
      document.body.classList.toggle("night", !!nightToggle.checked);
      setTheme();
    });
  }
  window.addEventListener("resize", () => {
    resizeGraph();
    if (simulation) simulation.alpha(0.1).restart();
  });
}

async function bootstrap() {
  bindEvents();
  setStatus("Loading 2D graph...");
  await loadGraph();
  if (!versionTimer) {
    versionTimer = window.setInterval(() => {
      pollVersionAndRefresh();
    }, 1400);
  }
  window.__nsbBooted = true;
  if (window.__nsbBootWatchdog) clearTimeout(window.__nsbBootWatchdog);
}

bootstrap();
