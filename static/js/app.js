// Global for bus graph 
let busGraph = {};
let busCoords = {};

async function loadBusNetwork() {
  try {
    const res = await fetch("/api/bus_network");
    if (!res.ok) return; // silently ignore if endpoint missing
    const data = await res.json();
    busGraph = data.graph || {};
    busCoords = data.coords || {};
  } catch (e) {
    // No network map available — fallback only
  }
}

const go = document.getElementById("go");
const input = document.getElementById("input");

go.onclick = async () => {
  const q = input.value.trim();
  if (!q) return;

  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: q })
  });

  const data = await res.json();
  if (!res.ok) {
    alert(data.error || "Request failed");
    return;
  }

  const p = data.parsed || {};

  document.getElementById("bus").textContent = p.bus_number || "—";
  document.getElementById("dest").textContent = p.destination || "—";
  document.getElementById("time").textContent = p.time || "—";
  document.getElementById("prediction").textContent = data.prediction_text || "No prediction";

  const notesList = document.getElementById("notes");
  notesList.innerHTML = "";
  (data.notes || []).forEach(n => {
    const li = document.createElement("li");
    li.textContent = n;
    notesList.appendChild(li);
  });

  const pathSummaryEl = document.getElementById("path-summary");
  const pathList = document.getElementById("path-list");
  pathList.innerHTML = "";

  const stops = data.display_stops || [];
  const fullCount = data.full_stop_count || 0;

  if (stops.length > 0) {
    pathSummaryEl.textContent = data.path_summary || `Showing ${stops.length} of ${fullCount} stops.`;
    stops.forEach((stopId, idx) => {
      const li = document.createElement("li");
      li.textContent = `Stop ${idx + 1}: ${stopId}`;
      pathList.appendChild(li);
    });
  } else {
    pathSummaryEl.textContent = "No path information available for this route.";
  }

  enhancePathDisplay(data);

  document.getElementById("results").style.display = "block";
};

// Haversine function
function haversine(coord1, coord2) {
  const r = 3958.8; // radius of the Earth in miles
  const lat1 = coord1.lat * Math.PI / 180;
  const long1 = coord1.lon * Math.PI / 180;
  const lat2 = coord2.lat * Math.PI / 180;
  const long2 = coord2.lon * Math.PI / 180;

  const dLat = lat2 - lat1;
  const dLon = long2 - long1;

  const a = Math.sin(dLat / 2)**2 + 
            Math.cos(lat1) * Math.cos(lat2) * 
            Math.sin(dLon / 2)**2;

  return 2 * r * Math.asin(Math.sqrt(a));
}

// Minimal Heap Implementation for A* 
class MinHeap {
  constructor() { this.items = []; } 
  push(item) {
    this.items.push(item);
    this.items.sort((a,b) => a.f - b.f);
  }
  pop() { return this.items.shift(); }
  isEmpty() { return this.items.length === 0; }
} 

// A* Algorithm 
function aStar(start, end, graph, coords) {
  if (!graph[start] || !graph[end]) return []; 

  const open = new MinHeap();
  open.push({ node: start, f: 0 });

  const cameFrom = {};
  const g = Object.fromEntries(Object.keys(graph).map(k => [k, Infinity]));
  g[start] = 0;

  while (!open.isEmpty()) {
    const current = open.pop().node;
    if (current === end) {
      const path = [];
      let temp = end;
      while (temp) {
        path.push(temp);
        temp = cameFrom[temp];
      }
      return path.reverse();
    }

    for (let neighbor in graph[current]) {
      const tentative = g[current] + graph[current][neighbor];
      if (tentative < g[neighbor]) {
        cameFrom[neighbor] = current;
        g[neighbor] = tentative;
        const f = tentative + haversine(coords[neighbor], coords[end]);
        open.push({ node: neighbor, f });
      }
    }
  }
  return [];
}

function buildSuggestedPathAStar(stops) {
  if (!stops || stops.length < 2) {
    return { summary: "Not enough stop data.", path: [] };
  }

  const start = stops[0];
  const end = stops[stops.length - 1];

  // Try A*
  const path = aStar(start, end, busGraph, busCoords);
  if (path.length > 1) {
    return {
      summary: `Best A* path from ${start} to ${end}:`,
      path
    };
  }

  return {
    summary: `Path toward destination (backend route)`,
    path: stops
  };
}

async function enhancePathDisplay(data) {
  const pathSummaryEl = document.getElementById("path-summary");
  const pathList = document.getElementById("path-list");
  pathList.innerHTML = "";

  const stops = data.display_stops || [];
  const destination = data.parsed.destination;
  const route = data.parsed.bus_number;
  const fullCount = data.full_stop_count || stops.length;

  if (!stops || stops.length === 0) {
    pathSummaryEl.textContent = "No path information available for this route.";
    return;
  }

  // Use A* path if graph is loaded
  let path = stops;
  let summary = destination && route
    ? `Path toward ${destination} on route ${route} — showing ${stops.length} of ${fullCount} stops`
    : `Path toward destination (backend route)`;

  if (Object.keys(busGraph).length > 0 && Object.keys(busCoords).length > 0) {
    const aStarPath = aStar(stops[0], stops[stops.length - 1], busGraph, busCoords);
    if (aStarPath && aStarPath.length > 1) {
      path = aStarPath;
      summary = `Best A* path from ${stops[0]} to ${stops[stops.length - 1]} — showing ${path.length} stops`;
    }
  }

  pathSummaryEl.textContent = summary;

  path.forEach((stopId, idx) => {
    const li = document.createElement("li");
    li.textContent = `Stop ${idx + 1}: ${stopId}`;
    pathList.appendChild(li);
  });
}
