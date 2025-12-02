// Global for bus graph 
let busGraph = {};
let busCoords = {};

async function loadBusNetwork() {
  const res = await fetch("/api/bus_network");
  const data = await res.json();
  busGraph = data.graph || {};
  busCoords = data.coords || {};
}
loadBusNetwork();

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
  const [lat1, long1] = coord1.map(x => x * Math.PI / 180);
  const [lat2, long2] = coord2.map(x => x * Math.PI / 180);

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
  if (!graph[start || !graph[end]]) return[];

  const openSet = new MinHeap();
  openSet.push({ node: start, f: 0});

  const cameFrom = {};
  const gScore = {}; 
  Object.keys(graph).forEach(n => gScore[n] = Infinity);
  gScore[start] = 0; 

  while (!openSet.isEmpty()) {
    const current = openSet.pop().node;
    if (current === end) {
      // reconstruct path 
      const path = [];
      let temp = end;
      while (temp) {
        path.push(temp);
        temp = cameFrom[temp];
      }
      return path.reverse();
    }

    for (let neighbor in graph[current]) {
      const tentative_g = gScore[current] + graph[current][neighbor];
      if (tentative_g < gScore[neighbor]) {
        cameFrom[neighbor] = current;
        gScore[neighbor] = tentative_g;
        const f = tentative_g + haversine(coords[neighbor], coords[end]);
        openSet.push({ node: neighbor, f: f });

      }
    }
  }

  return []; // no path found
}

function buildSuggestedPathAStar(stopsFromBackend) {
  if (!stopsFromBackend || stopsFromBackend.length < 2) {
    return { 
      summary: "Not enough data for pathfinding.", 
      path: [] 
    };
  }

  const start = stopsFromBackend[0];
  const end = stopsFromBackend[stopsFromBackend.length - 1];

  if (!busGraph[start] || !busGraph[end]) {
    return { 
      summary: "Pathfinding unavailable for these stops.", 
      path: stopsFromBackend 
    };
  }

  const path = aStar(start, end, busGraph, busCoords);
  return {
    summary: `Best route from ${start} to ${end} (computed via A*)`,
    path: path
  };
}

// Replace old path display call
async function enhancePathDisplay(data) {
  const pathSummaryEl = document.getElementById("path-summary");
  const pathList = document.getElementById("path-list");
  pathList.innerHTML = "";

  const stops = data.display_stops || [];

  const result = buildSuggestedPathAStar(stops);

  pathSummaryEl.textContent = result.summary;

  result.path.forEach((stopId, idx) => {
    const li = document.createElement("li");
    li.textContent = `Stop ${idx + 1}: ${stopId}`;
    pathList.appendChild(li);
  });
}
