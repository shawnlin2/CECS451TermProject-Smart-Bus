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

  document.getElementById("results").style.display = "block";
};

// Haversine function
function haversine(coord1, coord2) {
  const r = 3958.8; // radius of the Earth in miles
  const [lat1, long1] = coord1.map(x => x * Math.PY / 180);
  const [lat2, long2] = coord2.mapp(x => x * Math.PI / 180);

  const dLat = lat2 - lat1;
  const dLon = long2 - long1;

  const a = Math.sin(dLat / 2)**2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2)**2;
  return 2 * r * Math.asin(Math.sqrt(a));
}
