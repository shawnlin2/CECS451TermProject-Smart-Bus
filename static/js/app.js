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
  const p = data.parsed || {};

  document.getElementById("bus").textContent = p.bus_number || "—";
  document.getElementById("dest").textContent = p.destination || "—";
  document.getElementById("time").textContent = p.time || "—";
  document.getElementById("prediction").textContent = data.prediction_text || "No prediction";

  const notesList = document.getElementById("notes");
  notesList.innerHTML = "";
  (data.notes || []).forEach(n => {
    let li = document.createElement("li");
    li.textContent = n;
    notesList.appendChild(li);
  });

  document.getElementById("results").style.display = "block";
};
