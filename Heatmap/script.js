// ----------------------------
// IntelliWaste Heatmap Prototype
// ----------------------------

// Map init – New York
const map = L.map("map").setView([40.7128, -74.006], 12);

// Base map (OpenStreetMap)
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

// Legend control (bottom-right)
const legend = L.control({ position: "bottomright" });

legend.onAdd = function () {
  const div = L.DomUtil.create("div", "info legend");
  div.style.background = "rgba(15,23,42,0.9)";
  div.style.padding = "8px 10px";
  div.style.borderRadius = "10px";
  div.style.border = "1px solid rgba(148,163,184,0.6)";
  div.style.fontSize = "12px";
  div.style.color = "#e5e7eb";
  div.innerHTML = `
    <div style="margin-bottom:4px;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#9ca3af;">Legend</div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
      <span style="width:10px;height:10px;border-radius:50%;background:#4ade80;"></span> Low
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
      <span style="width:10px;height:10px;border-radius:50%;background:#facc15;"></span> Medium
    </div>
    <div style="display:flex;align-items:center;gap:6px;">
      <span style="width:10px;height:10px;border-radius:50%;background:#fb7185;"></span> High
    </div>
  `;
  return div;
};

legend.addTo(map);

// Mock data for layers (around New York)
const layersData = {
  overflow: {
    title: "Overflow Risk (24h)",
    points: [
      { lat: 40.715, lng: -74.01, level: "high", label: "Bin #21 – HIGH (87%)" },
      { lat: 40.708, lng: -74.0, level: "med", label: "Bin #33 – MED (61%)" },
      { lat: 40.72, lng: -73.995, level: "med", label: "Bin #07 – MED → HIGH in 12h" },
      { lat: 40.705, lng: -74.015, level: "low", label: "Bin #14 – LOW (32%)" },
    ],
    info: [
      "Bin #21 – HIGH (87%)",
      "Bin #33 – MED (61%)",
      "Bin #07 – MED → HIGH in 12h",
    ],
  },

  contamination: {
    title: "Contamination Hotspots (7d)",
    points: [
      { lat: 40.718, lng: -74.003, level: "high", label: "Bin #12 – 28% contamination" },
      { lat: 40.71, lng: -73.99, level: "high", label: "Bin #44 – 35% contamination" },
      { lat: 40.703, lng: -74.01, level: "med", label: "District C – mixed behavior" },
    ],
    info: [
      "Bin #12 – Plastic bin, contamination 28%",
      "Bin #44 – Paper bin, contamination 35%",
      "District C – candidate for education campaign",
    ],
  },

  density: {
    title: "Material Density (weekly pattern)",
    points: [
      { lat: 40.725, lng: -74.0, level: "med", label: "Campus zone – high paper" },
      { lat: 40.709, lng: -74.015, level: "high", label: "Food street – high organic" },
      { lat: 40.713, lng: -73.99, level: "low", label: "Business district – balanced" },
    ],
    info: [
      "Campus zone – paper density HIGH",
      "Food street – organic density HIGH",
      "Business district – balanced materials",
    ],
  },
};

// Layer group for circles
let circleGroup = L.layerGroup().addTo(map);

// Style helper for circle size + color
function circleStyle(level) {
  if (level === "high") {
    return {
      color: "#fb7185",
      fillColor: "#fb7185",
      radius: 350,
    };
  }
  if (level === "med") {
    return {
      color: "#facc15",
      fillColor: "#facc15",
      radius: 280,
    };
  }
  return {
    color: "#4ade80",
    fillColor: "#4ade80",
    radius: 220,
  };
}

// Draw circles for the selected layer
function drawPoints(data) {
  circleGroup.clearLayers();

  data.points.forEach((p) => {
    const style = circleStyle(p.level);

    const circle = L.circle([p.lat, p.lng], {
      radius: style.radius,
      color: style.color,
      fillColor: style.fillColor,
      fillOpacity: 0.45,
      weight: 0.8,
    }).addTo(circleGroup);

    // Popup on click
    circle.bindPopup(p.label);
  });
}

// Update right-side info cards
function updateInfo(list) {
  const box = document.getElementById("infoBox");
  box.innerHTML = "";

  list.forEach((item) => {
    const div = document.createElement("div");
    div.className = "dot-info";
    div.textContent = item;
    box.appendChild(div);
  });
}

// Main function to switch layers (called from HTML buttons)
function setLayer(name) {
  document.getElementById("layerTitle").textContent = layersData[name].title;

  // button active state
  document.querySelectorAll(".btn").forEach((btn) =>
    btn.classList.remove("active")
  );
  document
    .querySelector(`.btn[onclick="setLayer('${name}')"]`)
    .classList.add("active");

  // update map + info
  drawPoints(layersData[name]);
  updateInfo(layersData[name].info);
}

// Initial layer
setLayer("overflow");
