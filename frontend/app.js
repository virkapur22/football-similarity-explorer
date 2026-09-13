const API_BASE = window.location.protocol === "file:"
  ? "http://127.0.0.1:8000"
  : `${window.location.protocol}//${window.location.hostname || "127.0.0.1"}:8000`;
const form = document.querySelector("#search-form");
const playerInput = document.querySelector("#player-input");
const playerSuggestions = document.querySelector("#player-suggestions");
const countInput = document.querySelector("#count-input");
const positionInput = document.querySelector("#position-input");
const button = document.querySelector("#search-button");
const message = document.querySelector("#message");
const matchesContainer = document.querySelector("#matches");
const apiStatus = document.querySelector("#api-status");
let currentPayload = null;
let activeView = "3d";
let targetProfile = null;
let comparedProfile = null;
let requestedComparison = null;
const profileCache = new Map();
const profileRequests = new Map();
const GOALKEEPER_STAT_KEYS = new Set([
  "saves_p90", "save_pct", "clean_sheet_pct", "psxg_p90",
  "goals_prevented_p90", "gk_pass_completion_pct", "crosses_stopped_pct",
  "defensive_actions_outside_box_p90", "avg_defensive_action_distance",
]);
let suggestionResults = [];
let activeSuggestionIndex = -1;
let suggestionTimer = null;
let suggestionController = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function closeSuggestions() {
  suggestionResults = [];
  activeSuggestionIndex = -1;
  playerSuggestions.hidden = true;
  playerSuggestions.innerHTML = "";
  playerInput.setAttribute("aria-expanded", "false");
  playerInput.removeAttribute("aria-activedescendant");
}

function selectSuggestion(index) {
  const player = suggestionResults[index];
  if (!player) return;
  playerInput.value = player.name;
  closeSuggestions();
  playerInput.focus();
}

function setActiveSuggestion(index) {
  if (!suggestionResults.length) return;
  activeSuggestionIndex = (index + suggestionResults.length) % suggestionResults.length;
  playerSuggestions.querySelectorAll(".suggestion-item").forEach((item, itemIndex) => {
    const active = itemIndex === activeSuggestionIndex;
    item.classList.toggle("active", active);
    item.setAttribute("aria-selected", String(active));
    if (active) {
      playerInput.setAttribute("aria-activedescendant", item.id);
      item.scrollIntoView({ block: "nearest" });
    }
  });
}

function renderSuggestions(players) {
  suggestionResults = players;
  activeSuggestionIndex = -1;
  if (!players.length) {
    playerSuggestions.innerHTML = '<div class="suggestion-empty">No matching players found</div>';
  } else {
    playerSuggestions.innerHTML = players.map((player, index) => `
      <button class="suggestion-item" id="player-suggestion-${index}" type="button" role="option" aria-selected="false" data-index="${index}">
        <span class="suggestion-name">${escapeHtml(player.name)}</span>
        <span class="suggestion-meta">${escapeHtml(player.team)} · ${escapeHtml(player.position_group)}</span>
      </button>
    `).join("");
    playerSuggestions.querySelectorAll(".suggestion-item").forEach((item) => {
      item.addEventListener("mousedown", (event) => event.preventDefault());
      item.addEventListener("click", () => selectSuggestion(Number(item.dataset.index)));
    });
  }
  playerSuggestions.hidden = false;
  playerInput.setAttribute("aria-expanded", "true");
}

async function fetchSuggestions(query) {
  suggestionController?.abort();
  suggestionController = new AbortController();
  playerSuggestions.innerHTML = '<div class="suggestion-empty">Finding players…</div>';
  playerSuggestions.hidden = false;
  playerInput.setAttribute("aria-expanded", "true");
  try {
    const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}`, { signal: suggestionController.signal });
    if (!response.ok) throw new Error("Search unavailable");
    const players = await response.json();
    if (playerInput.value.trim() === query) renderSuggestions(players);
  } catch (error) {
    if (error.name !== "AbortError") closeSuggestions();
  }
}

function queueSuggestions() {
  window.clearTimeout(suggestionTimer);
  const query = playerInput.value.trim();
  if (query.length < 2) {
    suggestionController?.abort();
    closeSuggestions();
    return;
  }
  suggestionTimer = window.setTimeout(() => fetchSuggestions(query), 160);
}

playerInput.addEventListener("input", queueSuggestions);
playerInput.addEventListener("focus", () => {
  if (playerInput.value.trim().length >= 2) queueSuggestions();
});
playerInput.addEventListener("keydown", (event) => {
  if (playerSuggestions.hidden) return;
  if (event.key === "ArrowDown") {
    event.preventDefault();
    setActiveSuggestion(activeSuggestionIndex + 1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    setActiveSuggestion(activeSuggestionIndex - 1);
  } else if (event.key === "Enter" && activeSuggestionIndex >= 0) {
    event.preventDefault();
    selectSuggestion(activeSuggestionIndex);
  } else if (event.key === "Escape") {
    closeSuggestions();
  }
});
document.addEventListener("pointerdown", (event) => {
  if (!event.target.closest(".control-player")) closeSuggestions();
});

function setMessage(text = "") {
  message.textContent = text;
}

function setApiStatus(ready) {
  apiStatus.classList.toggle("offline", !ready);
  apiStatus.lastChild.textContent = ready ? " API ready" : " API offline";
}

function formatPercent(value) {
  return `${Number(value).toFixed(1)}%`;
}

function formatRatioPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatOrdinal(value) {
  const number = Math.round(Number(value));
  const remainder = number % 100;
  if (remainder >= 11 && remainder <= 13) return `${number}th`;
  if (number % 10 === 1) return `${number}st`;
  if (number % 10 === 2) return `${number}nd`;
  if (number % 10 === 3) return `${number}rd`;
  return `${number}th`;
}

function reportStatsForProfile(profile) {
  return (profile.percentile_stats || [])
    .filter((stat) => profile.position_group === "GK" || !GOALKEEPER_STAT_KEYS.has(stat.key))
    .slice(0, 14);
}

function updateSummary(payload) {
  const { target } = payload;
  document.querySelector("#target-name").textContent = target.name;
  document.querySelector("#target-meta").textContent = `${target.team} · ${target.position_group} · ${target.archetype}`;
  // Keep the PCA explanation in the explainer below the map; the main summary
  // should answer the more useful question: what does this player actually do?
  const stats = document.querySelector("#overview-stats");
  stats.innerHTML = '<span class="summary-label overview-stats-label">At a glance</span>';
  (target.key_stats || []).slice(0, 4).forEach((stat) => {
    stats.insertAdjacentHTML("beforeend", `
      <div class="overview-stat">
        <span>${stat.label}</span>
        <strong>${Number(stat.value).toFixed(2)}</strong>
        <em>${stat.unit}</em>
      </div>
    `);
  });
}

function renderRadar() {
  if (!targetProfile || !window.Plotly) return;
  const styles = getComputedStyle(document.documentElement);
  const targetColor = styles.getPropertyValue("--red").trim();
  const comparisonColor = styles.getPropertyValue("--blue").trim();
  document.querySelector("#radar-title").textContent = comparedProfile && comparedProfile.name !== targetProfile.name
    ? `${targetProfile.name} vs ${comparedProfile.name}`
    : `${targetProfile.name}'s stat profile`;
  const targetStats = new Map(targetProfile.all_stats.map((item) => [item.key, item]));
  const metrics = targetProfile.comparison_keys.map((key) => targetStats.get(key)).filter(Boolean);
  const categories = metrics.map((item) => item.label);
  const normalise = (stat) => Math.min(100, (Number(stat.value) / Number(stat.scale_max)) * 100);
  const values = metrics.map(normalise);
  const traces = [{
    type: "scatterpolar", r: [...values, values[0]], theta: [...categories, categories[0]],
    customdata: [...metrics.map((item) => `${Number(item.value).toFixed(2)} ${item.unit}`), `${Number(metrics[0].value).toFixed(2)} ${metrics[0].unit}`],
    fill: "toself", name: targetProfile.name, line: { color: targetColor, width: 2 },
    fillcolor: "rgba(200, 109, 76, .16)", hovertemplate: "%{theta}<br>%{customdata}<extra>" + targetProfile.name + "</extra>",
  }];
  if (comparedProfile && comparedProfile.name !== targetProfile.name) {
    const lookup = new Map(comparedProfile.all_stats.map((item) => [item.key, item]));
    const otherStats = metrics.map((metric) => lookup.get(metric.key) || { value: 0, unit: metric.unit });
    const otherValues = otherStats.map((stat, index) => Math.min(100, (Number(stat.value) / Number(metrics[index].scale_max)) * 100));
    traces.push({
      type: "scatterpolar", r: [...otherValues, otherValues[0]], theta: [...categories, categories[0]],
      customdata: [...otherStats.map((item) => `${Number(item.value).toFixed(2)} ${item.unit}`), `${Number(otherStats[0].value).toFixed(2)} ${otherStats[0].unit}`],
      fill: "toself", name: comparedProfile.name, line: { color: comparisonColor, width: 2 },
      fillcolor: "rgba(79, 113, 104, .16)", hovertemplate: "%{theta}<br>%{customdata}<extra>" + comparedProfile.name + "</extra>",
    });
  }
  Plotly.react("radar", traces, {
    paper_bgcolor: "transparent", margin: { l: 70, r: 70, t: 48, b: 42 }, showlegend: traces.length > 1, legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
    font: { family: "Inter, system-ui, sans-serif", color: styles.getPropertyValue("--muted").trim(), size: 12 },
    polar: { bgcolor: "transparent", radialaxis: { visible: false, range: [0, 100], gridcolor: styles.getPropertyValue("--line").trim(), linecolor: styles.getPropertyValue("--line").trim() }, angularaxis: { gridcolor: styles.getPropertyValue("--line").trim(), linecolor: styles.getPropertyValue("--line").trim() } },
    hoverlabel: { bgcolor: styles.getPropertyValue("--panel").trim(), bordercolor: comparisonColor, font: { color: styles.getPropertyValue("--text").trim() } },
  }, { responsive: true, displaylogo: false, displayModeBar: false });
}

function compactReportMarkup(profile, kind) {
  const percentileStats = reportStatsForProfile(profile);
  const insightItems = (items = []) => items.slice(0, 3).map((item) => `
    <span class="compact-insight"><span>${escapeHtml(item.label)}</span><strong>${formatOrdinal(item.percentile)}</strong></span>
  `).join("");
  return `<article class="comparison-report-card comparison-report-${kind}" data-report-kind="${kind}">
    <header class="compact-report-header">
      <div>
        <span class="summary-label">${kind === "target" ? "Searched player" : "Compared player"}</span>
        <h2>${escapeHtml(profile.name)}</h2>
        <p>${escapeHtml(profile.team)} · ${escapeHtml(cleanLeague(profile.league))} · ${escapeHtml(profile.nationality)} · Age ${profile.age}</p>
      </div>
      <span class="profile-position">${escapeHtml(profile.position_group)}</span>
    </header>
    <p class="compact-report-description">${escapeHtml(profile.playstyle_description)}</p>
    <div class="compact-actuals" aria-label="Key actual statistics">
      ${(profile.key_stats || []).slice(0, 6).map((stat) => `<div><span>${escapeHtml(stat.label)}</span><strong>${Number(stat.value).toFixed(2)}</strong><em>${escapeHtml(stat.unit)}</em></div>`).join("")}
    </div>
    <div class="compact-report-section">
      <div class="compact-section-heading"><h3>Position-relative profile</h3><span>${Number(profile.minutes).toLocaleString()} min</span></div>
      <div class="compact-percentiles">${percentileStats.map((stat) => `
        <div class="compact-percentile-row">
          <span><b>${escapeHtml(stat.label)}</b><small>${Number(stat.value).toFixed(2)} ${escapeHtml(stat.unit)}</small></span>
          <div class="percentile-track"><div class="percentile-fill" style="width:${stat.percentile}%"></div></div>
          <strong>${formatOrdinal(stat.percentile)}</strong>
        </div>
      `).join("")}</div>
    </div>
    <div class="compact-insight-grid">
      <section><h3>Strengths</h3>${insightItems(profile.strengths)}</section>
      <section><h3>Lower-volume</h3>${insightItems(profile.development_areas)}</section>
    </div>
    <button type="button" class="compact-full-report" data-full-report="${encodeURIComponent(profile.name)}">View full player report <span aria-hidden="true">→</span></button>
  </article>`;
}

function renderDualComparison() {
  const container = document.querySelector("#comparison-reports");
  if (!targetProfile || !comparedProfile || comparedProfile.name === targetProfile.name) return;
  container.dataset.mobileReport = "comparison";
  container.innerHTML = `
    <div class="comparison-report-tabs" role="tablist" aria-label="Choose player report">
      <button type="button" data-report-tab="target" role="tab" aria-selected="false">${escapeHtml(targetProfile.name)}</button>
      <button type="button" class="active" data-report-tab="comparison" role="tab" aria-selected="true">${escapeHtml(comparedProfile.name)}</button>
    </div>
    <div class="comparison-report-grid has-comparison">
      ${compactReportMarkup(targetProfile, "target")}
      ${compactReportMarkup(comparedProfile, "comparison")}
    </div>`;
  container.querySelectorAll("[data-report-tab]").forEach((tab) => tab.addEventListener("click", () => {
    container.dataset.mobileReport = tab.dataset.reportTab;
    container.querySelectorAll("[data-report-tab]").forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
  }));
  container.querySelectorAll("[data-full-report]").forEach((control) => control.addEventListener("click", () => {
    document.querySelector("#explore-compare-dialog").close();
    openPlayerReport(decodeURIComponent(control.dataset.fullReport));
  }));
}

function renderProfile() {
  const profileName = document.querySelector("#profile-name");
  const profileMeta = document.querySelector("#profile-meta");
  const profilePosition = document.querySelector("#profile-position");
  const statsGrid = document.querySelector("#stats-grid");
  const statsNote = document.querySelector("#stats-note");
  const reportButton = document.querySelector("#open-dual-report");
  const profile = comparedProfile || targetProfile;
  if (!profile) return;
  profileName.textContent = profile.name;
  profileMeta.textContent = `${profile.team} · ${cleanLeague(profile.league)} · ${profile.archetype}`;
  profilePosition.textContent = profile.position_group;
  statsGrid.innerHTML = profile.key_stats.slice(0, 6).map((stat) => `
    <div class="stat-card"><span>${escapeHtml(stat.label)}</span><strong>${Number(stat.value).toFixed(2)}</strong><em>${escapeHtml(stat.unit)}</em></div>
  `).join("");
  statsNote.textContent = `${profile.stats_note} Based on ${profile.minutes.toLocaleString()} minutes.`;
  reportButton.hidden = !(targetProfile && comparedProfile && comparedProfile.name !== targetProfile.name);
  renderRadar();
}

document.querySelector("#open-dual-report").addEventListener("click", () => {
  if (!targetProfile || !comparedProfile || comparedProfile.name === targetProfile.name) return;
  renderDualComparison();
  document.querySelector("#explore-compare-dialog").showModal();
});

async function loadProfile(playerName, { comparison = false } = {}) {
  if (comparison) requestedComparison = playerName;
  if (!comparison) {
    document.querySelector("#profile-name").textContent = "Loading...";
  }
  try {
    let profile = profileCache.get(playerName);
    if (!profile) {
      if (!profileRequests.has(playerName)) {
        profileRequests.set(playerName, fetch(`${API_BASE}/player/${encodeURIComponent(playerName)}`)
          .then(async (response) => {
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Profile unavailable.");
            profileCache.set(playerName, data);
            return data;
          })
          .finally(() => profileRequests.delete(playerName)));
      }
      profile = await profileRequests.get(playerName);
    }
    if (comparison && requestedComparison !== playerName) return;
    if (comparison) comparedProfile = profile;
    else targetProfile = profile;
    renderProfile();
    document.querySelectorAll(".match-card").forEach((card) => {
      const selected = decodeURIComponent(card.dataset.player) === profile.name;
      card.classList.toggle("active", selected);
      card.setAttribute("aria-pressed", String(selected));
    });
  } catch (error) {
    if (!comparison) {
      document.querySelector("#profile-name").textContent = "Profile unavailable";
      document.querySelector("#profile-meta").textContent = error.message;
      document.querySelector("#profile-position").textContent = "--";
      document.querySelector("#stats-grid").innerHTML = "";
      document.querySelector("#stats-note").textContent = "";
      document.querySelector("#open-dual-report").hidden = true;
    }
  }
}

function renderMatches(payload) {
  const matches = payload.players
    .filter((player) => player.is_similar)
    .sort((a, b) => b.similarity - a.similarity);
  const container = document.querySelector("#matches");
  if (!matches.length) {
    container.innerHTML = '<div class="empty-state">No comparable players found.</div>';
    return;
  }
  container.innerHTML = matches.map((player, index) => `
    <button type="button" class="match-card" data-player="${encodeURIComponent(player.name)}" aria-pressed="false" aria-label="Compare ${player.name}, ${formatPercent(player.similarity)} similar">
      <span class="match-rank">${String(index + 1).padStart(2, "0")}</span>
      <div>
        <div class="match-name" title="${player.name}">${player.name}</div>
        <div class="match-meta">${player.team} · ${player.position_group}</div>
      </div>
      <div>
        <div class="match-score">${formatPercent(player.similarity)}</div>
        <div class="match-cosine">cos ${player.cosine_similarity.toFixed(4)}</div>
      </div>
    </button>
  `).join("");
  container.querySelectorAll(".match-card").forEach((card) => {
    card.addEventListener("click", () => loadProfile(decodeURIComponent(card.dataset.player), { comparison: true }));
  });
}

function renderPlot(payload) {
  const allContext = payload.players.filter((player) => !player.is_target && !player.is_similar);
  const similar = payload.players.filter((player) => player.is_similar);
  const target = payload.target;
  const focusPoints = [target, ...similar];
  const localRange = (key) => {
    const values = focusPoints.map((player) => Number(player[key]));
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const padding = Math.max((maximum - minimum) * 0.38, 0.8);
    return [minimum - padding, maximum + padding];
  };
  const ranges = { x: localRange("x"), y: localRange("y"), z: localRange("z") };
  const context = allContext.filter((player) =>
    player.x >= ranges.x[0] && player.x <= ranges.x[1]
    && player.y >= ranges.y[0] && player.y <= ranges.y[1]
    && player.z >= ranges.z[0] && player.z <= ranges.z[1]
  );
  const ratio = payload.pca.explained_variance_ratio;
  const styles = getComputedStyle(document.documentElement);
  const textColor = styles.getPropertyValue("--text").trim();
  const mutedColor = styles.getPropertyValue("--muted").trim();
  const lineColor = styles.getPropertyValue("--line").trim();
  const panelColor = styles.getPropertyValue("--panel").trim();
  const blueColor = styles.getPropertyValue("--blue").trim();
  const redColor = styles.getPropertyValue("--red").trim();
  const contextColor = styles.getPropertyValue("--context").trim();

  const pointTrace = activeView === "3d" ? "scatter3d" : "scatter";
  const coordinate = (points, key) => points.map((p) => p[key]);
  const labelPositions = ["top center", "bottom center", "middle right", "middle left", "top right", "bottom left"];
  const annotationOffsets = [
    { xshift: 0, yshift: 19 },
    { xshift: 0, yshift: -19 },
    { xshift: 22, yshift: 0 },
    { xshift: -22, yshift: 0 },
    { xshift: 17, yshift: 17 },
  ];
  const sceneAnnotations = [
    ...similar.map((player, index) => ({
      x: player.x, y: player.y, z: player.z,
      text: player.name,
      showarrow: false,
      font: { color: blueColor, size: 15, family: "Georgia, Times New Roman, serif" },
      bgcolor: "rgba(251, 248, 240, .82)",
      borderpad: 2,
      ...annotationOffsets[index % annotationOffsets.length],
    })),
    {
      x: target.x, y: target.y, z: target.z,
      text: target.name,
      showarrow: false,
      font: { color: redColor, size: 15, family: "Georgia, Times New Roman, serif" },
      bgcolor: "rgba(251, 248, 240, .86)",
      borderpad: 2,
      xshift: 0,
      yshift: 22,
    },
  ];
  const traces = [
    {
      type: pointTrace,
      mode: "markers",
      name: "All players",
      x: coordinate(context, "x"), y: coordinate(context, "y"), ...(activeView === "3d" ? { z: coordinate(context, "z") } : {}),
      text: context.map((p) => p.name),
      customdata: context.map((p) => [p.team, p.position_group]),
      hovertemplate: "%{text}<br>%{customdata[0]} · %{customdata[1]}<extra>Player pool</extra>",
      marker: { size: 2.4, color: contextColor, opacity: 0.14 },
    },
    {
      type: activeView === "3d" ? "scatter3d" : "scatter",
      mode: "lines",
      name: "Similarity links",
      x: payload.connections.flatMap((connection) => [target.x, payload.players.find((p) => p.name === connection.to).x, null]), y: payload.connections.flatMap((connection) => [target.y, payload.players.find((p) => p.name === connection.to).y, null]), ...(activeView === "3d" ? { z: payload.connections.flatMap((connection) => [target.z, payload.players.find((p) => p.name === connection.to).z, null]) } : {}),
      hoverinfo: "skip",
      line: { color: blueColor, width: 3 },
      showlegend: false,
    },
    {
      type: pointTrace,
      mode: activeView === "3d" ? "markers" : "markers+text",
      name: "Closest matches",
      x: coordinate(similar, "x"), y: coordinate(similar, "y"), ...(activeView === "3d" ? { z: coordinate(similar, "z") } : {}),
      text: similar.map((player) => player.name), textposition: similar.map((_, index) => labelPositions[index % labelPositions.length]),
      hovertemplate: "%{customdata[3]}<br>%{customdata[1]} · %{customdata[2]}<br><b>%{customdata[0]:.1f}% similar</b><extra>Closest match</extra>",
      customdata: similar.map((p) => [p.similarity, p.team, p.position_group, p.name]),
      textfont: { color: blueColor, size: 15, family: "Georgia, Times New Roman, serif" },
      marker: { size: 9.5, color: blueColor, opacity: 1, line: { color: panelColor, width: 2 } },
    },
    {
      type: pointTrace,
      mode: activeView === "3d" ? "markers" : "markers+text",
      name: "Selected player",
      x: [target.x], y: [target.y], ...(activeView === "3d" ? { z: [target.z] } : {}), text: [target.name], textposition: "top center",
      hovertemplate: "%{text}<extra>Selected player</extra>",
      textfont: { color: redColor, size: 15, family: "Georgia, Times New Roman, serif" },
      marker: { size: 14, symbol: "diamond", color: redColor, opacity: 1, line: { color: panelColor, width: 2.5 } },
    },
  ];

  const axis = (title, variance, range) => ({
    title: `${title} · ${formatRatioPercent(variance)}`,
    titlefont: { color: textColor, size: 12 },
    tickfont: { color: mutedColor, size: 10 },
    gridcolor: lineColor,
    zerolinecolor: lineColor,
    showbackground: false,
    range,
    autorange: false,
    nticks: 5,
  });

  Plotly.react("plot", traces, {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { l: 0, r: 0, t: 0, b: 0 },
    font: { family: "Inter, system-ui, sans-serif", color: textColor },
    showlegend: false,
    ...(activeView === "3d" ? { scene: {
      xaxis: axis("Profile axis 1", ratio[0], ranges.x), yaxis: axis("Profile axis 2", ratio[1], ranges.y), zaxis: axis("Profile axis 3", ratio[2], ranges.z),
      bgcolor: "transparent",
      aspectmode: "manual",
      aspectratio: { x: 1.2, y: 1, z: .9 },
      camera: { eye: { x: 1.6, y: 1.6, z: 1.25 }, projection: { type: "orthographic" } },
      annotations: sceneAnnotations,
    } } : { xaxis: { title: `Profile axis 1 · ${formatRatioPercent(ratio[0])}`, gridcolor: lineColor, zerolinecolor: lineColor, range: ranges.x, autorange: false }, yaxis: { title: `Profile axis 2 · ${formatRatioPercent(ratio[1])}`, gridcolor: lineColor, zerolinecolor: lineColor, range: ranges.y, autorange: false, scaleanchor: "x" } }),
    hoverlabel: { bgcolor: panelColor, bordercolor: blueColor, font: { color: textColor } },
  }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso3d", "select2d", "lasso2d"] });
  document.querySelector("#plot").on("plotly_click", (event) => {
    const point = event.points?.[0];
    const playerName = point?.data?.name === "Closest matches" ? point.customdata?.[3] : point?.text;
    if (playerName) recenterOn(playerName);
  });
  document.querySelector("#plot").on("plotly_hover", (event) => {
    const point = event.points?.[0];
    if (point?.data?.name === "Closest matches" && point.customdata?.[3]) loadProfile(point.customdata[3], { comparison: true });
  });
}

function recenterOn(playerName) {
  playerInput.value = playerName;
  loadMap();
}

async function loadMap(event) {
  event?.preventDefault();
  const player = playerInput.value.trim();
  const count = Math.min(20, Math.max(1, Number(countInput.value) || 5));
  if (!player) {
    setMessage("Enter a player name.");
    return;
  }
  button.disabled = true;
  button.textContent = "Loading map...";
  setMessage("");
  try {
    const params = new URLSearchParams({ n: String(count), same_position_only: String(positionInput.checked) });
    const response = await fetch(`${API_BASE}/map/${encodeURIComponent(player)}?${params}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Player could not be found.");
    setApiStatus(true);
    currentPayload = payload;
    comparedProfile = null;
    updateSummary(payload);
    renderPlot(payload);
    renderMatches(payload);
    await loadProfile(payload.target.name);
  } catch (error) {
    const connectionFailed = error instanceof TypeError;
    setApiStatus(!connectionFailed);
    setMessage(connectionFailed
      ? "Could not reach the player API. Start it with: uvicorn src.api.app:app --reload"
      : error.message);
    document.querySelector("#matches").innerHTML = '<div class="empty-state">Map unavailable.</div>';
  } finally {
    button.disabled = false;
    button.innerHTML = 'Explore player <span aria-hidden="true">→</span>';
  }
}

form.addEventListener("submit", (event) => {
  closeSuggestions();
  loadMap(event);
});
window.addEventListener("load", () => loadMap());
document.querySelectorAll(".view-button").forEach((viewButton) => {
  viewButton.setAttribute("aria-pressed", String(viewButton.classList.contains("active")));
  viewButton.addEventListener("click", () => {
    activeView = viewButton.dataset.view;
    document.querySelectorAll(".view-button").forEach((item) => {
      const selected = item === viewButton;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-pressed", String(selected));
    });
    if (currentPayload) renderPlot(currentPayload);
  });
});

// --- Scouting workspace ----------------------------------------------------
const scoutState = {
  options: null,
  results: [],
  loaded: false,
  page: 1,
  pageCount: 1,
  shortlist: JSON.parse(localStorage.getItem("player-scout-shortlist") || "[]").slice(0, 3),
};

const scoutForm = document.querySelector("#scout-form");
const scoutResults = document.querySelector("#scout-results");
const scoutCount = document.querySelector("#scout-count");

function optionMarkup(values, placeholder) {
  return `<option value="">${placeholder}</option>${values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
}

function cleanLeague(value) {
  return String(value || "").replace(/^[a-z]{2,3}\s+/i, "");
}

async function loadScoutOptions() {
  if (scoutState.options) return;
  const response = await fetch(`${API_BASE}/archetype-options`);
  if (!response.ok) throw new Error("The archetype filters could not be loaded.");
  scoutState.options = await response.json();
  document.querySelector("#scout-role").innerHTML = optionMarkup(scoutState.options.archetypes.map((archetype) => archetype.name), "Any statistical profile");
  document.querySelectorAll("#scout-role option").forEach((option, index) => {
    if (index) option.value = scoutState.options.archetypes[index - 1].id;
  });
  document.querySelector("#scout-position").innerHTML = optionMarkup(scoutState.options.positions, "All positions");
  document.querySelector("#scout-club").innerHTML = optionMarkup(scoutState.options.clubs, "All clubs");
  document.querySelector("#scout-league").innerHTML = optionMarkup(scoutState.options.leagues, "All leagues");
  document.querySelector("#scout-nationality").innerHTML = optionMarkup(scoutState.options.nationalities, "All nationalities");
  document.querySelector("#threshold-list").innerHTML = scoutState.options.stats.map((stat) => `
    <div class="threshold-field">
      <label for="threshold-${stat.key}">${escapeHtml(stat.label)}</label>
      <output id="threshold-output-${stat.key}">Any</output>
      <input id="threshold-${stat.key}" type="range" min="0" max="95" step="5" value="0" data-threshold-key="${stat.key}" aria-label="Minimum ${escapeHtml(stat.label)} percentile" />
    </div>
  `).join("");
  document.querySelectorAll("[data-threshold-key]").forEach((range) => {
    range.addEventListener("input", () => {
      document.querySelector(`#threshold-output-${range.dataset.thresholdKey}`).textContent = Number(range.value) ? `${range.value}th` : "Any";
    });
  });
  const sort = document.querySelector("#scout-sort");
  scoutState.options.stats.forEach((stat) => sort.insertAdjacentHTML("beforeend", `<option value="${stat.key}">${escapeHtml(stat.label)}</option>`));
}

function applyArchetypePreset(archetypeId) {
  const archetype = scoutState.options?.archetypes.find((item) => item.id === archetypeId);
  document.querySelector("#role-description").textContent = archetype?.description || "Choose an archetype to compare players with its statistical vector.";
  if (archetype?.positions?.length === 1) document.querySelector("#scout-position").value = archetype.positions[0];
}

function scoutParams() {
  const params = new URLSearchParams();
  const mappings = {
    q: "#scout-name", archetype: "#scout-role", position: "#scout-position",
    club: "#scout-club", league: "#scout-league", nationality: "#scout-nationality",
    min_age: "#scout-min-age", max_age: "#scout-max-age",
    min_minutes: "#scout-min-minutes",
  };
  Object.entries(mappings).forEach(([key, selector]) => {
    const value = document.querySelector(selector).value.trim();
    if (value) params.set(key, value);
  });
  const thresholds = [...document.querySelectorAll("[data-threshold-key]")]
    .filter((range) => Number(range.value) > 0)
    .map((range) => `${range.dataset.thresholdKey}:${range.value}`);
  if (thresholds.length) params.set("thresholds", thresholds.join(","));
  params.set("sort_by", document.querySelector("#scout-sort").value);
  params.set("sort_order", document.querySelector("#sort-direction").dataset.order);
  params.set("page", String(scoutState.page));
  params.set("page_size", "10");
  return params;
}

async function runScoutSearch(event, { preservePage = false } = {}) {
  event?.preventDefault();
  if (!preservePage) scoutState.page = 1;
  const keepCurrentResults = Boolean(scoutResults.querySelector(".scout-result"));
  if (keepCurrentResults) {
    scoutResults.classList.add("is-loading");
    scoutResults.setAttribute("aria-busy", "true");
  } else {
    scoutResults.innerHTML = '<div class="scout-state">Searching the player pool…</div>';
  }
  scoutCount.textContent = "Applying your recruitment brief…";
  try {
    const response = await fetch(`${API_BASE}/archetype-search?${scoutParams()}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Scouting search failed.");
    scoutState.results = payload.results;
    renderScoutResults(payload);
  } catch (error) {
    scoutCount.textContent = "Search unavailable";
    scoutResults.innerHTML = `<div class="scout-state">${escapeHtml(error.message)}</div>`;
  } finally {
    scoutResults.classList.remove("is-loading");
    scoutResults.removeAttribute("aria-busy");
  }
}

function renderScoutResults(payload) {
  scoutState.page = payload.page || 1;
  scoutState.pageCount = payload.page_count || 1;
  scoutCount.textContent = `${payload.count.toLocaleString()} players match · percentiles are position-relative`;
  const fitLabel = payload.score_label || "Archetype similarity";
  document.querySelector("#fit-column-label").textContent = fitLabel;
  document.querySelector("#fit-sort-option").textContent = fitLabel;
  if (!payload.results.length) {
    scoutResults.innerHTML = '<div class="scout-state">No players match this brief. Try lowering a percentile or widening the age range.</div>';
    document.querySelector("#scout-pagination").hidden = true;
    return;
  }
  scoutResults.innerHTML = payload.results.map((player, index) => `
    <article class="scout-result" data-report-player="${encodeURIComponent(player.name)}" tabindex="0" aria-label="Open report for ${escapeHtml(player.name)}">
      <div class="result-player">
        <span class="result-rank">${String(((scoutState.page - 1) * payload.page_size) + index + 1).padStart(2, "0")}</span>
        <div><strong>${escapeHtml(player.name)}</strong><span>${escapeHtml(player.team)} · ${escapeHtml(cleanLeague(player.league))}</span><div class="result-archetype">${player.minutes.toLocaleString()} minutes · ${escapeHtml(player.position)}</div></div>
      </div>
      <div class="result-stats">${player.stats.map((stat) => `<div class="result-stat"><span>${escapeHtml(stat.label)}</span><strong>${Number(stat.value).toFixed(2)}</strong><em>${formatOrdinal(stat.percentile)} percentile</em></div>`).join("")}</div>
      <div class="role-score"><strong>${player.archetype_similarity == null ? "—" : Number(player.archetype_similarity).toFixed(1)}</strong><span>${escapeHtml(player.archetype_similarity == null ? "Select archetype" : "Archetype similarity")}</span><span class="result-context">${player.cosine_similarity == null ? "" : `cos ${Number(player.cosine_similarity).toFixed(4)} · `}Age ${player.age} · ${player.nationality}</span></div>
      <button type="button" class="shortlist-button ${scoutState.shortlist.includes(player.name) ? "active" : ""}" data-shortlist-player="${encodeURIComponent(player.name)}" aria-label="${scoutState.shortlist.includes(player.name) ? "Remove from" : "Add to"} shortlist">${scoutState.shortlist.includes(player.name) ? "✓" : "+"}</button>
    </article>
  `).join("");
  const pagination = document.querySelector("#scout-pagination");
  pagination.hidden = false;
  document.querySelector("#scout-page-counter").textContent = `Page ${scoutState.page} of ${scoutState.pageCount}`;
  document.querySelector("#scout-previous").disabled = !payload.has_previous;
  document.querySelector("#scout-next").disabled = !payload.has_next;
  bindScoutResultEvents();
}

function bindScoutResultEvents() {
  document.querySelectorAll("[data-report-player]").forEach((row) => {
    row.addEventListener("click", () => openPlayerReport(decodeURIComponent(row.dataset.reportPlayer)));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); openPlayerReport(decodeURIComponent(row.dataset.reportPlayer)); }
    });
  });
  document.querySelectorAll("[data-shortlist-player]").forEach((control) => {
    control.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleShortlist(decodeURIComponent(control.dataset.shortlistPlayer));
    });
  });
}

function toggleShortlist(playerName) {
  const index = scoutState.shortlist.indexOf(playerName);
  if (index >= 0) {
    scoutState.shortlist.splice(index, 1);
  } else if (scoutState.shortlist.length < 3) {
    scoutState.shortlist.push(playerName);
  } else {
    scoutCount.textContent = "Your shortlist can hold up to three players. Remove one before adding another.";
    return;
  }
  localStorage.setItem("player-scout-shortlist", JSON.stringify(scoutState.shortlist));
  renderShortlist();
  document.querySelectorAll("[data-shortlist-player]").forEach((control) => {
    const selected = scoutState.shortlist.includes(decodeURIComponent(control.dataset.shortlistPlayer));
    control.classList.toggle("active", selected);
    control.textContent = selected ? "✓" : "+";
    control.setAttribute("aria-label", `${selected ? "Remove from" : "Add to"} shortlist`);
  });
}

function renderShortlist() {
  const tray = document.querySelector("#shortlist-tray");
  tray.hidden = !scoutState.shortlist.length;
  document.querySelector("#shortlist-count").textContent = `${scoutState.shortlist.length} player${scoutState.shortlist.length === 1 ? "" : "s"}`;
  document.querySelector("#shortlist-chips").innerHTML = scoutState.shortlist.map((name) => `<span class="shortlist-chip">${escapeHtml(name)}<button type="button" data-remove-shortlist="${encodeURIComponent(name)}" aria-label="Remove ${escapeHtml(name)}">×</button></span>`).join("");
  document.querySelectorAll("[data-remove-shortlist]").forEach((control) => control.addEventListener("click", () => toggleShortlist(decodeURIComponent(control.dataset.removeShortlist))));
  document.querySelector("#compare-shortlist").disabled = scoutState.shortlist.length < 2;
}

function insightMarkup(items) {
  return items.map((item) => `<div class="insight-item"><span>${escapeHtml(item.label)}</span><strong>${formatOrdinal(item.percentile)}</strong></div>`).join("");
}

function archetypeSimilarityMarkup(archetype) {
  if (!archetype?.primary) return "";
  const rows = archetype.matches.map((match, index) => `
    <button type="button" class="archetype-match-button${index === 0 ? " active" : ""}" data-archetype-match="${index}" aria-pressed="${index === 0}">
      <span>${escapeHtml(match.name)}</span><strong>${Number(match.similarity).toFixed(1)}</strong>
    </button>`).join("");
  return `<section class="role-fit-section" aria-labelledby="archetype-similarity-heading">
    <div class="role-fit-summary"><span class="summary-label" id="archetype-selection-label">Primary archetype</span><div><strong id="archetype-selection-score">${Number(archetype.primary.similarity).toFixed(1)}</strong><span><b id="archetype-similarity-heading">${escapeHtml(archetype.primary.name)}</b><em>Archetype similarity</em></span></div><p id="archetype-selection-description">${escapeHtml(archetype.primary.description)}</p></div>
    <div class="role-fit-list"><span>All eligible archetype matches</span>${rows}</div>
  </section>`;
}

function bindArchetypeMatchSelector(container, archetype) {
  const score = container.querySelector("#archetype-selection-score");
  const name = container.querySelector("#archetype-similarity-heading");
  const description = container.querySelector("#archetype-selection-description");
  const label = container.querySelector("#archetype-selection-label");
  container.querySelectorAll("[data-archetype-match]").forEach((control) => {
    control.addEventListener("click", () => {
      const match = archetype.matches[Number(control.dataset.archetypeMatch)];
      if (!match) return;
      score.textContent = Number(match.similarity).toFixed(1);
      name.textContent = match.name;
      description.textContent = match.description;
      label.textContent = match.id === archetype.primary.id ? "Primary archetype" : "Selected archetype";
      container.querySelectorAll("[data-archetype-match]").forEach((item) => {
        const selected = item === control;
        item.classList.toggle("active", selected);
        item.setAttribute("aria-pressed", String(selected));
      });
    });
  });
}

async function openPlayerReport(playerName) {
  const dialog = document.querySelector("#player-dialog");
  const container = document.querySelector("#player-report");
  container.innerHTML = '<div class="scout-state">Building player report…</div>';
  if (!dialog.open) dialog.showModal();
  try {
    const response = await fetch(`${API_BASE}/player/${encodeURIComponent(playerName)}`);
    const profile = await response.json();
    if (!response.ok) throw new Error(profile.detail || "Player report unavailable.");
    const percentileStats = reportStatsForProfile(profile);
    container.innerHTML = `<article class="player-report">
      <header class="report-header"><div><span class="summary-label">Player report</span><h2>${escapeHtml(profile.name)}</h2><p class="report-meta">${escapeHtml(profile.team)} · ${escapeHtml(cleanLeague(profile.league))} · ${escapeHtml(profile.nationality)} · Age ${profile.age} · ${profile.minutes.toLocaleString()} minutes</p></div><span class="report-position">${escapeHtml(profile.position_group)}</span></header>
      <p class="report-description">${escapeHtml(profile.playstyle_description)}</p>
      ${archetypeSimilarityMarkup(profile.archetype_similarity)}
      <section class="report-radar-wrap"><div class="report-radar-heading"><h3>Playing profile</h3><span>Actual values · stat-specific scale</span></div><div class="report-radar" id="report-radar" aria-label="${escapeHtml(profile.name)} playing profile radar"></div></section>
      <div class="report-grid">
        <section><span class="section-index">Position-relative output</span><h2>Percentile profile</h2><div class="percentile-list">${percentileStats.map((stat) => `<div class="percentile-row"><span>${escapeHtml(stat.label)} · ${Number(stat.value).toFixed(2)}</span><div class="percentile-track"><div class="percentile-fill" style="width:${stat.percentile}%"></div></div><strong>${formatOrdinal(stat.percentile)}</strong></div>`).join("")}</div></section>
        <aside>
          <div class="insight-block"><h3>Standout strengths</h3><div class="insight-list">${insightMarkup(profile.strengths)}</div></div>
          <div class="insight-block"><h3>Lower-volume areas</h3><div class="insight-list">${insightMarkup(profile.development_areas)}</div></div>
          <div class="insight-block"><h3>Similar players</h3><div class="insight-list">${profile.similar_players.map((player) => `<button type="button" class="insight-item similar-report-player" data-similar-report="${encodeURIComponent(player.name)}"><span>${escapeHtml(player.name)}</span><strong>${Number(player.similarity).toFixed(1)}%</strong></button>`).join("")}</div></div>
          <div class="report-actions"><button type="button" class="secondary-action report-shortlist-action" data-report-shortlist="${encodeURIComponent(profile.name)}">${scoutState.shortlist.includes(profile.name) ? "Remove from shortlist" : "Add to shortlist"}</button><button type="button" class="report-map-action" data-open-map-player="${encodeURIComponent(profile.name)}">Open in similarity map →</button></div>
        </aside>
      </div>
    </article>`;
    bindArchetypeMatchSelector(container, profile.archetype_similarity);
    container.querySelectorAll("[data-similar-report]").forEach((control) => control.addEventListener("click", () => openPlayerReport(decodeURIComponent(control.dataset.similarReport))));
    container.querySelector("[data-report-shortlist]").addEventListener("click", (event) => {
      toggleShortlist(profile.name);
      event.currentTarget.textContent = scoutState.shortlist.includes(profile.name) ? "Remove from shortlist" : "Add to shortlist";
    });
    container.querySelector("[data-open-map-player]").addEventListener("click", () => {
      dialog.close();
      document.querySelector('[data-product-view="explore"]').click();
      playerInput.value = profile.name;
      loadMap();
    });
    const statLookup = new Map(profile.all_stats.map((stat) => [stat.key, stat]));
    const radarStats = profile.comparison_keys.map((key) => statLookup.get(key)).filter(Boolean);
    if (radarStats.length && window.Plotly) {
      const labels = radarStats.map((stat) => stat.label);
      const values = radarStats.map((stat) => Math.min(100, Number(stat.value) / Number(stat.scale_max) * 100));
      Plotly.react("report-radar", [{ type: "scatterpolar", mode: "lines+markers", fill: "toself", r: [...values, values[0]], theta: [...labels, labels[0]], customdata: [...radarStats.map((stat) => `${Number(stat.value).toFixed(2)} ${stat.unit}`), `${Number(radarStats[0].value).toFixed(2)} ${radarStats[0].unit}`], line: { color: "#c86d4c", width: 2 }, marker: { color: "#c86d4c", size: 5 }, fillcolor: "rgba(200,109,76,.15)", hovertemplate: "%{theta}<br>%{customdata}<extra>" + profile.name + "</extra>" }], { paper_bgcolor: "transparent", showlegend: false, margin: { l: 75, r: 75, t: 36, b: 36 }, font: { family: "Inter, system-ui, sans-serif", color: "#6f776f", size: 10 }, polar: { bgcolor: "transparent", radialaxis: { visible: false, range: [0, 100] }, angularaxis: { gridcolor: "#cfc4af", linecolor: "#cfc4af" } } }, { responsive: true, displayModeBar: false, displaylogo: false });
    }
  } catch (error) {
    container.innerHTML = `<div class="scout-state">${escapeHtml(error.message)}</div>`;
  }
}

async function compareShortlist() {
  if (scoutState.shortlist.length < 2) return;
  const dialog = document.querySelector("#compare-dialog");
  const heading = dialog.querySelector(".compare-dialog-heading");
  let comparison = dialog.querySelector(".shortlist-comparison");
  if (!comparison) {
    comparison = document.createElement("div");
    comparison.className = "shortlist-comparison";
    dialog.appendChild(comparison);
  }
  comparison.innerHTML = '<div class="scout-state">Comparing shortlisted players…</div>';
  dialog.showModal();
  try {
    const profiles = await Promise.all(scoutState.shortlist.map(async (name) => {
      const response = await fetch(`${API_BASE}/player/${encodeURIComponent(name)}`);
      if (!response.ok) throw new Error("One shortlisted report is unavailable.");
      return response.json();
    }));
    const sharedKeys = profiles[0].comparison_keys.slice(0, 7);
    comparison.dataset.count = String(profiles.length);
    comparison.innerHTML = profiles.map((profile) => {
      const stats = new Map(profile.percentile_stats.map((stat) => [stat.key, stat]));
      return `<article class="comparison-player"><header><span class="comparison-player-index">${String(profiles.indexOf(profile) + 1).padStart(2, "0")}</span><div><h3>${escapeHtml(profile.name)}</h3><p>${escapeHtml(profile.team)} · Age ${profile.age} · ${escapeHtml(profile.position_group)}</p></div></header>${sharedKeys.map((key) => { const stat = stats.get(key); return stat ? `<div class="comparison-stat"><span>${escapeHtml(stat.label)}</span><strong>${Number(stat.value).toFixed(2)}</strong><em>${formatOrdinal(stat.percentile)} percentile</em></div>` : `<div class="comparison-stat is-unavailable"><span>${escapeHtml(key.replaceAll("_", " "))}</span><strong>—</strong><em>Not available</em></div>`; }).join("")}</article>`;
    }).join("");
    heading.querySelector("p").textContent = `${profiles.length} players · Actual values with position-relative percentiles.`;
  } catch (error) {
    comparison.innerHTML = `<div class="scout-state">${escapeHtml(error.message)}</div>`;
  }
}

document.querySelectorAll("[data-product-view]").forEach((control) => {
  control.addEventListener("click", async () => {
    const view = control.dataset.productView;
    document.querySelectorAll("[data-product-view]").forEach((button) => {
      const active = button === control;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    document.querySelector("#explore-view").hidden = view !== "explore";
    document.querySelector("#scout-view").hidden = view !== "scout";
    if (view === "scout" && !scoutState.loaded) {
      try {
        await loadScoutOptions();
        scoutState.loaded = true;
        await runScoutSearch();
      } catch (error) {
        scoutResults.innerHTML = `<div class="scout-state">${escapeHtml(error.message)}</div>`;
      }
    }
  });
});

document.querySelector("#scout-role").addEventListener("change", (event) => applyArchetypePreset(event.target.value));
scoutForm.addEventListener("submit", runScoutSearch);
document.querySelector("#scout-previous").addEventListener("click", () => {
  if (scoutState.page <= 1) return;
  scoutState.page -= 1;
  runScoutSearch(null, { preservePage: true });
});
document.querySelector("#scout-next").addEventListener("click", () => {
  if (scoutState.page >= scoutState.pageCount) return;
  scoutState.page += 1;
  runScoutSearch(null, { preservePage: true });
});
document.querySelector("#scout-sort").addEventListener("change", () => scoutState.loaded && runScoutSearch());
document.querySelector("#sort-direction").addEventListener("click", (event) => {
  const next = event.currentTarget.dataset.order === "desc" ? "asc" : "desc";
  event.currentTarget.dataset.order = next;
  event.currentTarget.textContent = next === "desc" ? "↓" : "↑";
  event.currentTarget.setAttribute("aria-label", `Sort ${next === "desc" ? "descending" : "ascending"}`);
  runScoutSearch();
});
document.querySelector("#scout-reset").addEventListener("click", () => {
  scoutForm.reset();
  document.querySelectorAll("[data-threshold-key]").forEach((range) => { range.value = 0; range.dispatchEvent(new Event("input")); });
  document.querySelector("#role-description").textContent = "Choose an archetype to compare players with its statistical vector.";
  runScoutSearch();
});
document.querySelector("#compare-shortlist").addEventListener("click", compareShortlist);
document.querySelectorAll("[data-close-dialog]").forEach((control) => control.addEventListener("click", () => document.querySelector(`#${control.dataset.closeDialog}`).close()));
document.querySelectorAll("dialog").forEach((dialog) => dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); }));
renderShortlist();
const initialParams = new URLSearchParams(window.location.search);
if (initialParams.get("view") === "scout") {
  document.querySelector('[data-product-view="scout"]').click();
  if (initialParams.get("player")) window.setTimeout(() => openPlayerReport(initialParams.get("player")), 400);
}
