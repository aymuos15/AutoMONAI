let lossChart = null;
let metricsChart = null;
let currentResult = null;
let allResults = [];

async function loadResults() {
	try {
		const response = await fetch("/api/results");
		allResults = await response.json();
		displayResultsList(allResults);
	} catch (e) {
		console.error("Failed to load results:", e);
		document.getElementById("results-list").innerHTML =
			'<div class="results-empty">Failed to load results</div>';
	}
}

function filterResults() {
	const query = document.getElementById("results-search").value.toLowerCase();
	if (!query) {
		displayResultsList(allResults);
		return;
	}
	const filtered = allResults.filter(
		(r) =>
			r.dataset.toLowerCase().includes(query) ||
			r.model.toLowerCase().includes(query) ||
			r.timestamp.includes(query),
	);
	displayResultsList(filtered);
}

function _resultSummary(result) {
	const parts = [];
	const cfg = result.config || {};
	if (cfg.model && cfg.model !== "unet") parts.push(`model: ${cfg.model}`);
	if (cfg.dataset && cfg.dataset !== "Dataset001_Cellpose") parts.push(`dataset: ${cfg.dataset}`);
	if (result.epochs && result.epochs !== 1) parts.push(`epochs: ${result.epochs}`);
	if (cfg.batch_size && cfg.batch_size !== "4" && cfg.batch_size !== 4) parts.push(`batch_size: ${cfg.batch_size}`);
	if (cfg.optimizer && cfg.optimizer !== "adam") parts.push(`optimizer: ${cfg.optimizer}`);
	if (cfg.mixed_precision && cfg.mixed_precision !== "no") parts.push(`mixed_precision: ${cfg.mixed_precision}`);
	if (cfg.loss && cfg.loss !== "dice") parts.push(`loss: ${cfg.loss}`);
	return parts.length === 0 ? "base" : parts.join(" \u00b7 ");
}

function _escapeHtml(str) {
	const div = document.createElement("div");
	div.textContent = str;
	return div.innerHTML;
}

function displayResultsList(results) {
	const list = document.getElementById("results-list");

	if (!results || results.length === 0) {
		list.innerHTML = '<div class="results-empty">No results found</div>';
		return;
	}

	const isComplete = (r) => r.status === "complete";

	list.innerHTML = results.map((result, i) => {
		const summary = _resultSummary(result);
		const status = isComplete(result)
			? '<span class="cmd-link result-status-complete">Complete</span>'
			: '<span class="cmd-link result-status-progress">In Progress</span>';
		const stats = `${result.epochs} epoch${result.epochs !== 1 ? "s" : ""} \u00b7 loss ${result.best_loss.toFixed(4)} \u00b7 ${formatDate(result.timestamp)}`;
		const id = `result-${i}`;

		return `<div class="launch-bar config-card result-card" id="${id}">
			<div class="launch-config-line">
				<span class="config-text">${_escapeHtml(summary)}</span>
				<span class="card-actions">${status}<button type="button" class="cmd-link" onclick="viewResultByIndex(${i})">Charts</button><button type="button" class="cmd-link delete-link" onclick="deleteResultByIndex(${i})">Delete</button></span>
			</div>
			<div class="result-stats">${_escapeHtml(stats)}</div>
		</div>`;
	}).join("");
}

function viewResultByIndex(i) {
	if (allResults[i]) viewResult(allResults[i]);
}

function deleteResultByIndex(i) {
	if (allResults[i]) deleteResult(allResults[i]);
}

window.viewResultByIndex = viewResultByIndex;
window.deleteResultByIndex = deleteResultByIndex;

function viewResult(result) {
	currentResult = result;
	document.getElementById("results-list").style.display = "none";
	document.querySelector(".results-search").style.display = "none";
	document.getElementById("results-viewer").style.display = "block";

	document.getElementById("results-title").textContent =
		`${result.dataset} / ${result.model}`;

	drawLossChart(result.metrics);
	drawMetricsChart(result.metrics);
}

function closeResultsViewer() {
	document.getElementById("results-list").style.display = "";
	document.querySelector(".results-search").style.display = "";
	document.getElementById("results-viewer").style.display = "none";
	currentResult = null;

	if (lossChart) {
		lossChart.destroy();
		lossChart = null;
	}
	if (metricsChart) {
		metricsChart.destroy();
		metricsChart = null;
	}
}

async function deleteResult(result) {
	const confirmDelete = confirm(
		`Delete run: ${result.dataset} / ${result.model}?\n\nThis cannot be undone.`,
	);
	if (!confirmDelete) return;

	try {
		const res = await fetch(
			`/api/results/${result.dataset}/${result.model}/${result.timestamp}`,
			{ method: "DELETE" },
		);

		if (!res.ok) {
			alert("Failed to delete result");
			return;
		}

		// Refresh list
		loadResults();
	} catch (e) {
		alert(`Error: ${e.message}`);
	}
}

async function deleteCurrentResult() {
	if (!currentResult) return;
	deleteResult(currentResult);
	closeResultsViewer();
}


function chartColors() {
	const isDark = !document.documentElement.getAttribute("data-theme");
	return {
		text: isDark ? "#888" : "#666",
		grid: isDark ? "#1a1a1a" : "#eee",
		line: isDark ? "#f5f5f5" : "#0a0a0a",
		lineDim: isDark ? "#555" : "#aaa",
	};
}

function drawLossChart(metrics) {
	if (lossChart) lossChart.destroy();

	const ctx = document.getElementById("loss-chart").getContext("2d");
	const c = chartColors();

	lossChart = new Chart(ctx, {
		type: "line",
		data: {
			labels: metrics.map((m) => m.epoch),
			datasets: [
				{
					label: "Loss",
					data: metrics.map((m) => m.loss),
					borderColor: c.line,
					backgroundColor: "transparent",
					borderWidth: 1.5,
					tension: 0.3,
					pointRadius: 3,
					pointBackgroundColor: c.line,
					pointBorderColor: c.line,
					pointHoverRadius: 5,
				},
			],
		},
		options: chartOpts(c),
	});
}

function drawMetricsChart(metrics) {
	if (metricsChart) metricsChart.destroy();

	const names = Object.keys(metrics[0] || {}).filter(
		(k) => k !== "epoch" && k !== "loss",
	);
	if (names.length === 0) return;

	const ctx = document.getElementById("metrics-chart").getContext("2d");
	const c = chartColors();

	// Alternate between solid and dashed for multiple metrics
	const styles = [
		{ dash: [], width: 1.5 },
		{ dash: [4, 4], width: 1.5 },
		{ dash: [1, 3], width: 1.5 },
		{ dash: [8, 4], width: 1.5 },
	];

	const datasets = names.map((name, i) => {
		const s = styles[i % styles.length];
		return {
			label: name,
			data: metrics.map((m) => m[name]),
			borderColor: i === 0 ? c.line : c.lineDim,
			backgroundColor: "transparent",
			borderWidth: s.width,
			borderDash: s.dash,
			tension: 0.3,
			pointRadius: 3,
			pointBackgroundColor: i === 0 ? c.line : c.lineDim,
			pointBorderColor: i === 0 ? c.line : c.lineDim,
			pointHoverRadius: 5,
		};
	});

	metricsChart = new Chart(ctx, {
		type: "line",
		data: {
			labels: metrics.map((m) => m.epoch),
			datasets,
		},
		options: chartOpts(c, { yMin: 0, yMax: 1 }),
	});
}

function chartOpts(c, extra = {}) {
	return {
		responsive: true,
		maintainAspectRatio: true,
		plugins: {
			legend: {
				display: true,
				labels: {
					color: c.text,
					font: { family: "'JetBrains Mono', monospace", size: 11 },
					boxWidth: 12,
					boxHeight: 1,
					padding: 12,
				},
			},
		},
		scales: {
			y: {
				min: extra.yMin,
				max: extra.yMax,
				ticks: {
					color: c.text,
					font: { family: "'JetBrains Mono', monospace", size: 10 },
				},
				grid: { color: c.grid, drawBorder: false },
				border: { display: false },
			},
			x: {
				ticks: {
					color: c.text,
					font: { family: "'JetBrains Mono', monospace", size: 10 },
				},
				grid: { color: c.grid, drawBorder: false },
				border: { display: false },
			},
		},
	};
}

function formatDate(timestamp) {
	if (!timestamp) return "";
	const y = timestamp.slice(0, 4);
	const mo = timestamp.slice(4, 6);
	const d = timestamp.slice(6, 8);
	const h = timestamp.slice(9, 11);
	const mi = timestamp.slice(11, 13);
	return `${y}-${mo}-${d} ${h}:${mi}`;
}
