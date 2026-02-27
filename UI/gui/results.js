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

function displayResultsList(results) {
	const list = document.getElementById("results-list");
	list.innerHTML = "";

	if (!results || results.length === 0) {
		list.innerHTML = '<div class="results-empty">No results found</div>';
		return;
	}

	results.forEach((result) => {
		const item = document.createElement("div");
		item.className = "results-item";
		item.innerHTML = `
			<div class="results-item-content">
				<div class="results-item-header">
					<span class="results-item-title">${result.dataset} / ${result.model}</span>
					<span class="results-item-date">${formatDate(result.timestamp)}</span>
				</div>
				<div class="results-item-stats">
					<span>${result.epochs} epoch${result.epochs !== 1 ? "s" : ""}</span>
					<span>loss ${result.best_loss.toFixed(4)}</span>
				</div>
			</div>
			<button class="results-item-delete" title="Delete this run">🗑</button>
		`;
		item.onclick = () => viewResult(result);

		// Delete button click handler (stop propagation to prevent opening the result)
		const deleteBtn = item.querySelector(".results-item-delete");
		deleteBtn.onclick = (e) => {
			e.stopPropagation();
			deleteResult(result);
		};

		list.appendChild(item);
	});
}

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
		`Delete run: ${result.dataset} / ${result.model}?\n\nThis cannot be undone.`
	);
	if (!confirmDelete) return;

	try {
		const res = await fetch(
			`/api/results/${result.dataset}/${result.model}/${result.timestamp}`,
			{method: "DELETE"}
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
