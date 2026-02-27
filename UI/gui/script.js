async function loadDatasets() {
	try {
		const response = await fetch("/api/datasets");
		const datasets = await response.json();
		const select = document.getElementById("dataset");
		select.innerHTML = "";
		for (const [key, val] of Object.entries(datasets)) {
			const opt = document.createElement("option");
			opt.value = key;
			opt.textContent = `${key} (${val.name})`;
			select.appendChild(opt);
		}
		updateCommand();
	} catch (e) {
		console.error("Failed to load datasets:", e);
	}
}

async function loadDatasetClasses() {
	try {
		const response = await fetch("/api/models");
		const _data = await response.json();

		const trainSelect = document.getElementById("train_dataset");
		const inferenceSelect = document.getElementById("inference_dataset");

		for (const select of [trainSelect, inferenceSelect]) {
			select.innerHTML = "";
		}

		const classes = [
			"Dataset",
			"CacheDataset",
			"PersistentDataset",
			"SmartCacheDataset",
		];
		for (const className of classes) {
			const optT = document.createElement("option");
			optT.value = className;
			optT.textContent = className;
			trainSelect.appendChild(optT);

			const optI = document.createElement("option");
			optI.value = className;
			optI.textContent = className;
			inferenceSelect.appendChild(optI);
		}
		updateCommand();
	} catch (e) {
		console.error("Failed to load dataset classes:", e);
	}
}

async function loadModels() {
	try {
		const response = await fetch("/api/models");
		const models = await response.json();
		const select = document.getElementById("model");
		select.innerHTML = "";
		for (const [key, val] of Object.entries(models)) {
			const opt = document.createElement("option");
			opt.value = key;
			opt.textContent = `${val.name}`;
			select.appendChild(opt);
		}
		updateCommand();
	} catch (e) {
		console.error("Failed to load models:", e);
	}
}

function formatRawCommand(cmd) {
	// Parse command into flags and arguments
	const parts = cmd.split(" --");
	const baseCmd = parts[0];

	// Group flags by category
	const groups = {
		dataset: [],
		model: [],
		metrics: [],
		training: [],
		image: [],
		preprocessing: [],
		augmentation: [],
		output: [],
	};

	for (let i = 1; i < parts.length; i++) {
		const [flag, ...argParts] = parts[i].split(" ");
		const arg = argParts.join(" ");
		const flagArg = { flag, arg };

		if (["dataset", "model"].includes(flag)) {
			groups.dataset.push(flagArg);
		} else if (["metrics", "loss"].includes(flag)) {
			groups.metrics.push(flagArg);
		} else if (["epochs", "batch_size", "lr", "num_workers", "optimizer", "mixed_precision", "scheduler", "early_stopping", "patience"].includes(flag)) {
			groups.training.push(flagArg);
		} else if (["img_size"].includes(flag)) {
			groups.image.push(flagArg);
		} else if (["norm", "crop"].includes(flag)) {
			groups.preprocessing.push(flagArg);
		} else if (flag.includes("aug")) {
			groups.augmentation.push(flagArg);
		} else if (["output_dir", "device"].includes(flag)) {
			groups.output.push(flagArg);
		}
	}

	// Build formatted raw command
	let raw = `${baseCmd} \\\n`;

	const groupOrder = [
		"dataset",
		"metrics",
		"training",
		"image",
		"preprocessing",
		"augmentation",
		"output",
	];
	for (const groupName of groupOrder) {
		const items = groups[groupName];
		if (items.length === 0) continue;

		raw += `  `;
		for (let j = 0; j < items.length; j++) {
			const { flag, arg } = items[j];
			raw += `--${flag}`;
			if (arg) {
				raw += ` ${arg}`;
			}
			if (j < items.length - 1) {
				raw += ` `;
			}
		}
		raw += ` \\\n`;
	}

	// Remove trailing backslash and newline from last line
	raw = raw.replace(/ \\\n$/, "");

	return raw;
}

function formatCommand(cmd) {
	// Parse command into flags and arguments
	const parts = cmd.split(" --");
	const baseCmd = parts[0];

	// Group flags by category
	const groups = {
		dataset: [],
		model: [],
		metrics: [],
		training: [],
		image: [],
		preprocessing: [],
		augmentation: [],
		output: [],
	};

	for (let i = 1; i < parts.length; i++) {
		const [flag, ...argParts] = parts[i].split(" ");
		const arg = argParts.join(" ");
		const flagArg = { flag, arg };

		if (["dataset", "model"].includes(flag)) {
			groups.dataset.push(flagArg);
		} else if (["metrics", "loss"].includes(flag)) {
			groups.metrics.push(flagArg);
		} else if (["epochs", "batch_size", "lr", "num_workers", "optimizer", "mixed_precision", "scheduler", "early_stopping", "patience"].includes(flag)) {
			groups.training.push(flagArg);
		} else if (["img_size"].includes(flag)) {
			groups.image.push(flagArg);
		} else if (["norm", "crop"].includes(flag)) {
			groups.preprocessing.push(flagArg);
		} else if (flag.includes("aug")) {
			groups.augmentation.push(flagArg);
		} else if (["output_dir", "device"].includes(flag)) {
			groups.output.push(flagArg);
		}
	}

	// Build formatted HTML
	let html = `<span class="cmd">${baseCmd}</span> \\<br>`;

	const groupOrder = [
		"dataset",
		"metrics",
		"training",
		"image",
		"preprocessing",
		"augmentation",
		"output",
	];
	for (const groupName of groupOrder) {
		const items = groups[groupName];
		if (items.length === 0) continue;

		html += `  `;
		for (let j = 0; j < items.length; j++) {
			const { flag, arg } = items[j];
			html += `<span class="flag">--${flag}</span>`;
			if (arg) {
				html += ` <span class="arg">${arg}</span>`;
			}
			if (j < items.length - 1) {
				html += ` `;
			}
		}
		html += ` \\<br>`;
	}

	// Remove trailing backslash and br from last line
	html = html.replace(/ \\<br>$/, "");

	return html;
}

function updateCommand() {
	const dataset = document.getElementById("dataset").value;
	const model = document.getElementById("model").value;
	const trainDataset = document.getElementById("train_dataset").value;
	const inferenceDataset = document.getElementById("inference_dataset").value;
	const output_dir = document.getElementById("output_dir").value;

	const epochs = document.getElementById("epochs").value;
	const batch_size = document.getElementById("batch_size").value;
	const device = document.getElementById("device").value;
	const lr = document.getElementById("lr").value;
	const img_size = document.getElementById("img_size").value;
	const num_workers = document.getElementById("num_workers").value;

	const aug_rotate = document.getElementById("aug_rotate").checked;
	const aug_rotate_prob = document.getElementById("aug_rotate_prob").value;
	const aug_flip = document.getElementById("aug_flip").checked;
	const aug_flip_prob = document.getElementById("aug_flip_prob").value;

	// Auto-enable augmentation if any option is checked
	const augmentEnabled = aug_rotate || aug_flip;
	console.log("augment enabled (auto):", augmentEnabled);

	const norm_minmax = document.getElementById("norm_minmax").checked;
	const norm_zscore = document.getElementById("norm_zscore").checked;
	const crop_center = document.getElementById("crop_center").checked;
	const crop_random = document.getElementById("crop_random").checked;

	const loss = document.getElementById("loss").value;
	// Collect selected metrics from checkboxes
	const metricsCheckboxes = document.querySelectorAll('input[name="metrics"]:checked');
	const metrics = Array.from(metricsCheckboxes).map(cb => cb.value).join(' ');

	const optimizer = document.getElementById("optimizer").value;
	const mixed_precision = document.getElementById("mixed_precision").value;
	const scheduler = document.getElementById("scheduler").value;
	const patience = document.getElementById("patience").value;
	const early_stopping = patience && patience.toUpperCase() !== "X";

	document.getElementById("aug_rotate_prob_val").textContent = aug_rotate_prob;
	document.getElementById("aug_flip_prob_val").textContent = aug_flip_prob;

	let command = `python3 -m src.run --dataset ${dataset} --model ${model} --metrics ${metrics} --loss ${loss}`;

	if (trainDataset && trainDataset !== "Dataset") {
		command += ` --train_dataset_class ${trainDataset}`;
	}
	if (inferenceDataset && inferenceDataset !== "Dataset") {
		command += ` --inference_dataset_class ${inferenceDataset}`;
	}

	command += ` --epochs ${epochs} --batch_size ${batch_size} --lr ${lr} --img_size ${img_size} --num_workers ${num_workers} --output_dir ${output_dir} --device ${device}`;

	const normOpts = [];
	if (norm_minmax) normOpts.push("minmax");
	if (norm_zscore) normOpts.push("zscore");
	if (normOpts.length > 0) {
		command += ` --norm ${normOpts.join(" ")}`;
	}

	const cropOpts = [];
	if (crop_center) cropOpts.push("center");
	if (crop_random) cropOpts.push("random");
	if (cropOpts.length > 0) {
		command += ` --crop ${cropOpts.join(" ")}`;
	}

	if (augmentEnabled) {
		command += " --augment";
		if (aug_rotate) {
			command += ` --aug_rotate --aug_rotate_prob ${aug_rotate_prob}`;
		}
		if (aug_flip) {
			command += ` --aug_flip --aug_flip_prob ${aug_flip_prob}`;
		}
	}

	if (optimizer !== "adam") {
		command += ` --optimizer ${optimizer}`;
	}
	if (mixed_precision !== "no") {
		command += ` --mixed_precision ${mixed_precision}`;
	}
	if (scheduler !== "none") {
		command += ` --scheduler ${scheduler}`;
	}
	if (early_stopping) {
		command += ` --early_stopping --patience ${patience}`;
	}

	const formattedHtml = formatCommand(command);
	const rawCmd = formatRawCommand(command);

	// Command sub-tab display
	const cmdDisplay = document.getElementById("command-display");
	if (cmdDisplay) {
		cmdDisplay.innerHTML = formattedHtml;
		cmdDisplay.dataset.raw = rawCmd;
	}

	// Command modal display
	const cmdModalDisplay = document.getElementById("command-modal-display");
	if (cmdModalDisplay) {
		cmdModalDisplay.innerHTML = formattedHtml;
		cmdModalDisplay.dataset.raw = rawCmd;
	}

	updateSummaryPanel(
		norm_minmax,
		norm_zscore,
		crop_center,
		crop_random,
		augmentEnabled,
		aug_rotate,
		aug_rotate_prob,
		aug_flip,
		aug_flip_prob,
	);
}

function updateSummaryPanel(
	normMinmax,
	normZscore,
	cropCenter,
	cropRandom,
	augmentEnabled,
	augRotate,
	augRotateProb,
	augFlip,
	augFlipProb,
) {
	console.log("updateSummaryPanel: START", {
		augmentEnabled,
		augRotate,
		augFlip,
	});

	const preprocBadges = document.getElementById("summary-preproc-badges");
	const augBadges = document.getElementById("summary-aug-badges");
	const preprocRow = document.getElementById("summary-preproc");
	const augRow = document.getElementById("summary-aug");

	console.log("Elements found:", {
		preprocBadges,
		augBadges,
		preprocRow,
		augRow,
	});

	if (!preprocBadges || !augBadges || !preprocRow || !augRow) {
		console.error("Summary panel elements not found!");
		return;
	}

	let preprocHtml = "";
	if (normMinmax) preprocHtml += '<span class="badge badge-blue">MinMax</span>';
	if (normZscore)
		preprocHtml += '<span class="badge badge-blue">Z-Score</span>';
	if (cropCenter)
		preprocHtml += '<span class="badge badge-green">Center</span>';
	if (cropRandom)
		preprocHtml += '<span class="badge badge-green">Random</span>';

	console.log(
		"augmentEnabled:",
		augmentEnabled,
		"type:",
		typeof augmentEnabled,
	);
	console.log("augRotate:", augRotate, "augFlip:", augFlip);

	let augHtml = "";
	if (augmentEnabled) {
		console.log("Augmentation is ENABLED");
		if (augRotate) {
			console.log("Adding Rotate badge");
			augHtml += `<span class="badge badge-orange">Rotate (${augRotateProb})</span>`;
		}
		if (augFlip) {
			console.log("Adding Flip badge");
			augHtml += `<span class="badge badge-orange">Flip (${augFlipProb})</span>`;
		}
	} else {
		console.log("Augmentation is DISABLED, augmentEnabled =", augmentEnabled);
	}

	preprocBadges.innerHTML =
		preprocHtml || '<span class="badge-none">None</span>';
	augBadges.innerHTML = augHtml || '<span class="badge-none">Disabled</span>';

	preprocRow.style.display = preprocHtml ? "flex" : "flex";
	augRow.style.display = "flex";
}

function copyCommand() {
	const source = document.getElementById("command-display");
	const command = source ? source.dataset.raw : "";
	navigator.clipboard.writeText(command).then(() => {
		const btn =
			document.activeElement.closest(".btn") ||
			document.querySelector(".sub-page.active .btn");
		if (btn) {
			btn.textContent = "Copied!";
			setTimeout(() => {
				btn.innerHTML = "Copy <kbd>Ctrl+Shift+C</kbd>";
			}, 1500);
		}
	});
}

function openCommandModal() {
	document.getElementById("command-modal").classList.add("active");
}

function closeCommandModal() {
	document.getElementById("command-modal").classList.remove("active");
}

function toggleTheme() {
	const html = document.documentElement;
	const current = html.getAttribute("data-theme");
	const next = current === "light" ? "dark" : "light";

	if (next === "light") {
		html.setAttribute("data-theme", "light");
	} else {
		html.removeAttribute("data-theme");
	}

	localStorage.setItem("theme", next);
}

function initTheme() {
	const saved = localStorage.getItem("theme");
	if (saved === "light") {
		document.documentElement.setAttribute("data-theme", "light");
	}
}

function switchPage(pageName) {
	const pages = document.querySelectorAll(".page");
	pages.forEach((page) => {
		page.classList.remove("active");
	});

	const tabs = document.querySelectorAll(".nav-tab");
	tabs.forEach((tab) => {
		tab.classList.remove("active");
	});

	document.getElementById(`${pageName}-page`).classList.add("active");
	event.target.classList.add("active");

	// Load results if switching to results page
	if (pageName === "results") {
		loadResults();
	}

	// Sync command if switching to launch page
	if (pageName === "launch") {
		syncLaunchCommand();
	}
}

function switchSubTab(subTabName) {
	const activePage = document.querySelector(".page.active");
	const subPages = activePage.querySelectorAll(".sub-page");
	subPages.forEach((page) => {
		page.classList.remove("active");
	});

	const subTabs = activePage.querySelectorAll(".sub-tab");
	subTabs.forEach((tab) => {
		tab.classList.remove("active");
	});

	activePage.querySelector(`#sub-${subTabName}`).classList.add("active");
	event.target.classList.add("active");
}

let selectedResultIndex = 0;
const allTabs = [
	{ name: "Command", id: "command", page: "generate" },
	{ name: "Models", id: "models", page: "generate" },
	{ name: "Dataset Classes", id: "dataset-classes", page: "generate" },
	{ name: "Training Options", id: "training", page: "generate" },
	{ name: "Preprocessing", id: "preprocessing", page: "generate" },
	{ name: "Augmentation", id: "augmentation", page: "generate" },
	{ name: "Metrics", id: "metrics", page: "generate" },
	{ name: "Loss Functions", id: "loss", page: "generate" },
	{ name: "Device", id: "device", page: "generate" },
	{ name: "Active Transforms", id: "summary", page: "generate" },
	{ name: "Models (Docs)", id: "models", page: "docs" },
	{ name: "Dataset Classes (Docs)", id: "dataset-classes", page: "docs" },
	{ name: "Training Options (Docs)", id: "training", page: "docs" },
	{ name: "Preprocessing (Docs)", id: "preprocessing", page: "docs" },
	{ name: "Augmentation (Docs)", id: "augmentation", page: "docs" },
	{ name: "Metrics (Docs)", id: "metrics", page: "docs" },
	{ name: "Loss Functions (Docs)", id: "loss", page: "docs" },
	{ name: "Device (Docs)", id: "device", page: "docs" },
];

function openTabSearch() {
	const modal = document.getElementById("tab-search-modal");
	modal.classList.add("active");
	document.getElementById("tab-search-input").focus();
	selectedResultIndex = 0;
	updateSearchResults("");
}

function closeTabSearch() {
	document.getElementById("tab-search-modal").classList.remove("active");
	document.getElementById("tab-search-input").value = "";
}

function updateSearchResults(query) {
	const results = query
		? allTabs.filter((tab) =>
				tab.name.toLowerCase().includes(query.toLowerCase()),
			)
		: allTabs;

	const resultsDiv = document.getElementById("tab-search-results");
	resultsDiv.innerHTML = "";
	selectedResultIndex = 0;

	results.forEach((tab, index) => {
		const div = document.createElement("div");
		div.className = "tab-search-result";
		if (index === 0) div.classList.add("selected");
		div.textContent = tab.name;
		div.onclick = () => selectTab(tab);
		resultsDiv.appendChild(div);
	});
}

function selectTab(tab) {
	switchPage(tab.page);
	setTimeout(() => {
		const button = document.querySelector(
			`#${tab.page}-page .sub-tab[onclick="switchSubTab('${tab.id}')"]`,
		);
		if (button) button.click();
		closeTabSearch();
	}, 0);
}

function navigateResults(direction) {
	const results = document.querySelectorAll(".tab-search-result");
	if (results.length === 0) return;

	results[selectedResultIndex].classList.remove("selected");
	selectedResultIndex =
		(selectedResultIndex + direction + results.length) % results.length;
	results[selectedResultIndex].classList.add("selected");
	results[selectedResultIndex].scrollIntoView({ block: "nearest" });
}

document.addEventListener("keydown", (e) => {
	if (e.ctrlKey && e.shiftKey && e.key === "C") {
		e.preventDefault();
		copyCommand();
	}

	if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "K") {
		e.preventDefault();
		console.log("Ctrl+Shift+K pressed");
		const modal = document.getElementById("tab-search-modal");
		console.log("Modal found:", modal);
		if (modal && modal.classList.contains("active")) {
			closeTabSearch();
		} else if (modal) {
			openTabSearch();
		}
	}

	if (e.altKey && e.key.toLowerCase() === "c") {
		e.preventDefault();
		const modal = document.getElementById("command-modal");
		modal.classList.contains("active")
			? closeCommandModal()
			: openCommandModal();
	}
});

document.addEventListener("DOMContentLoaded", () => {
	initTheme();
	updateCommand();
	loadDatasets();
	loadModels();
	loadDatasetClasses();

	window.toggleTheme = toggleTheme;
	window.switchPage = switchPage;
	window.switchSubTab = switchSubTab;

	const searchInput = document.getElementById("tab-search-input");
	searchInput.addEventListener("input", (e) => {
		updateSearchResults(e.target.value);
	});

	searchInput.addEventListener("keydown", (e) => {
		if (e.key === "ArrowDown") {
			e.preventDefault();
			navigateResults(1);
		} else if (e.key === "ArrowUp") {
			e.preventDefault();
			navigateResults(-1);
		} else if (e.key === "Enter") {
			e.preventDefault();
			const selected = document.querySelector(".tab-search-result.selected");
			if (selected) {
				const index = Array.from(
					document.querySelectorAll(".tab-search-result"),
				).indexOf(selected);
				selectTab(allTabs[index]);
			}
		} else if (e.key === "Escape") {
			closeTabSearch();
		}
	});

	document.getElementById("tab-search-modal").addEventListener("click", (e) => {
		if (e.target.id === "tab-search-modal") {
			closeTabSearch();
		}
	});

	document.getElementById("command-modal").addEventListener("click", (e) => {
		if (e.target.id === "command-modal") {
			closeCommandModal();
		}
	});

	// Add Esc key handling for closing modals
	document.addEventListener("keydown", (e) => {
		if (e.key === "Escape") {
			const cmdModal = document.getElementById("command-modal");
			const searchModal = document.getElementById("tab-search-modal");
			if (cmdModal && cmdModal.classList.contains("active")) {
				closeCommandModal();
			} else if (searchModal && searchModal.classList.contains("active")) {
				closeTabSearch();
			}
		}
	});
});
