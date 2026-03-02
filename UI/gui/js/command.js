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
		} else if (
			[
				"epochs",
				"batch_size",
				"lr",
				"num_workers",
				"optimizer",
				"mixed_precision",
				"scheduler",
				"early_stopping",
				"patience",
			].includes(flag)
		) {
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
		} else if (
			[
				"epochs",
				"batch_size",
				"lr",
				"num_workers",
				"optimizer",
				"mixed_precision",
				"scheduler",
				"early_stopping",
				"patience",
			].includes(flag)
		) {
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
	const metricsCheckboxes = document.querySelectorAll(
		'input[name="metrics"]:checked',
	);
	const metrics = Array.from(metricsCheckboxes)
		.map((cb) => cb.value)
		.join(" ");

	const optimizer = document.getElementById("optimizer").value;
	const mixed_precision = document.getElementById("mixed_precision").value;
	const scheduler = document.getElementById("scheduler").value;
	const patience = document.getElementById("patience").value;
	const early_stopping = patience && patience.toUpperCase() !== "X";

	document.getElementById("aug_rotate_prob_val").textContent = aug_rotate_prob;
	document.getElementById("aug_flip_prob_val").textContent = aug_flip_prob;

	const resumeFrom = document.getElementById("resume_from")?.value;

	let command = `python3 -m src.run --dataset ${dataset} --model ${model} --metrics ${metrics} --loss ${loss}`;

	if (resumeFrom) {
		command += ` --resume ${resumeFrom}`;
	}

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
		// Use the average of rotate and flip probabilities
		let avgProb = aug_rotate_prob;
		if (aug_rotate && aug_flip) {
			avgProb = (parseFloat(aug_rotate_prob) + parseFloat(aug_flip_prob)) / 2;
		} else if (aug_flip) {
			avgProb = aug_flip_prob;
		}
		command += ` --aug_prob ${avgProb}`;
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

let _resumeResults = [];

async function loadResumeOptions() {
	try {
		const response = await fetch("/api/results");
		_resumeResults = await response.json();

		const resumeSelect = document.getElementById("resume_from");
		if (!resumeSelect) return;

		resumeSelect.innerHTML = '<option value="">-- New Training --</option>';

		_resumeResults
			.filter((r) => r.status === "complete")
			.forEach((result) => {
				const option = document.createElement("option");
				const runPath = `results/${result.dataset}/${result.model}/${result.timestamp}`;
				option.value = runPath;
				option.textContent = `${result.dataset} / ${result.model} (${formatDate(result.timestamp)})`;
				resumeSelect.appendChild(option);
			});

		const savedResume = localStorage.getItem("resumeFrom");
		if (savedResume) {
			resumeSelect.value = savedResume;
			handleResumeSelect();
			localStorage.removeItem("resumeFrom");
		}
	} catch (e) {
		console.error("Failed to load resume options:", e);
	}
}

function handleResumeSelect() {
	const resumeSelect = document.getElementById("resume_from");
	const selectedPath = resumeSelect?.value;

	if (!selectedPath) {
		updateCommand();
		return;
	}

	const result = _resumeResults.find(
		(r) => `results/${r.dataset}/${r.model}/${r.timestamp}` === selectedPath,
	);

	if (!result) {
		updateCommand();
		return;
	}

	const config = result.config || {};

	if (config.dataset) {
		document.getElementById("dataset").value = config.dataset;
	}
	if (config.model) {
		document.getElementById("model").value = config.model;
	}
	if (config.epochs) {
		document.getElementById("epochs").value = config.epochs;
	}
	if (config.batch_size) {
		document.getElementById("batch_size").value = config.batch_size;
	}
	if (config.image_size) {
		document.getElementById("img_size").value = config.image_size;
	}
	if (config.learning_rate) {
		document.getElementById("lr").value = config.learning_rate;
	}
	if (config.optimizer) {
		document.getElementById("optimizer").value = config.optimizer;
	}
	if (config.scheduler) {
		document.getElementById("scheduler").value = config.scheduler;
	}
	if (config.mixed_precision) {
		document.getElementById("mixed_precision").value = config.mixed_precision;
	}
	if (config.loss) {
		document.getElementById("loss").value = config.loss;
	}
	if (config.num_workers) {
		document.getElementById("num_workers").value = config.num_workers;
	}
	if (config.patience) {
		document.getElementById("patience").value = config.patience;
	}
	if (config.early_stopping_enabled) {
		document.getElementById("patience").value = config.patience || 10;
	}

	updateCommand();
}
