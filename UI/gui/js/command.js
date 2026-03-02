function formatRawCommand(cmd) {
	// Parse command into flags and arguments
	const parts = cmd.split(" --");
	const baseCmd = parts[0];

	// Build formatted raw command with one flag per line
	let raw = `${baseCmd} \\\n`;

	for (let i = 1; i < parts.length; i++) {
		const [flag, ...argParts] = parts[i].split(" ");
		const arg = argParts.join(" ");
		const isLast = i === parts.length - 1;

		raw += `  --${flag}`;
		if (arg) {
			raw += ` ${arg}`;
		}
		raw += isLast ? "" : ` \\\n`;
	}

	return raw;
}

function formatCommand(cmd) {
	// Parse command into flags and arguments
	const parts = cmd.split(" --");
	const baseCmd = parts[0];

	// Build formatted HTML with one flag per line
	let html = `<span class="cmd">${baseCmd}</span> \\<br>`;

	for (let i = 1; i < parts.length; i++) {
		const [flag, ...argParts] = parts[i].split(" ");
		const arg = argParts.join(" ");
		const isLast = i === parts.length - 1;

		html += `  <span class="flag">--${flag}</span>`;
		if (arg) {
			html += ` <span class="arg">${arg}</span>`;
		}
		if (!isLast) {
			html += ` \\<br>`;
		}
	}

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

	let command = `python3 -m src.run --dataset ${dataset} --model ${model} --train_dataset_class ${trainDataset} --inference_dataset_class ${inferenceDataset} --epochs ${epochs} --batch_size ${batch_size} --lr ${lr} --img_size ${img_size} --num_workers ${num_workers} --output_dir ${output_dir} --device ${device} --metrics ${metrics} --loss ${loss}`;

	if (resumeFrom) {
		command += ` --resume ${resumeFrom}`;
	}

	const normOpts = [];
	if (norm_minmax) normOpts.push("minmax");
	if (norm_zscore) normOpts.push("zscore");
	command += ` --norm ${normOpts.length > 0 ? normOpts.join(" ") : "none"}`;

	const cropOpts = [];
	if (crop_center) cropOpts.push("center");
	if (crop_random) cropOpts.push("random");
	command += ` --crop ${cropOpts.length > 0 ? cropOpts.join(" ") : "none"}`;

	command += ` --augment ${augmentEnabled ? "true" : "false"}`;
	if (augmentEnabled) {
		// Use the average of rotate and flip probabilities
		let avgProb = aug_rotate_prob;
		if (aug_rotate && aug_flip) {
			avgProb = (parseFloat(aug_rotate_prob) + parseFloat(aug_flip_prob)) / 2;
		} else if (aug_flip) {
			avgProb = aug_flip_prob;
		}
		command += ` --aug_prob ${avgProb}`;
	}

	command += ` --optimizer ${optimizer}`;
	command += ` --mixed_precision ${mixed_precision}`;
	command += ` --scheduler ${scheduler}`;
	command += ` --early_stopping ${early_stopping ? "true" : "false"}`;
	if (early_stopping) {
		command += ` --patience ${patience}`;
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
