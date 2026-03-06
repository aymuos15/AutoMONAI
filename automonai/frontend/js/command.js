function onDatasetClassChange(prefix) {
	const cls = document.getElementById(
		prefix === "train" ? "train_dataset" : "inference_dataset",
	).value;

	const needsCacheRate = ["CacheDataset", "SmartCacheDataset"].includes(cls);
	const needsReplaceRate = cls === "SmartCacheDataset";
	const needsCacheDir = [
		"PersistentDataset",
		"LMDBDataset",
		"CacheNTransDataset",
	].includes(cls);

	document.getElementById(`${prefix}-opt-cache-rate`).style.display =
		needsCacheRate ? "" : "none";
	document.getElementById(`${prefix}-opt-replace-rate`).style.display =
		needsReplaceRate ? "" : "none";
	document.getElementById(`${prefix}-opt-cache-dir`).style.display =
		needsCacheDir ? "" : "none";

	document.getElementById(`${prefix}-dataset-options`).style.display =
		needsCacheRate || needsReplaceRate || needsCacheDir ? "" : "none";
}

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
	const crossValEnabled =
		document.getElementById("cross_val_enabled")?.value !== "false";
	const crossValCountInput = document.getElementById("cross_val_count");
	const crossValCount = Math.max(
		2,
		parseInt(crossValCountInput?.value || "5", 10) || 5,
	);
	if (crossValCountInput) {
		crossValCountInput.disabled = !crossValEnabled;
		crossValCountInput.value = String(crossValCount);
	}

	const inferer = document.getElementById("inferer").value;

	// Auto-set val_split=kfold when cross_val is enabled (before reading)
	const valSplitSelect = document.getElementById("val_split");
	if (crossValEnabled && valSplitSelect && valSplitSelect.value === "none") {
		valSplitSelect.value = "kfold";
	}

	const effectiveValSplit = document.getElementById("val_split").value;
	const valRatio = document.getElementById("val_ratio").value;
	const splitSeed = document.getElementById("split_seed").value;
	const bestMetric = document.getElementById("best_metric").value;

	// Show/hide val_ratio group based on split mode
	const valRatioGroup = document.getElementById("val-ratio-group");
	if (valRatioGroup) {
		valRatioGroup.style.display = effectiveValSplit === "holdout" ? "" : "none";
	}
	// Show/hide best_metric when val_split is not none
	const bestMetricGroup = document.getElementById("best-metric-group");
	if (bestMetricGroup) {
		bestMetricGroup.style.display = effectiveValSplit !== "none" ? "" : "none";
	}

	document.getElementById("aug_rotate_prob_val").textContent = aug_rotate_prob;
	document.getElementById("aug_flip_prob_val").textContent = aug_flip_prob;

	// Collect extra transforms
	const extraTransformCheckboxes = document.querySelectorAll(
		'input[name="extra_transforms"]:checked',
	);
	const extraTransforms = Array.from(extraTransformCheckboxes)
		.map((cb) => cb.value)
		.join(" ");

	let command = `python3 -m automonai.core.run --dataset ${dataset} --model ${model} --train_dataset_class ${trainDataset} --inference_dataset_class ${inferenceDataset} --epochs ${epochs} --batch_size ${batch_size} --lr ${lr} --img_size ${img_size} --num_workers ${num_workers} --output_dir ${output_dir} --device ${device} --metrics ${metrics} --loss ${loss}`;

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

	if (extraTransforms) {
		command += ` --extra_transforms ${extraTransforms}`;
	}

	// Train dataset class options
	if (["CacheDataset", "SmartCacheDataset"].includes(trainDataset)) {
		command += ` --cache_rate ${document.getElementById("train_cache_rate").value}`;
	}
	if (trainDataset === "SmartCacheDataset") {
		command += ` --smart_replace_rate ${document.getElementById("train_replace_rate").value}`;
	}
	const trainCacheDir = document.getElementById("train_cache_dir").value;
	if (
		["PersistentDataset", "LMDBDataset", "CacheNTransDataset"].includes(
			trainDataset,
		) &&
		trainCacheDir
	) {
		command += ` --cache_dir ${trainCacheDir}`;
	}

	// Inference dataset class options
	if (["CacheDataset", "SmartCacheDataset"].includes(inferenceDataset)) {
		command += ` --inference_cache_rate ${document.getElementById("inference_cache_rate").value}`;
	}
	const infCacheDir = document.getElementById("inference_cache_dir").value;
	if (
		["PersistentDataset", "LMDBDataset", "CacheNTransDataset"].includes(
			inferenceDataset,
		) &&
		infCacheDir
	) {
		command += ` --inference_cache_dir ${infCacheDir}`;
	}

	command += ` --optimizer ${optimizer}`;
	command += ` --mixed_precision ${mixed_precision}`;
	command += ` --scheduler ${scheduler}`;
	command += ` --early_stopping ${early_stopping ? "true" : "false"}`;
	if (early_stopping) {
		command += ` --patience ${patience}`;
	}

	command += ` --inferer ${inferer}`;

	const deepSupervision = document.getElementById("deep_supervision").checked;
	command += ` --deep_supervision ${deepSupervision ? "true" : "false"}`;

	if (effectiveValSplit !== "none") {
		command += ` --val_split ${effectiveValSplit}`;
		if (effectiveValSplit === "holdout") {
			command += ` --val_ratio ${valRatio}`;
		}
		command += ` --split_seed ${splitSeed}`;
		command += ` --best_metric ${bestMetric}`;
	}

	const formattedHtml = formatCommand(command);
	const rawCmd = formatRawCommand(command);

	// Command sub-tab: structured parameter grid
	const cmdDisplay = document.getElementById("command-display");
	if (cmdDisplay) {
		cmdDisplay.dataset.raw = rawCmd;
	}
	const paramsGrid = document.getElementById("cmd-params-grid");
	if (paramsGrid) {
		const tag = (v) => `<span class="cmd-tag">${v}</span>`;
		const dim = (v) => `<span class="cmd-val-dim">${v}</span>`;

		const normList = [];
		if (norm_minmax) normList.push("MinMax");
		if (norm_zscore) normList.push("Z-Score");

		const cropList = [];
		if (crop_center) cropList.push("Center");
		if (crop_random) cropList.push("Random");

		const augList = [];
		if (aug_rotate) augList.push(`Rotate ${dim(aug_rotate_prob)}`);
		if (aug_flip) augList.push(`Flip ${dim(aug_flip_prob)}`);

		const extraTxEls = document.querySelectorAll('input[name="extra_transforms"]:checked');
		const extraTxNames = Array.from(extraTxEls).map((cb) => cb.nextElementSibling?.textContent || cb.value);

		const cell = (label, value, flex) =>
			`<div class="cmd-param-group" style="flex:${flex}"><div class="cmd-param-group-label">${label}</div><div class="cmd-param-value">${value}</div></div>`;

		const rows = [
			// Row 1: dataset full width
			[
				cell("Dataset", `${tag(dataset)} ${dim("·")} ${dim("train")} ${trainDataset} ${dim("·")} ${dim("infer")} ${inferenceDataset}`, 1),
			],
			// Row 2: model narrow + training params
			[
				cell("Model", tag(model), 1),
				cell("Epochs / Batch", `${epochs} ${dim("ep")} / ${batch_size} ${dim("bs")}`, 1),
				cell("LR / Optimizer", `${lr} ${dim("lr")} / ${optimizer}`, 1),
				cell("Scheduler", scheduler === "none" ? dim("none") : tag(scheduler), 1),
			],
			// Row 3: metrics wide | loss narrow
			[
				cell("Metrics", metrics ? metrics.split(" ").map(tag).join(" ") : dim("none"), 1),
				cell("Loss", tag(loss), 1),
			],
			// Row 4: three columns, middle wider
			[
				cell("Image Size", `${img_size}${dim("px")}`, 1),
				cell("Augmentation", augList.length ? augList.join(dim(" · ")) + (extraTxNames.length ? dim(" + ") + extraTxNames.map(tag).join(" ") : "") : dim("off"), 1),
				cell("Device", tag(device), 1),
				cell("Precision", tag(mixed_precision), 1),
			],
			// Row 5: two even
			[
				cell("Normalize", normList.length ? normList.map(tag).join(" ") : dim("none"), 1),
				cell("Crop", cropList.length ? cropList.map(tag).join(" ") : dim("none"), 1),
			],
		];

		// Conditional row
		const extras = [];
		if (early_stopping) extras.push(cell("Early Stop", `patience ${tag(patience)}`, 1));
		if (crossValEnabled) extras.push(cell("Cross-Val", tag(crossValCount + "-fold"), 1));
		if (effectiveValSplit !== "none") {
			let valDesc = tag(effectiveValSplit);
			if (effectiveValSplit === "holdout") valDesc += ` ${dim(valRatio)}`;
			valDesc += ` ${dim("seed=" + splitSeed)}`;
			extras.push(cell("Val Split", valDesc, 1));
			extras.push(cell("Best Metric", tag(bestMetric), 1));
		}
		extras.push(cell("Inferer", tag(inferer), 1));
		if (extras.length) rows.push(extras);

		paramsGrid.innerHTML = rows
			.map((cells) => `<div class="cmd-param-row">${cells.join("")}</div>`)
			.join("");
	}
	const rawBlock = document.getElementById("cmd-raw-block");
	if (rawBlock) {
		rawBlock.textContent = rawCmd;
	}

	// Command modal display
	const cmdModalDisplay = document.getElementById("command-modal-display");
	if (cmdModalDisplay) {
		cmdModalDisplay.innerHTML = formattedHtml;
		cmdModalDisplay.dataset.raw = rawCmd;
	}
}

function getCrossValidationSettings() {
	const enabled = document.getElementById("cross_val_enabled")?.value !== "false";
	const foldCount = Math.max(
		2,
		parseInt(document.getElementById("cross_val_count")?.value || "5", 10) || 5,
	);
	return { enabled, fold_count: foldCount };
}

window.getCrossValidationSettings = getCrossValidationSettings;
