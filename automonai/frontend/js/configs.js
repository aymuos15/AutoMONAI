// Config management with per-card launch support

// Per-variant state: configName::variantId -> {eventSource, totalEpochs, currentEpoch, running}
const _cardState = new Map();
const _cardSelectedVariant = new Map();

const BASE_DEFAULTS = {
	model: "unet",
	dataset: "Dataset001_Cellpose",
	epochs: "1",
	batch_size: "4",
	img_size: "128",
	lr: "0.0001",
	optimizer: "adam",
	num_workers: "0",
	scheduler: "none",
	patience: "X",
	device: "cuda",
	mixed_precision: "no",
	loss: "dice",
	metrics: "dice",
	norm: "none",
	crop: "none",
	augment: "false",
	early_stopping: "false",
};

function _configDiff(command) {
	const normalized = command
		.replace(/\s*\\\s*/g, " ")
		.replace(/\s--cross_val(?:\s+|=)\d+/g, "")
		.replace(/\s--cv_fold(?:\s+|=)\d+/g, "")
		.replace(/\s--val_split(?:\s+|=)\S+/g, "")
		.replace(/\s--val_ratio(?:\s+|=)[\d.]+/g, "")
		.replace(/\s--split_seed(?:\s+|=)\d+/g, "")
		.replace(/\s--best_metric(?:\s+|=)\S+/g, "")
		.replace(/\s--ensemble_folds/g, "")
		.replace(/\s--ensemble_method(?:\s+|=)\S+/g, "")
		.replace(/\s+/g, " ")
		.trim();
	const diffs = [];
	for (const [key, defaultVal] of Object.entries(BASE_DEFAULTS)) {
		const re = new RegExp(`--${key}\\s+(.+?)(?=\\s+--|$)`);
		const match = normalized.match(re);
		if (match && match[1].trim() !== defaultVal) {
			diffs.push({ key, value: match[1].trim().replace(/\s+/g, ",") });
		}
	}
	return diffs;
}

function _normalizeCommand(command) {
	return command
		.replace(/\s*\\\s*/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function _variantStateKey(configName, variantId) {
	return `${configName}::${variantId || "no_val"}`;
}

function _effectiveRunId(configName, variantId) {
	return variantId && variantId !== "no_val"
		? `${configName}__${variantId}`
		: configName;
}

function _getLaunchVariants(config) {
	if (Array.isArray(config.launch_variants) && config.launch_variants.length > 0) {
		return config.launch_variants;
	}
	const command = _normalizeCommand(config.command || "");
	return [{ id: "no_val", label: "No Val", command }];
}

function _getSelectedVariantId(configName) {
	return _cardSelectedVariant.get(configName) || "no_val";
}

function _findActiveVariantId(configName, activeRuns) {
	if (activeRuns?.[configName]?.running) return "no_val";
	const prefix = `${configName}__`;
	for (const [runId, state] of Object.entries(activeRuns || {})) {
		if (state?.running && runId.startsWith(prefix)) {
			return runId.slice(prefix.length);
		}
	}
	return "no_val";
}

async function generateNewConfig() {
	const cmdDisplay = document.getElementById("command-display");
	const command = cmdDisplay?.dataset?.raw || cmdDisplay?.innerText;

	if (!command || command.trim() === "") {
		alert("No command to save. Please generate a command first.");
		return;
	}

	try {
		const listResponse = await fetch("/api/configs/list");
		const configs = await listResponse.json();
		const normalizedCommand = _normalizeCommand(command);

		const existingConfig = configs.find(
			(cfg) => _normalizeCommand(cfg.command || "") === normalizedCommand,
		);
		if (existingConfig) {
			alert(`This config already exists: ${existingConfig.name}`);
			switchPage("configs");
			loadConfigs();
			return;
		}

		const params = extractCommandParams(command);
		const diffs = _configDiff(command);
		const base =
			diffs.length === 0
				? "base"
				: diffs.map((d) => `${d.key}_${d.value}`).join("_");
		let configName = base;

		// Append numeric suffix if name already taken
		const existing = configs.map((c) => c.name);
		if (existing.includes(configName)) {
			let i = 2;
			while (existing.includes(`${configName}_${i}`)) i++;
			configName = `${configName}_${i}`;
		}
		const response = await fetch("/api/configs/save", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				name: configName,
				command: normalizedCommand,
				params: params,
				cv:
					typeof window.getCrossValidationSettings === "function"
						? window.getCrossValidationSettings()
						: { enabled: true, fold_count: 5 },
			}),
		});

		if (!response.ok) {
			const error = await response.json();
			throw new Error(error.detail);
		}

		document.getElementById("command-display").innerText = command;

		switchPage("configs");

		showNotification(`Config saved: ${configName}`);
	} catch (error) {
		alert("Error saving config: " + error.message);
	}
}

function extractCommandParams(command) {
	const params = {};
	const patterns = {
		model: /--model\s+(\S+)/,
		dataset: /--dataset\s+(\S+)/,
		epochs: /--epochs\s+(\d+)/,
	};
	for (const [key, pattern] of Object.entries(patterns)) {
		const match = command.match(pattern);
		if (match) params[key] = match[1];
	}
	return params;
}

async function loadConfigs() {
	try {
		const response = await fetch("/api/configs/list");
		if (!response.ok) throw new Error("Failed to load configs");
		const configs = await response.json();

		// Fetch active runs to re-attach streams
		const runRes = await fetch("/api/launch/list");
		const activeRuns = await runRes.json();

		renderConfigs(configs, activeRuns);
	} catch (error) {
		console.error("Error loading configs:", error);
		document.getElementById("configs-list").innerHTML =
			'<div class="configs-empty">Error loading configs</div>';
	}
}

function renderConfigs(configs, activeRuns = {}) {
	const container = document.getElementById("configs-list");

	if (configs.length === 0) {
		container.innerHTML = "";
		return;
	}

	container.innerHTML = configs
		.map((config) => {
			const diffs = _configDiff(config.command);
			const summary =
				diffs.length === 0
					? "base"
					: diffs.map((d) => `${d.key}: ${d.value}`).join(" · ");
			const name = config.name;
			const variants = _getLaunchVariants(config);
			const activeVariantId = _findActiveVariantId(name, activeRuns);
			const selectedVariantId = _cardSelectedVariant.get(name) || activeVariantId;
			_cardSelectedVariant.set(name, selectedVariantId);

			const selectedState = _cardState.get(
				_variantStateKey(name, selectedVariantId),
			);
			const isRunning = selectedState?.running === true;
			const isDone = selectedState?.done === true;
			const isInferred = selectedState?.inferred === true;
			const statusClass = isInferred
				? "card-status-done"
				: isDone || isRunning
					? "card-status-running"
					: "card-status-idle";

			// Determine if all fold variants are done/inferred (for ensemble gating)
			const foldVariants = variants.filter((v) => v.id.startsWith("fold_"));
			const foldState = config.fold_state || {};
			const allFoldsDone = foldVariants.length > 0 && foldVariants.every((v) => {
				const fs = foldState[v.id] || {};
				return fs.status === "done" || fs.status === "inferred";
			});

			const optionsHtml = variants
				.map((variant) => {
					const selectedAttr =
						variant.id === selectedVariantId ? " selected" : "";
					const disabledAttr =
						variant.id === "ensemble" && !allFoldsDone ? " disabled" : "";
					return `<option value="${escapeAttr(variant.id)}"${selectedAttr}${disabledAttr}>${escapeHtml(variant.label || variant.id)}</option>`;
				})
				.join("");

			return `<div class="launch-bar config-card ${statusClass}" id="card-${name}">
			<div class="launch-config-line">
				<span class="config-text">${escapeHtml(summary)}</span>
				<span class="card-actions"><button type="button" class="cmd-link card-full-btn" onclick="cardToggleConfig('${escapeAttr(name)}')">Config</button><button type="button" class="cmd-link delete-link" onclick="deleteConfig('${escapeAttr(name)}')">Delete</button></span>
			</div>
			<div class="launch-progress-container">
				<select class="card-variant-select" onchange="cardVariantChanged('${escapeAttr(name)}', this.value)">${optionsHtml}</select>
				<span class="card-spinner" ${isRunning ? "" : 'style="display:none"'}></span>
				<div class="progress-bar">
					<div class="progress-fill card-progress-fill" style="width:0%"></div>
					<div class="progress-text card-progress-text"></div>
				</div>
				<button type="button" class="cmd-link launch-link card-launch-btn" onclick="cardLaunch('${escapeAttr(name)}')">Launch</button>
				<button type="button" class="cmd-link launch-link card-infer-btn" onclick="cardInfer('${escapeAttr(name)}')" style="display:none">Infer</button>
				<button type="button" class="cmd-link launch-link card-stop-btn" onclick="cardStop('${escapeAttr(name)}')" style="display:none">Stop</button>
			</div>
			<div class="output full-width card-command-preview" style="display:none; margin-top:12px;">${escapeHtml(config.command)}</div>
		</div>`;
		})
		.join("");

	// Re-attach states and streams per variant
	for (const config of configs) {
		const name = config.name;
		const variants = _getLaunchVariants(config);
		const foldCkptEpochs = config.fold_checkpoint_epochs || {};
		const defaultVariantId = _findActiveVariantId(name, activeRuns);
		if (!_cardSelectedVariant.has(name)) {
			_cardSelectedVariant.set(name, defaultVariantId);
		}

		for (const variant of variants) {
			const key = _variantStateKey(name, variant.id);
			const runId = _effectiveRunId(name, variant.id);
			const isRunning = activeRuns[runId]?.running === true;
			const epochs = parseInt(variant.command.match(/--epochs\s+(\d+)/)?.[1]) || 1;
			const foldState = (config.fold_state || {})[variant.id] || {};
			const foldStatus = foldState.status || "idle";
			const isDone = foldStatus === "done" || foldStatus === "inferred";
			const isInferred = foldStatus === "inferred";

			if (isRunning) {
				const prev = _cardState.get(key);
				_cardState.set(key, {
					eventSource: prev?.eventSource || null,
					totalEpochs: epochs,
					currentEpoch: prev?.currentEpoch || 0,
					running: true,
					done: false,
					inferred: false,
					variantId: variant.id,
				});
				if (_getSelectedVariantId(name) === variant.id) {
					_cardStartLogStream(name, variant.id);
				}
			} else if (isDone) {
				_cardState.set(key, {
					eventSource: null,
					totalEpochs: epochs,
					currentEpoch: epochs,
					running: false,
					done: true,
					inferred: isInferred,
					variantId: variant.id,
				});
			} else if ((foldCkptEpochs[variant.id] || 0) > 0) {
				const prev = _cardState.get(key);
				_cardState.set(key, {
					eventSource: prev?.eventSource || null,
					totalEpochs: epochs,
					currentEpoch: foldCkptEpochs[variant.id],
					running: false,
					done: false,
					inferred: false,
					variantId: variant.id,
				});
			}
		}

		_cardApplySelectedState(name);
	}
}

function escapeHtml(str) {
	const div = document.createElement("div");
	div.textContent = str;
	return div.innerHTML;
}

function escapeAttr(str) {
	return str.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
}

function cardToggleConfig(name) {
	const card = document.getElementById(`card-${name}`);
	if (!card) return;
	const preview = card.querySelector(".card-command-preview");
	const btn = card.querySelector(".card-full-btn");

	const showing = preview.style.display !== "none";
	preview.style.display = showing ? "none" : "block";
	btn.textContent = showing ? "Config" : "Hide";
}

async function cardLaunch(name) {
	const variantId = _getSelectedVariantId(name);
	const stateKey = _variantStateKey(name, variantId);

	let variant;
	try {
		const res = await fetch(`/api/configs/get/${encodeURIComponent(name)}`);
		if (!res.ok) throw new Error("Config not found");
		const config = await res.json();
		const variants = _getLaunchVariants(config);
		variant =
			variants.find((item) => item.id === variantId) ||
			variants.find((item) => item.id === "no_val") ||
			variants[0];
	} catch (e) {
		alert("Error loading config: " + e.message);
		return;
	}

	try {
		const res = await fetch("/api/launch", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				command: _normalizeCommand(variant.command),
				run_id: name,
				variant_id: variant.id,
			}),
		});

		if (!res.ok) {
			const data = await res.json();
			alert(`Error: ${data.detail}`);
			return;
		}

		const epochs = parseInt(variant.command.match(/--epochs\s+(\d+)/)?.[1]) || 1;
		_cardState.set(stateKey, {
			eventSource: null,
			totalEpochs: epochs,
			currentEpoch: 0,
			running: true,
			done: false,
			inferred: false,
			variantId: variant.id,
		});
		_cardApplySelectedState(name);
		_cardStartLogStream(name, variant.id);
	} catch (e) {
		alert(`Error: ${e.message}`);
	}
}

async function cardInfer(name) {
	const variantId = _getSelectedVariantId(name);
	const stateKey = _variantStateKey(name, variantId);

	let variant;
	try {
		const res = await fetch(`/api/configs/get/${encodeURIComponent(name)}`);
		if (!res.ok) throw new Error("Config not found");
		const config = await res.json();
		const variants = _getLaunchVariants(config);
		variant =
			variants.find((item) => item.id === variantId) ||
			variants.find((item) => item.id === "no_val") ||
			variants[0];
	} catch (e) {
		alert("Error loading config: " + e.message);
		return;
	}

	try {
		const res = await fetch("/api/launch", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				command: `${_normalizeCommand(variant.command)} --mode infer`,
				run_id: name,
				variant_id: variant.id,
			}),
		});

		if (!res.ok) {
			const data = await res.json();
			alert(`Error: ${data.detail}`);
			return;
		}

		_cardState.set(stateKey, {
			eventSource: null,
			totalEpochs: 0,
			currentEpoch: 0,
			running: true,
			done: false,
			inferring: true,
			inferred: false,
			variantId: variant.id,
		});
		_cardApplySelectedState(name);
		_cardSetProgress(name, 0);
		_cardStartLogStream(name, variant.id);
	} catch (e) {
		alert(`Error: ${e.message}`);
	}
}

async function cardStop(name) {
	const variantId = _getSelectedVariantId(name);
	const state = _cardState.get(_variantStateKey(name, variantId));
	if (state) state.stopped = true;

	try {
		await fetch("/api/launch/stop", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ run_id: name, variant_id: variantId }),
		});
	} catch (e) {
		alert(`Error stopping: ${e.message}`);
	}
}

function cardVariantChanged(name, variantId) {
	_cardSelectedVariant.set(name, variantId || "no_val");
	_cardApplySelectedState(name);
}

function _cardApplySelectedState(name) {
	const variantId = _getSelectedVariantId(name);
	const state = _cardState.get(_variantStateKey(name, variantId));
	if (!state) {
		_cardSetRunningUI(name, false, false, false);
		_cardSetProgress(name, 0);
		return;
	}
	_cardSetRunningUI(name, state.running === true, state.done === true, state.inferred === true);
	const pct =
		state.done === true
			? 100
			: state.totalEpochs > 0
				? Math.round(
						Math.min((state.currentEpoch / state.totalEpochs) * 100, 100),
					)
				: 0;
	_cardSetProgress(name, pct);
}

function _cardSetRunningUI(name, running, done = false, inferred = false) {
	const card = document.getElementById(`card-${name}`);
	if (!card) return;

	const launchBtn = card.querySelector(".card-launch-btn");
	const inferBtn = card.querySelector(".card-infer-btn");
	const stopBtn = card.querySelector(".card-stop-btn");
	const spinner = card.querySelector(".card-spinner");

	if (spinner) spinner.style.display = running && !done ? "" : "none";

	card.classList.remove(
		"card-status-idle",
		"card-status-running",
		"card-status-done",
	);
	if (inferred) {
		card.classList.add("card-status-done");
	} else if (done || running) {
		card.classList.add("card-status-running");
	} else {
		card.classList.add("card-status-idle");
	}

	if (done || inferred) {
		launchBtn.style.display = "";
		inferBtn.style.display = inferred ? "none" : "";
		stopBtn.style.display = "none";
	} else if (running) {
		launchBtn.style.display = "none";
		inferBtn.style.display = "none";
		stopBtn.style.display = "";
	} else {
		launchBtn.style.display = "";
		inferBtn.style.display = "none";
		stopBtn.style.display = "none";
	}
}

function _cardStartLogStream(name, variantId) {
	const stateKey = _variantStateKey(name, variantId);
	const state = _cardState.get(stateKey);
	if (!state) return;

	if (state.eventSource) {
		state.eventSource.close();
	}

	const es = new EventSource(
		`/api/launch/logs?run_id=${encodeURIComponent(_effectiveRunId(name, variantId))}`,
	);
	state.eventSource = es;

	es.onmessage = (e) => {
		const line = e.data;
		const resumeMatch = line.match(/Resuming training from epoch (\d+)\/(\d+)/);
		if (resumeMatch) {
			state.currentEpoch = parseInt(resumeMatch[1]);
			state.totalEpochs = parseInt(resumeMatch[2]);
		}
		const epochMatch = line.match(/Epoch\s+(\d+)\/(\d+)/);
		if (epochMatch) {
			state.currentEpoch = parseInt(epochMatch[1]);
			state.totalEpochs = parseInt(epochMatch[2]);
		}
		const inferMatch = line.match(/Inference\s+(\d+)\/(\d+)/);
		if (inferMatch) {
			state.currentEpoch = parseInt(inferMatch[1]);
			state.totalEpochs = parseInt(inferMatch[2]);
		}
		if (_getSelectedVariantId(name) === variantId) {
			_cardApplySelectedState(name);
		}
	};

	es.addEventListener("done", () => {
		es.close();
		state.eventSource = null;
		state.running = false;

		if (state.stopped) {
			state.stopped = false;
			state.done = false;
			state.inferred = false;
		} else if (state.inferring) {
			state.done = true;
			state.inferring = false;
			state.inferred = true;
		} else {
			state.done = true;
			state.inferred = false;
		}
		if (_getSelectedVariantId(name) === variantId) {
			_cardApplySelectedState(name);
		}
	});

	es.onerror = () => {
		state.eventSource = null;
		state.running = false;
		if (_getSelectedVariantId(name) === variantId) {
			_cardApplySelectedState(name);
		}
	};
}

function _cardSetProgress(name, pct) {
	const card = document.getElementById(`card-${name}`);
	if (!card) return;

	const fill = card.querySelector(".card-progress-fill");
	const text = card.querySelector(".card-progress-text");

	fill.style.width = pct + "%";
	text.textContent = pct > 0 ? `${pct}%` : "";
}

async function loadConfig(configName) {
	try {
		const encodedName = encodeURIComponent(configName);
		const response = await fetch(`/api/configs/get/${encodedName}`);
		if (!response.ok) throw new Error("Failed to load config");

		const config = await response.json();
		parseAndLoadCommand(config.command);
		switchPage("generate");
		showNotification(`Config loaded: ${configName}`);
	} catch (error) {
		alert("Error loading config: " + error.message);
	}
}

function _showConfirmModal(message) {
	return new Promise((resolve) => {
		const modal = document.getElementById("confirm-modal");
		const msg = document.getElementById("confirm-modal-message");
		const okBtn = document.getElementById("confirm-modal-ok");
		const cancelBtn = document.getElementById("confirm-modal-cancel");

		msg.textContent = message;
		modal.classList.add("active");

		function cleanup(result) {
			modal.classList.remove("active");
			okBtn.removeEventListener("click", onOk);
			cancelBtn.removeEventListener("click", onCancel);
			modal.removeEventListener("click", onBackdrop);
			document.removeEventListener("keydown", onKey);
			resolve(result);
		}

		function onOk() { cleanup(true); }
		function onCancel() { cleanup(false); }
		function onBackdrop(e) { if (e.target === modal) cleanup(false); }
		function onKey(e) { if (e.key === "Escape") cleanup(false); }

		okBtn.addEventListener("click", onOk);
		cancelBtn.addEventListener("click", onCancel);
		modal.addEventListener("click", onBackdrop);
		document.addEventListener("keydown", onKey);
	});
}

async function deleteConfig(configName) {
	const variantId = _getSelectedVariantId(configName);
	const state = _cardState.get(_variantStateKey(configName, variantId));
	if (state?.running) {
		if (!(await _showConfirmModal(`Config "${configName}" is running. Stop and delete?`))) return;
		await cardStop(configName);
		await new Promise((r) => setTimeout(r, 500));
	} else {
		if (!(await _showConfirmModal(`Delete config "${configName}"?`))) return;
	}

	// Clean up SSE stream
	if (state?.eventSource) {
		state.eventSource.close();
	}
	for (const [key, item] of _cardState.entries()) {
		if (key.startsWith(`${configName}::`)) {
			if (item?.eventSource) item.eventSource.close();
			_cardState.delete(key);
		}
	}
	_cardSelectedVariant.delete(configName);

	try {
		const encodedName = encodeURIComponent(configName);
		const response = await fetch(`/api/configs/delete/${encodedName}`, {
			method: "DELETE",
		});

		if (!response.ok) {
			const error = await response.json();
			throw new Error(error.detail);
		}

		await loadConfigs();
		showNotification(`Config deleted: ${configName}`);
	} catch (error) {
		console.error("Delete error:", error);
		alert("Error deleting config: " + error.message);
	}
}

async function copyConfigCommand(configName) {
	try {
		const encodedName = encodeURIComponent(configName);
		const response = await fetch(`/api/configs/get/${encodedName}`);
		if (!response.ok) throw new Error("Failed to load config");

		const config = await response.json();
		await navigator.clipboard.writeText(config.command);
		showNotification("Command copied to clipboard");
	} catch (error) {
		alert("Error copying config: " + error.message);
	}
}

function parseAndLoadCommand(command) {
	const normalized = _normalizeCommand(command);
	const patterns = {
		dataset: /--dataset\s+(\S+)/,
		model: /--model\s+(\S+)/,
		epochs: /--epochs\s+(\d+)/,
		batch_size: /--batch_size\s+(\d+)/,
		img_size: /--img_size\s+(\d+)/,
		lr: /--lr\s+([\d.]+)/,
		optimizer: /--optimizer\s+(\S+)/,
		num_workers: /--num_workers\s+(\d+)/,
		scheduler: /--scheduler\s+(\S+)/,
		patience: /--patience\s+(\S+)/,
		device: /--device\s+(\S+)/,
		mixed_precision: /--mixed_precision\s+(\S+)/,
		loss: /--loss\s+(\S+)/,
	};

	for (const [field, pattern] of Object.entries(patterns)) {
		const input = document.getElementById(field);
		if (input) {
			const match = normalized.match(pattern);
			if (match) input.value = match[1];
		}
	}

	const checkboxPatterns = {
		norm_minmax: /--norm_minmax/,
		norm_zscore: /--norm_zscore/,
		crop_center: /--crop_center/,
		crop_random: /--crop_random/,
		aug_rotate: /--aug_rotate/,
		aug_flip: /--aug_flip/,
		"metrics-dice": /--metrics.*dice/,
		"metrics-iou": /--metrics.*iou/,
	};

	for (const [field, pattern] of Object.entries(checkboxPatterns)) {
		const checkbox = document.getElementById(field);
		if (checkbox) {
			checkbox.checked = pattern.test(normalized);
		}
	}

	const crossValEnabledInput = document.getElementById("cross_val_enabled");
	const crossValCountInput = document.getElementById("cross_val_count");
	const crossValMatch = normalized.match(/--cross_val\s+(\d+)/);
	if (crossValEnabledInput) {
		crossValEnabledInput.value = crossValMatch ? "true" : "false";
	}
	if (crossValCountInput) {
		crossValCountInput.value = crossValMatch ? crossValMatch[1] : "5";
	}

	// Val split fields
	const valSplitPatterns = {
		val_split: /--val_split\s+(\S+)/,
		val_ratio: /--val_ratio\s+([\d.]+)/,
		split_seed: /--split_seed\s+(\d+)/,
		best_metric: /--best_metric\s+(\S+)/,
	};
	for (const [field, pattern] of Object.entries(valSplitPatterns)) {
		const input = document.getElementById(field);
		if (input) {
			const match = normalized.match(pattern);
			if (match) input.value = match[1];
		}
	}

	updateCommand();
}

function showNotification(message) {
	// TODO: implement UI notification toast
}

async function syncWandb() {
	const btn = document.querySelector("#configs-page .sub-tab");
	btn.disabled = true;
	btn.textContent = "Syncing...";

	try {
		const response = await fetch("/api/configs/sync-wandb", { method: "POST" });
		if (!response.ok) {
			const err = await response.json();
			throw new Error(err.detail);
		}

		const result = await response.json();
		const parts = [];
		if (result.deleted.length > 0)
			parts.push(`Deleted ${result.deleted.length} orphaned run(s)`);
		if (result.updated.length > 0)
			parts.push(`Updated ${result.updated.length}`);
		if (parts.length === 0) parts.push("W&B already in sync");

		showNotification(parts.join(", "));
	} catch (error) {
		alert("W&B sync error: " + error.message);
	} finally {
		btn.disabled = false;
		btn.textContent = "Sync W&B";
	}
}

window.syncWandb = syncWandb;

// Load configs when configs page is viewed
document.addEventListener("DOMContentLoaded", () => {
	loadConfigs();
});

// Explicit exports for config management
window.deleteConfig = deleteConfig;
window.cardLaunch = cardLaunch;
window.cardInfer = cardInfer;
window.cardStop = cardStop;
window.cardVariantChanged = cardVariantChanged;
window.cardToggleConfig = cardToggleConfig;
window.loadConfig = loadConfig;
