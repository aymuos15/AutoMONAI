// Config management with per-card launch support

// Per-card state: configName -> {eventSource, totalEpochs, currentEpoch, running}
const _cardState = new Map();

const BASE_DEFAULTS = {
	model: "unet", dataset: "Dataset001_Cellpose",
	epochs: "1", batch_size: "4", img_size: "128",
	lr: "0.0001", optimizer: "adam", num_workers: "0",
	scheduler: "none", patience: "X", device: "cuda",
	mixed_precision: "no", loss: "dice", metrics: "dice",
	norm: "none", crop: "none", augment: "false",
	early_stopping: "false",
};

function _configDiff(command) {
	const diffs = [];
	for (const [key, defaultVal] of Object.entries(BASE_DEFAULTS)) {
		const re = new RegExp(`--${key}\\s+(\\S+)`);
		const match = command.match(re);
		if (match && match[1] !== defaultVal) {
			diffs.push({ key, value: match[1] });
		}
	}
	return diffs;
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

		const existingConfig = configs.find(cfg => cfg.command === command);
		if (existingConfig) {
			alert(`This config already exists: ${existingConfig.name}`);
			switchPage("configs");
			loadConfigs();
			return;
		}
	} catch (error) {
		console.error("Error checking for duplicate configs:", error);
	}

	const params = extractCommandParams(command);
	const diffs = _configDiff(command);
	const base = diffs.length === 0
		? "base"
		: diffs.map(d => `${d.key}_${d.value}`).join("_");
	let configName = base;

	// Append numeric suffix if name already taken
	try {
		const listRes = await fetch("/api/configs/list");
		const existing = (await listRes.json()).map(c => c.name);
		if (existing.includes(configName)) {
			let i = 2;
			while (existing.includes(`${configName}_${i}`)) i++;
			configName = `${configName}_${i}`;
		}
	} catch (_) {}

	try {
		const response = await fetch("/api/configs/save", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				name: configName,
				command: command,
				params: params,
			}),
		});

		if (!response.ok) {
			const error = await response.json();
			throw new Error(error.detail || "Failed to save config");
		}

		const commandDisplay = document.getElementById("command-display");
		if (commandDisplay) {
			commandDisplay.innerText = command;
		}

		switchPage("configs");

		if (window.syncLaunchCommand) {
			syncLaunchCommand();
		}

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
		let activeRuns = {};
		try {
			const runRes = await fetch("/api/launch/list");
			activeRuns = await runRes.json();
		} catch (_) {}

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

	container.innerHTML = configs.map((config) => {
		const diffs = _configDiff(config.command);
		const summary = diffs.length === 0
			? "base"
			: diffs.map(d => `${d.key}: ${d.value}`).join(" · ");
		const name = config.name;
		const status = config.status || "idle";
		const isRunning = status === "running" || activeRuns[name]?.running === true;
		const isDone = status === "done";
		const epochs = parseInt(config.command.match(/--epochs\s+(\d+)/)?.[1]) || 1;
		const ckptEpoch = config.checkpoint_epoch || 0;
		const pct = isDone ? 100 : (epochs > 0 ? Math.round((ckptEpoch / epochs) * 100) : 0);

		const hideLaunch = isRunning || isDone;
		const hideStop = !isRunning;

		return `<div class="launch-bar config-card" id="card-${name}">
			<div class="launch-config-line">
				<span class="config-text">${escapeHtml(summary)}</span>
				<span class="card-actions"><button type="button" class="cmd-link card-full-btn" onclick="cardToggleFull('${name}')">Config</button><button type="button" class="cmd-link delete-link" onclick="deleteConfig('${name}')">Delete</button></span>
			</div>
			<div class="launch-progress-container">
				<span class="card-spinner" ${isRunning ? '' : 'style="display:none"'}></span>
				<div class="progress-bar">
					<div class="progress-fill card-progress-fill" style="width:${pct}%"></div>
					<div class="progress-text card-progress-text">${pct > 0 ? escapeHtml(name) + " \u00b7 " + pct + "%" : escapeHtml(name)}</div>
				</div>
				<button type="button" class="cmd-link launch-link card-launch-btn" onclick="cardLaunch('${name}')" ${hideLaunch ? 'style="display:none"' : ""}>Launch</button>
				<button type="button" class="cmd-link launch-link card-stop-btn" onclick="cardStop('${name}')" ${hideStop ? 'style="display:none"' : ""}>Stop</button>
			</div>
			<div class="output full-width card-command-preview" style="display:none; margin-top:12px;">${escapeHtml(config.command)}</div>
			<div class="card-terminal" style="display:none; max-height:200px; overflow-y:auto; font-family:var(--font-mono); font-size:0.7rem; background:var(--bg); padding:8px; border:1px solid var(--border); margin-top:8px;"></div>
		</div>`;
	}).join("");

	// Re-attach SSE streams for running configs, set done/progress state
	for (const config of configs) {
		const name = config.name;
		const status = config.status || "idle";
		const isRunning = status === "running" || activeRuns[name]?.running;
		const epochs = parseInt(config.command.match(/--epochs\s+(\d+)/)?.[1]) || 1;
		const ckptEpoch = config.checkpoint_epoch || 0;

		if (isRunning) {
			_cardState.set(name, { eventSource: null, totalEpochs: epochs, currentEpoch: ckptEpoch, running: true, done: false });
			_cardSetRunningUI(name, true);
			_cardUpdateProgress(name);
			_cardStartLogStream(name);
		} else if (status === "done") {
			_cardState.set(name, { eventSource: null, totalEpochs: epochs, currentEpoch: epochs, running: false, done: true });
		} else if (ckptEpoch > 0) {
			// Idle but has checkpoint progress — show it
			_cardState.set(name, { eventSource: null, totalEpochs: epochs, currentEpoch: ckptEpoch, running: false, done: false });
			_cardUpdateProgress(name);
		}
	}
}

function escapeHtml(str) {
	const div = document.createElement("div");
	div.textContent = str;
	return div.innerHTML;
}

function cardToggleFull(name) {
	const card = document.getElementById(`card-${name}`);
	if (!card) return;
	const preview = card.querySelector(".card-command-preview");
	const terminal = card.querySelector(".card-terminal");
	const btn = card.querySelector(".card-full-btn");

	const showing = preview.style.display !== "none";
	preview.style.display = showing ? "none" : "block";
	terminal.style.display = showing ? "none" : "block";
	btn.textContent = showing ? "Config" : "Hide";
}

async function cardLaunch(name) {
	const existingState = _cardState.get(name);
	if (existingState?.done) return;

	// Fetch the config's command
	let command;
	try {
		const res = await fetch(`/api/configs/get/${encodeURIComponent(name)}`);
		if (!res.ok) throw new Error("Config not found");
		const config = await res.json();
		command = config.command.replace(/\s*\\\s*/g, " ").replace(/\s+/g, " ").trim();
	} catch (e) {
		alert("Error loading config: " + e.message);
		return;
	}

	try {
		const res = await fetch("/api/launch", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ command, run_id: name }),
		});

		if (!res.ok) {
			let errorMsg = "Failed to launch";
			try {
				const data = await res.json();
				errorMsg = data.detail || errorMsg;
			} catch (_) {}
			alert(`Error: ${errorMsg}`);
			return;
		}

		const epochs = parseInt(command.match(/--epochs\s+(\d+)/)?.[1]) || 1;
		const prev = _cardState.get(name);
		_cardState.set(name, { eventSource: null, totalEpochs: epochs, currentEpoch: prev?.currentEpoch || 0, running: true });
		_cardSetRunningUI(name, true);
		_cardClearTerminal(name);
		_cardUpdateProgress(name);
		_cardStartLogStream(name);
	} catch (e) {
		alert(`Error: ${e.message}`);
	}
}

async function cardStop(name) {
	const state = _cardState.get(name);
	if (state) state.stopped = true;

	try {
		await fetch("/api/launch/stop", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ run_id: name }),
		});
	} catch (_) {}
}

function _cardSetRunningUI(name, running, done = false) {
	const card = document.getElementById(`card-${name}`);
	if (!card) return;

	const launchBtn = card.querySelector(".card-launch-btn");
	const stopBtn = card.querySelector(".card-stop-btn");
	const spinner = card.querySelector(".card-spinner");

	if (spinner) spinner.style.display = running && !done ? "" : "none";

	if (done) {
		launchBtn.style.display = "none";
		stopBtn.style.display = "none";
	} else if (running) {
		launchBtn.style.display = "none";
		stopBtn.style.display = "";
	} else {
		launchBtn.style.display = "";
		stopBtn.style.display = "none";
	}
}

function _cardClearTerminal(name) {
	const card = document.getElementById(`card-${name}`);
	if (!card) return;
	card.querySelector(".card-terminal").innerHTML = "";
}

function _cardStartLogStream(name) {
	const state = _cardState.get(name);
	if (!state) return;

	// Close existing stream
	if (state.eventSource) {
		state.eventSource.close();
	}

	const es = new EventSource(`/api/launch/logs?run_id=${encodeURIComponent(name)}`);
	state.eventSource = es;

	es.onmessage = (e) => _cardAppendLog(name, e.data);

	es.addEventListener("done", () => {
		es.close();
		state.eventSource = null;
		state.running = false;

		if (state.stopped) {
			// User stopped — allow re-launch
			state.stopped = false;
			_cardSetRunningUI(name, false);
		} else {
			// Natural completion — block re-launch
			state.done = true;
			_cardSetRunningUI(name, false, true);
			_cardSetProgress(name, 100);
		}
	});

	es.onerror = () => {
		_cardSetRunningUI(name, false);
		state.eventSource = null;
		state.running = false;
	};
}

function _cardAppendLog(name, line) {
	const card = document.getElementById(`card-${name}`);
	if (!card) return;

	const terminal = card.querySelector(".card-terminal");
	const div = document.createElement("div");
	div.className = "log-line";
	div.textContent = line;
	terminal.appendChild(div);
	terminal.scrollTop = terminal.scrollHeight;

	const state = _cardState.get(name);
	if (!state) return;

	// Pick up resume starting point
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

	_cardUpdateProgress(name);
}

function _cardUpdateProgress(name) {
	const state = _cardState.get(name);
	if (!state) return;

	const progress = state.totalEpochs > 0
		? Math.min((state.currentEpoch / state.totalEpochs) * 100, 100)
		: 0;

	_cardSetProgress(name, Math.round(progress));
}

function _cardSetProgress(name, pct) {
	const card = document.getElementById(`card-${name}`);
	if (!card) return;

	const fill = card.querySelector(".card-progress-fill");
	const text = card.querySelector(".card-progress-text");

	fill.style.width = pct + "%";
	const runName = name || "";
	text.textContent = pct > 0 ? `${runName} · ${pct}%` : runName;
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

async function deleteConfig(configName) {
	// If running, stop first
	const state = _cardState.get(configName);
	if (state?.running) {
		if (!confirm(`Config "${configName}" is running. Stop and delete?`)) return;
		await cardStop(configName);
		// Brief wait for stop to take effect
		await new Promise(r => setTimeout(r, 500));
	} else {
		if (!confirm(`Delete config "${configName}"?`)) return;
	}

	// Clean up SSE stream
	if (state?.eventSource) {
		state.eventSource.close();
	}
	_cardState.delete(configName);

	try {
		const encodedName = encodeURIComponent(configName);
		const response = await fetch(`/api/configs/delete/${encodedName}`, {
			method: "DELETE",
		});

		if (!response.ok) {
			const error = await response.json();
			throw new Error(error.detail || "Failed to delete config");
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
			const match = command.match(pattern);
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
			checkbox.checked = pattern.test(command);
		}
	}

	updateCommand();
}

function showNotification(message) {
	console.log(message);
}

async function syncWandb() {
	const btn = document.querySelector('#configs-page .sub-tab');
	if (btn) {
		btn.disabled = true;
		btn.textContent = 'Syncing...';
	}

	try {
		const response = await fetch('/api/configs/sync-wandb', { method: 'POST' });
		if (!response.ok) {
			const err = await response.json();
			throw new Error(err.detail || 'Sync failed');
		}

		const result = await response.json();
		const parts = [];
		if (result.deleted.length > 0) parts.push(`Deleted ${result.deleted.length} orphaned run(s)`);
		if (result.updated.length > 0) parts.push(`Updated ${result.updated.length}`);
		if (parts.length === 0) parts.push('W&B already in sync');

		showNotification(parts.join(', '));
	} catch (error) {
		alert('W&B sync error: ' + error.message);
	} finally {
		if (btn) {
			btn.disabled = false;
			btn.textContent = 'Sync W&B';
		}
	}
}

window.syncWandb = syncWandb;

// Load configs when configs page is viewed
document.addEventListener("DOMContentLoaded", function() {
	loadConfigs();
});

// Explicit exports for config management
window.deleteConfig = deleteConfig;
window.cardLaunch = cardLaunch;
window.cardStop = cardStop;
window.cardToggleFull = cardToggleFull;
window.loadConfig = loadConfig;
