// Config management

async function generateNewConfig() {
	const cmdDisplay = document.getElementById("command-display");
	const command = cmdDisplay?.dataset?.raw || cmdDisplay?.innerText;

	if (!command || command.trim() === "") {
		alert("No command to save. Please generate a command first.");
		return;
	}

	// Check if this exact command already exists
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
		// Continue anyway - allow creation
	}

	// Extract key parameters from command for display
	const params = extractCommandParams(command);
	const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5);
	const configName = `config_${timestamp}`;

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

		// Set command display and sync to launch page
		const commandDisplay = document.getElementById("command-display");
		if (commandDisplay) {
			commandDisplay.innerText = command;
		}

		switchPage("configs");

		// Sync the launch command
		if (window.syncLaunchCommand) {
			syncLaunchCommand();
		}

		showNotification(`Config saved: ${configName}`);
	} catch (error) {
		alert("Error saving config: " + error.message);
	}
}

function extractCommandParams(command) {
	// Simple extraction of key parameters from command
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
		renderConfigs(configs);
	} catch (error) {
		console.error("Error loading configs:", error);
		document.getElementById("configs-list").innerHTML =
			'<div class="configs-empty">Error loading configs</div>';
	}
}

function renderConfigs(configs) {
	const container = document.getElementById("configs-list");

	if (configs.length === 0) {
		container.innerHTML =
			'<div class="configs-empty">No configs yet. Click "Generate Config" to create one.</div>';
		return;
	}

	container.innerHTML = configs.map((config, index) => `
		<div class="config-item">
			<div class="launch-bar config-launch-bar">
				<!-- Config on one line -->
				<div class="launch-config-line">
					<span class="config-text">${config.name}</span>
					<button type="button" class="cmd-link" onclick="toggleConfigCommandPreview('${index}')">Full</button>
					<button type="button" class="cmd-link launch-link" onclick="launchConfigTraining('${config.name}')">Launch</button>
					<button type="button" class="cmd-link delete-link" onclick="deleteConfig('${config.name}')">Delete</button>
				</div>

				<!-- Progress bar -->
				<div class="launch-progress-container">
					<div class="progress-bar">
						<div class="progress-fill config-progress-fill-${index}"></div>
						<div class="progress-text config-progress-text-${index}">0%</div>
					</div>
					<span class="progress-number config-progress-number-${index}">0%</span>
					<button type="button" class="cmd-link" onclick="switchPage('results')">View</button>
				</div>

				<div class="output full-width config-command-preview-${index}" style="display:none; margin-top:12px;"></div>
			</div>
		</div>
	`).join("");
}

function toggleConfigCommandPreview(index) {
	const preview = document.querySelector(`.config-command-preview-${index}`);
	if (preview) {
		preview.style.display = preview.style.display === 'none' ? 'block' : 'none';
		if (preview.style.display === 'block' && !preview.innerHTML) {
			// Get config from the rendered list and show its command
			const configs = document.querySelectorAll('.config-item');
			if (configs[index]) {
				const configName = configs[index].querySelector('.config-text').textContent;
				// Fetch config data if needed
				const encodedName = encodeURIComponent(configName);
				fetch(`/api/configs/get/${encodedName}`)
					.then(r => r.json())
					.then(config => {
						preview.textContent = config.command;
					})
					.catch(err => console.error("Error loading config preview:", err));
			}
		}
	}
}

async function loadConfig(configName) {
	try {
		const encodedName = encodeURIComponent(configName);
		const response = await fetch(`/api/configs/get/${encodedName}`);
		if (!response.ok) throw new Error("Failed to load config");

		const config = await response.json();
		// Parse command and populate form fields
		parseAndLoadCommand(config.command);
		switchPage("generate");
		showNotification(`Config loaded: ${configName}`);
	} catch (error) {
		alert("Error loading config: " + error.message);
	}
}

async function deleteConfig(configName) {
	if (!confirm(`Delete config "${configName}"?`)) return;

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
	// Parse command and populate form fields
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

	// Handle checkboxes for transforms
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
	// Simple notification - you could enhance this
	console.log(message);
}

async function launchConfigTraining(configName) {
	try {
		const encodedName = encodeURIComponent(configName);
		const response = await fetch(`/api/configs/get/${encodedName}`);
		if (!response.ok) throw new Error("Failed to load config");

		const config = await response.json();

		// Set the command display and prepare for launch
		const commandDisplay = document.getElementById("command-display");
		if (commandDisplay) {
			commandDisplay.innerText = config.command;
		}

		// Sync the launch command (update config line and progress bar)
		if (window.syncLaunchCommand) {
			syncLaunchCommand();
		}

		// Launch training immediately
		if (window.launchTraining) {
			setTimeout(() => launchTraining(), 100);
		}

		showNotification(`Launching config: ${configName}`);
	} catch (error) {
		alert("Error launching config: " + error.message);
	}
}

// Load configs when configs page is viewed
document.addEventListener("DOMContentLoaded", function() {
	// Load configs on page initialization
	loadConfigs();
});
