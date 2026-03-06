// Legacy stub — all launch functionality now lives in configs.js (per-card launch).
// syncLaunchCommand is still called by nav.js and configs.js.

function syncLaunchCommand() {
	const src = document.getElementById("command-display");
	const dst = document.getElementById("launch-command-preview");
	if (src && dst) {
		dst.innerHTML = src.innerHTML;
		dst.dataset.raw = src.dataset.raw;
	}
}

window.syncLaunchCommand = syncLaunchCommand;
