function copyCommand() {
	const source = document.getElementById("command-display");
	const command = source ? source.dataset.raw : "";
	navigator.clipboard.writeText(command).then(() => {
		const btn = document.querySelector(".cmd-copy-btn");
		if (btn) {
			btn.classList.add("copied");
			const copyIcon = btn.querySelector(".cmd-copy-icon");
			const checkIcon = btn.querySelector(".cmd-check-icon");
			const label = btn.querySelector(".cmd-copy-label");
			if (copyIcon) copyIcon.style.display = "none";
			if (checkIcon) checkIcon.style.display = "";
			if (label) label.textContent = "Copied";
			setTimeout(() => {
				btn.classList.remove("copied");
				if (copyIcon) copyIcon.style.display = "";
				if (checkIcon) checkIcon.style.display = "none";
				if (label) label.textContent = "Copy";
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
