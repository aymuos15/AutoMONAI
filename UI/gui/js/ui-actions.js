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
