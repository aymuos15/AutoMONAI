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
	loadResumeOptions();

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
				const tabPage = selected.dataset.tabPage;
				const tabId =
					selected.dataset.tabId === "null" ? null : selected.dataset.tabId;
				selectTab({ page: tabPage, id: tabId });
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
