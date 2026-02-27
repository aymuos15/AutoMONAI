let selectedResultIndex = 0;
const allTabs = [
	// Generate page tabs
	{ name: "Command", id: "command", page: "generate" },
	{ name: "Models", id: "models", page: "generate" },
	{ name: "Dataset Classes", id: "dataset-classes", page: "generate" },
	{ name: "Training Options", id: "training", page: "generate" },
	{ name: "Preprocessing", id: "preprocessing", page: "generate" },
	{ name: "Augmentation", id: "augmentation", page: "generate" },
	{ name: "Metrics", id: "metrics", page: "generate" },
	{ name: "Loss Functions", id: "loss", page: "generate" },
	{ name: "Device", id: "device", page: "generate" },
	{ name: "Active Transforms", id: "summary", page: "generate" },
	// Docs page tabs
	{ name: "Models (Docs)", id: "models", page: "docs" },
	{ name: "Dataset Classes (Docs)", id: "dataset-classes", page: "docs" },
	{ name: "Training Options (Docs)", id: "training", page: "docs" },
	{ name: "Preprocessing (Docs)", id: "preprocessing", page: "docs" },
	{ name: "Augmentation (Docs)", id: "augmentation", page: "docs" },
	{ name: "Metrics (Docs)", id: "metrics", page: "docs" },
	{ name: "Loss Functions (Docs)", id: "loss", page: "docs" },
	{ name: "Device (Docs)", id: "device", page: "docs" },
	// Main pages
	{ name: "Launch", id: null, page: "launch" },
	{ name: "Results", id: null, page: "results" },
];

function openTabSearch() {
	const modal = document.getElementById("tab-search-modal");
	modal.classList.add("active");
	document.getElementById("tab-search-input").focus();
	selectedResultIndex = 0;
	updateSearchResults("");
}

function closeTabSearch() {
	document.getElementById("tab-search-modal").classList.remove("active");
	document.getElementById("tab-search-input").value = "";
}

function updateSearchResults(query) {
	const results = query
		? allTabs.filter((tab) =>
				tab.name.toLowerCase().includes(query.toLowerCase()),
			)
		: allTabs;

	const resultsDiv = document.getElementById("tab-search-results");
	resultsDiv.innerHTML = "";
	selectedResultIndex = 0;

	results.forEach((tab, index) => {
		const div = document.createElement("div");
		div.className = "tab-search-result";
		if (index === 0) div.classList.add("selected");
		div.textContent = tab.name;
		div.dataset.tabPage = tab.page;
		div.dataset.tabId = tab.id;
		div.onclick = () => selectTab(tab);
		resultsDiv.appendChild(div);
	});
}

function selectTab(tab) {
	switchPage(tab.page);
	// Only switch sub-tabs if the tab has an id (pages like launch/results don't have sub-tabs)
	if (tab.id) {
		setTimeout(() => {
			const button = document.querySelector(
				`#${tab.page}-page .sub-tab[onclick="switchSubTab('${tab.id}')"]`,
			);
			if (button) {
				// Manually activate the sub-page and sub-tab
				const activePage = document.querySelector(".page.active");
				const subPages = activePage.querySelectorAll(".sub-page");
				subPages.forEach((page) => page.classList.remove("active"));

				const subTabs = activePage.querySelectorAll(".sub-tab");
				subTabs.forEach((t) => t.classList.remove("active"));

				activePage.querySelector(`#sub-${tab.id}`).classList.add("active");
				button.classList.add("active");
			}
			closeTabSearch();
		}, 0);
	} else {
		closeTabSearch();
	}
}

function navigateResults(direction) {
	const results = document.querySelectorAll(".tab-search-result");
	if (results.length === 0) return;

	results[selectedResultIndex].classList.remove("selected");
	selectedResultIndex =
		(selectedResultIndex + direction + results.length) % results.length;
	results[selectedResultIndex].classList.add("selected");
	results[selectedResultIndex].scrollIntoView({ block: "nearest" });
}
