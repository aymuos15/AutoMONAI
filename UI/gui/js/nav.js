function switchPage(pageName) {
	const pages = document.querySelectorAll(".page");
	pages.forEach((page) => {
		page.classList.remove("active");
	});

	const tabs = document.querySelectorAll(".nav-tab");
	tabs.forEach((tab) => {
		tab.classList.remove("active");
	});

	document.getElementById(`${pageName}-page`).classList.add("active");

	// Find and activate the corresponding nav tab
	const navTab = Array.from(tabs).find((tab) => {
		const pageName2 = tab.onclick.toString().match(/switchPage\('(\w+)'\)/)[1];
		return pageName2 === pageName;
	});
	if (navTab) navTab.classList.add("active");

	// Load results if switching to results page
	if (pageName === "results") {
		loadResults();
	}

	// Sync command if switching to launch page
	if (pageName === "launch") {
		syncLaunchCommand();
	}

	// Load configs if switching to configs page
	if (pageName === "configs") {
		loadConfigs();
	}
}

function switchSubTab(subTabName) {
	const activePage = document.querySelector(".page.active");
	const subPages = activePage.querySelectorAll(".sub-page");
	subPages.forEach((page) => {
		page.classList.remove("active");
	});

	const subTabs = activePage.querySelectorAll(".sub-tab");
	subTabs.forEach((tab) => {
		tab.classList.remove("active");
	});

	activePage.querySelector(`#sub-${subTabName}`).classList.add("active");
	event.target.classList.add("active");
}
