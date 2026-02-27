function toggleTheme() {
	const html = document.documentElement;
	const current = html.getAttribute("data-theme");
	const next = current === "light" ? "dark" : "light";

	if (next === "light") {
		html.setAttribute("data-theme", "light");
	} else {
		html.removeAttribute("data-theme");
	}

	localStorage.setItem("theme", next);
}

function initTheme() {
	const saved = localStorage.getItem("theme");
	if (saved === "light") {
		document.documentElement.setAttribute("data-theme", "light");
	}
}
