async function loadDatasets() {
	try {
		const response = await fetch("/api/datasets");
		const datasets = await response.json();
		const select = document.getElementById("dataset");
		select.innerHTML = "";
		for (const [key, val] of Object.entries(datasets)) {
			const opt = document.createElement("option");
			opt.value = key;
			opt.textContent = `${key} (${val.name})`;
			select.appendChild(opt);
		}
		updateCommand();
	} catch (e) {
		console.error("Failed to load datasets:", e);
	}
}

async function loadDatasetClasses() {
	try {
		const response = await fetch("/api/models");
		const _data = await response.json();

		const trainSelect = document.getElementById("train_dataset");
		const inferenceSelect = document.getElementById("inference_dataset");

		for (const select of [trainSelect, inferenceSelect]) {
			select.innerHTML = "";
		}

		const classes = [
			"Dataset",
			"CacheDataset",
			"PersistentDataset",
			"SmartCacheDataset",
		];
		for (const className of classes) {
			const optT = document.createElement("option");
			optT.value = className;
			optT.textContent = className;
			trainSelect.appendChild(optT);

			const optI = document.createElement("option");
			optI.value = className;
			optI.textContent = className;
			inferenceSelect.appendChild(optI);
		}
		updateCommand();
	} catch (e) {
		console.error("Failed to load dataset classes:", e);
	}
}

async function loadModels() {
	try {
		const response = await fetch("/api/models");
		const models = await response.json();
		const select = document.getElementById("model");
		select.innerHTML = "";
		for (const [key, val] of Object.entries(models)) {
			const opt = document.createElement("option");
			opt.value = key;
			opt.textContent = `${val.name}`;
			select.appendChild(opt);
		}
		updateCommand();
	} catch (e) {
		console.error("Failed to load models:", e);
	}
}
