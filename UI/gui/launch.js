let _eventSource = null;
let _pollInterval = null;
let _logCount = 0;
let _totalEpochs = 1;
let _currentEpoch = 0;

// Called by switchPage('launch') — sync command preview from Generate tab
function syncLaunchCommand() {
    const src = document.getElementById("command-display");
    const dst = document.getElementById("launch-command-preview");
    if (src && dst) {
        dst.innerHTML = src.innerHTML;
        dst.dataset.raw = src.dataset.raw;
    }

    // Update config summary
    updateConfigSummary();
}

function updateConfigSummary() {
    const dataset = document.getElementById("dataset").value || "-";
    const model = document.getElementById("model").value || "-";
    const epochs = document.getElementById("epochs").value || "-";

    // Extract dataset ID (e.g., "001" from "Dataset001_Cellpose")
    const datasetId = dataset.replace(/^Dataset(\d+).*/, "$1") || dataset;

    document.getElementById("config-dataset").textContent = datasetId;
    document.getElementById("config-model").textContent = model;
    document.getElementById("config-epochs").textContent = epochs;
}

function toggleCommandPreview() {
    const preview = document.getElementById("launch-command-preview");
    const btn = document.querySelector(".cmd-toggle");
    if (preview.style.display === "none") {
        preview.style.display = "block";
        btn.textContent = "Hide Full Command";
    } else {
        preview.style.display = "none";
        btn.textContent = "View Full Command";
    }
}

async function launchTraining() {
    let command = document.getElementById("launch-command-preview").dataset.raw;
    if (!command) {
        alert("No command to launch. Configure training in the Generate tab first.");
        return;
    }

    // Clean up multi-line command: remove backslashes and extra whitespace
    command = command.replace(/\s*\\\s*/g, " ").replace(/\s+/g, " ").trim();

    try {
        const res = await fetch("/api/launch", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({command}),
        });

        if (!res.ok) {
            let errorMsg = "Failed to launch";
            try {
                const data = await res.json();
                errorMsg = data.error || errorMsg;
            } catch (e) {
                // If response is not JSON, use status text
                errorMsg = res.statusText || errorMsg;
            }
            alert(`Error: ${errorMsg}`);
            return;
        }

        setRunningUI(true);
        document.getElementById("launch-terminal").innerHTML = "";
        _logCount = 0;
        _totalEpochs = parseInt(document.getElementById("config-epochs").textContent) || 1;
        _currentEpoch = 0;
        updateProgress();
        _startLogStream();
        _startResultsPolling();
    } catch (e) {
        alert(`Error: ${e.message}`);
    }
}

async function stopTraining() {
    await fetch("/api/launch/stop", {method: "POST"});
}

function _startLogStream() {
    _eventSource = new EventSource("/api/launch/logs");
    _eventSource.onmessage = (e) => _appendLog(e.data);
    _eventSource.addEventListener("done", () => {
        setRunningUI(false);
        _eventSource.close();
        _eventSource = null;
        _stopResultsPolling();
        loadResults(); // final refresh
    });
    _eventSource.onerror = () => {
        setRunningUI(false);
        _eventSource = null;
        _stopResultsPolling();
    };
}

function _appendLog(line) {
    const term = document.getElementById("launch-terminal");
    const div = document.createElement("div");
    div.className = "log-line";
    div.textContent = line;
    term.appendChild(div);
    term.scrollTop = term.scrollHeight;

    // Parse epoch progress from lines like "Epoch 1/2 - Loss: ..."
    const epochMatch = line.match(/Epoch\s+(\d+)\/(\d+)/);
    if (epochMatch) {
        _currentEpoch = parseInt(epochMatch[1]);
        _totalEpochs = parseInt(epochMatch[2]);
        console.log(`[Progress] Epoch ${_currentEpoch}/${_totalEpochs}`);
    }

    // Update progress
    _logCount++;
    updateProgress();
}

function updateProgress() {
    const progressFill = document.getElementById("launch-progress-fill");
    const progressText = document.getElementById("launch-progress-text");
    const progressNumber = document.getElementById("launch-progress-number");

    if (!setRunningUI.isRunning) {
        progressFill.style.width = "0%";
        progressText.textContent = "0%";
        progressNumber.textContent = "0%";
    } else {
        // Calculate progress based on epochs
        const progress = _totalEpochs > 0 ? (_currentEpoch / _totalEpochs) * 100 : 0;
        const clampedProgress = Math.min(progress, 100);
        progressFill.style.width = clampedProgress + "%";
        progressText.textContent = Math.round(clampedProgress) + "%";
        progressNumber.textContent = Math.round(clampedProgress) + "%";
        console.log(`[Bar Update] ${_currentEpoch}/${_totalEpochs} = ${Math.round(clampedProgress)}%`);
    }
}

function _startResultsPolling() {
    _pollInterval = setInterval(() => {
        // Always refresh; results.js loadResults() is idempotent
        loadResults();
        // If currently viewing a run's charts, refresh them too
        if (currentResult) {
            const updated = allResults.find(
                r => r.timestamp === currentResult.timestamp
            );
            if (updated) {
                drawLossChart(updated.metrics);
                drawMetricsChart(updated.metrics);
            }
        }
    }, 4000);
}

function _stopResultsPolling() {
    clearInterval(_pollInterval);
    _pollInterval = null;
}

function setRunningUI(running) {
    const launchBtn = document.getElementById("launch-btn");
    const stopBtn = document.getElementById("stop-btn");
    const progressFill = document.getElementById("launch-progress-fill");

    setRunningUI.isRunning = running;

    if (running) {
        launchBtn.style.display = "none";
        stopBtn.style.display = "";
    } else {
        launchBtn.style.display = "";
        stopBtn.style.display = "none";
        progressFill.style.width = "100%";
    }
}
