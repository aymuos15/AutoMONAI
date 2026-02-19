package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const apiURL = "http://localhost:8888"

var serverProcess *exec.Cmd

// subTabFullNames are the sub-tab labels on wide terminals.
var subTabFullNames = []string{
	"Command",
	"Models",
	"Dataset Classes",
	"Training Options",
	"Preprocessing",
	"Augmentation",
	"Metrics",
	"Loss Functions",
	"Device",
	"Active Transforms",
}

// subTabShortNames are used on narrow terminals where full names don't fit.
var subTabShortNames = []string{
	"Cmd",
	"Models",
	"Dataset",
	"Training",
	"Preproc",
	"Augment",
	"Metrics",
	"Loss",
	"Device",
	"Transforms",
}

// Field index constants.
const (
	fieldDataset       = 0
	fieldModel         = 1
	fieldMetrics       = 2
	fieldLoss          = 3
	fieldTrainDS       = 4
	fieldInferDS       = 5
	fieldEpochs        = 6
	fieldBatchSize     = 7
	fieldLR            = 8
	fieldImgSize       = 9
	fieldWorkers       = 10
	fieldOutputDir     = 11
	fieldDevice        = 12
	fieldNormMinmax    = 13
	fieldNormZscore    = 14
	fieldCropCenter    = 15
	fieldCropRandom    = 16
	fieldAugment       = 17
	fieldAugRotate     = 18
	fieldAugRotateProb = 19
	fieldAugFlip       = 20
	fieldAugFlipProb   = 21
)

// Per-sub-tab grid layouts for wide terminals. Each entry is a row of field indices.
var subTabWideGrids = [][][]int{
	// Command (0) - no navigable fields
	{},
	// Models (1)
	{{fieldModel}},
	// Dataset Classes (2)
	{{fieldDataset}, {fieldTrainDS, fieldInferDS}, {fieldOutputDir}},
	// Training Options (3)
	{{fieldEpochs, fieldBatchSize}, {fieldLR, fieldImgSize}, {fieldWorkers}},
	// Preprocessing (4)
	{{fieldNormMinmax, fieldNormZscore}, {fieldCropCenter, fieldCropRandom}},
	// Augmentation (5)
	{{fieldAugRotate, fieldAugRotateProb}, {fieldAugFlip, fieldAugFlipProb}},
	// Metrics (6)
	{{fieldMetrics}},
	// Loss Functions (7)
	{{fieldLoss}},
	// Device (8)
	{{fieldDevice}},
	// Active Transforms (9) - no navigable fields
	{},
}

// Per-sub-tab grid layouts for narrow terminals (all stacked).
var subTabNarrowGrids = [][][]int{
	// Command (0) - no navigable fields
	{},
	// Models (1)
	{{fieldModel}},
	// Dataset Classes (2)
	{{fieldDataset}, {fieldTrainDS}, {fieldInferDS}, {fieldOutputDir}},
	// Training Options (3)
	{{fieldEpochs}, {fieldBatchSize}, {fieldLR}, {fieldImgSize}, {fieldWorkers}},
	// Preprocessing (4)
	{{fieldNormMinmax}, {fieldNormZscore}, {fieldCropCenter}, {fieldCropRandom}},
	// Augmentation (5)
	{{fieldAugRotate}, {fieldAugRotateProb}, {fieldAugFlip}, {fieldAugFlipProb}},
	// Metrics (6)
	{{fieldMetrics}},
	// Loss Functions (7)
	{{fieldLoss}},
	// Device (8)
	{{fieldDevice}},
	// Active Transforms (9) - no navigable fields
	{},
}

// Per-sub-tab row labels for wide terminals.
var subTabWideLabels = [][][]string{
	// Command (0) - no fields
	{},
	// Models (1)
	{{"MODEL"}},
	// Dataset Classes (2)
	{{"DATASET"}, {"TRAIN DATASET", "INFERENCE DATASET"}, {"OUTPUT DIR"}},
	// Training Options (3)
	{{"EPOCHS", "BATCH SIZE"}, {"LEARNING RATE", "IMAGE SIZE"}, {"WORKERS"}},
	// Preprocessing (4)
	{{"MINMAX NORM", "ZSCORE NORM"}, {"CENTER CROP", "RANDOM CROP"}},
	// Augmentation (5)
	{{"ROTATE", "ROTATE PROB"}, {"FLIP", "FLIP PROB"}},
	// Metrics (6)
	{{"METRICS"}},
	// Loss Functions (7)
	{{"LOSS"}},
	// Device (8)
	{{"DEVICE"}},
}

// Per-sub-tab row labels for narrow terminals.
var subTabNarrowLabels = [][][]string{
	// Command (0) - no fields
	{},
	// Models (1)
	{{"MODEL"}},
	// Dataset Classes (2)
	{{"DATASET"}, {"TRAIN DATASET CLASS"}, {"INFERENCE DATASET CLASS"}, {"OUTPUT DIR"}},
	// Training Options (3)
	{{"EPOCHS"}, {"BATCH SIZE"}, {"LEARNING RATE"}, {"IMAGE SIZE"}, {"WORKERS"}},
	// Preprocessing (4)
	{{"MINMAX NORM"}, {"ZSCORE NORM"}, {"CENTER CROP"}, {"RANDOM CROP"}},
	// Augmentation (5)
	{{"ROTATE"}, {"ROTATE PROB"}, {"FLIP"}, {"FLIP PROB"}},
	// Metrics (6)
	{{"METRICS"}},
	// Loss Functions (7)
	{{"LOSS"}},
	// Device (8)
	{{"DEVICE"}},
}

// collapseThreshold is the minimum content width before switching to single-column layout.
const collapseThreshold = 60

var (
	darkBg      = lipgloss.Color("0")
	darkFg      = lipgloss.Color("15")
	darkDim     = lipgloss.Color("8")
	darkBorder  = lipgloss.Color("8")
	darkInputBg = lipgloss.Color("0")

	labelStyle          = lipgloss.NewStyle().Foreground(darkDim).Bold(true)
	fieldBorder         = lipgloss.NewStyle().Border(lipgloss.NormalBorder()).BorderForeground(darkBorder)
	activeField         = lipgloss.NewStyle().Border(lipgloss.NormalBorder()).BorderForeground(darkFg)
	navTabActiveStyle   = lipgloss.NewStyle().Foreground(darkFg).Underline(true).Bold(true)
	navTabInactiveStyle = lipgloss.NewStyle().Foreground(darkDim)
	subTabActiveStyle   = lipgloss.NewStyle().Foreground(darkBg).Background(darkFg).Padding(0, 1).Bold(true)
	subTabInactiveStyle = lipgloss.NewStyle().Foreground(darkDim).Padding(0, 1)
	buttonStyle         = lipgloss.NewStyle().Foreground(darkBg).Background(darkFg).Padding(0, 2).Bold(true)
	dividerStyle        = lipgloss.NewStyle().Foreground(darkBorder)
	cmdStyle            = lipgloss.NewStyle().Foreground(darkFg).Background(darkInputBg).Border(lipgloss.NormalBorder()).BorderForeground(darkBorder).PaddingTop(1).PaddingBottom(1).PaddingLeft(1)
	cmdHighlight        = lipgloss.NewStyle().Foreground(lipgloss.Color("#FF5FAF")).Bold(true)
	docStyle            = lipgloss.NewStyle().Foreground(darkDim)
	docTitleStyle       = lipgloss.NewStyle().Foreground(darkFg).Bold(true)
	docInfoStyle        = lipgloss.NewStyle().Foreground(darkDim)
	footerStyle         = lipgloss.NewStyle().Foreground(darkDim)
	sectionHeadStyle    = lipgloss.NewStyle().Foreground(darkFg).Bold(true)
	cardBorderStyle     = lipgloss.NewStyle().Border(lipgloss.NormalBorder()).BorderForeground(darkBorder).Padding(0, 1)
)

type apiData struct {
	Datasets map[string]map[string]interface{} `json:"datasets"`
	Models   map[string]map[string]interface{} `json:"models"`
}

type formField struct {
	label   string
	value   string
	options []string
	isText  bool
	input   textinput.Model
}

type model struct {
	page        string
	subTab      int
	width       int
	height      int
	loading     bool
	data        apiData
	gridRow     int
	gridCol     int
	editing     bool
	viewportRow int
	fields      []formField
}

func (m model) isNarrow() bool {
	return m.contentWidth() < collapseThreshold
}

func (m model) currentGrid() [][]int {
	if m.isNarrow() {
		return subTabNarrowGrids[m.subTab]
	}
	return subTabWideGrids[m.subTab]
}

func (m model) currentLabels() [][]string {
	if m.isNarrow() {
		return subTabNarrowLabels[m.subTab]
	}
	return subTabWideLabels[m.subTab]
}

func (m model) selectedField() int {
	grid := m.currentGrid()
	if len(grid) == 0 {
		return -1
	}
	if m.gridRow >= len(grid) {
		return grid[0][0]
	}
	row := grid[m.gridRow]
	if m.gridCol >= len(row) {
		return row[0]
	}
	return row[m.gridCol]
}

func (m *model) adjustViewport() {
	grid := m.currentGrid()
	if len(grid) == 0 {
		m.viewportRow = 0
		return
	}

	visibleRows := m.availableContentRows()
	maxRows := len(grid)

	if m.gridRow < m.viewportRow {
		m.viewportRow = m.gridRow
	}
	if m.gridRow >= m.viewportRow+visibleRows {
		m.viewportRow = m.gridRow - visibleRows + 1
	}
	if m.viewportRow < 0 {
		m.viewportRow = 0
	}
	if m.viewportRow >= maxRows {
		m.viewportRow = maxRows - 1
	}
}

func (m *model) getVisibleRowRange() (int, int) {
	grid := m.currentGrid()
	if len(grid) == 0 {
		return 0, 0
	}

	visibleRows := m.availableContentRows()
	maxRows := len(grid)
	endRow := m.viewportRow + visibleRows
	if endRow > maxRows {
		endRow = maxRows
	}
	return m.viewportRow, endRow
}

func (m model) availableContentRows() int {
	linesPerRow := 5
	headerLines := strings.Count(m.renderHeader(), "\n")
	footerLines := strings.Count(m.renderFooter(), "\n")
	availableLines := m.height - headerLines - footerLines
	visibleRows := availableLines / linesPerRow
	if visibleRows < 1 {
		visibleRows = 1
	}
	return visibleRows
}

func newModel() model {
	fields := []formField{
		{label: "DATASET", value: "", options: []string{}, isText: false},
		{label: "MODEL", value: "", options: []string{}, isText: false},
		{label: "METRICS", value: "dice iou", options: []string{"dice", "iou", "dice iou"}, isText: false},
		{label: "LOSS", value: "dice", options: []string{"dice", "cross_entropy", "focal"}, isText: false},
		{label: "TRAIN DATASET CLASS", value: "Dataset", options: []string{"Dataset", "CacheDataset", "PersistentDataset", "SmartCacheDataset"}, isText: false},
		{label: "INFERENCE DATASET CLASS", value: "Dataset", options: []string{"Dataset", "CacheDataset", "PersistentDataset", "SmartCacheDataset"}, isText: false},
		{label: "EPOCHS", value: "1", options: []string{}, isText: true},
		{label: "BATCH SIZE", value: "4", options: []string{}, isText: true},
		{label: "LEARNING RATE", value: "0.0001", options: []string{}, isText: true},
		{label: "IMAGE SIZE", value: "128", options: []string{}, isText: true},
		{label: "WORKERS", value: "0", options: []string{}, isText: true},
		{label: "OUTPUT DIR", value: "./predictions", options: []string{}, isText: true},
		{label: "DEVICE", value: "cuda", options: []string{"cuda", "cpu"}, isText: false},
		{label: "NORM MINMAX", value: "false", options: []string{"false", "true"}, isText: false},
		{label: "NORM ZSCORE", value: "false", options: []string{"false", "true"}, isText: false},
		{label: "CROP CENTER", value: "false", options: []string{"false", "true"}, isText: false},
		{label: "CROP RANDOM", value: "false", options: []string{"false", "true"}, isText: false},
		{label: "AUGMENT", value: "false", options: []string{"false", "true"}, isText: false},
		{label: "AUG ROTATE", value: "false", options: []string{"false", "true"}, isText: false},
		{label: "AUG ROTATE PROB", value: "0.5", options: []string{}, isText: true},
		{label: "AUG FLIP", value: "false", options: []string{"false", "true"}, isText: false},
		{label: "AUG FLIP PROB", value: "0.5", options: []string{}, isText: true},
	}

	for i := range fields {
		if fields[i].isText {
			fields[i].input = textinput.New()
			fields[i].input.SetValue(fields[i].value)
		}
	}

	return model{
		page:        "generate",
		subTab:      0,
		loading:     true,
		fields:      fields,
		gridRow:     0,
		gridCol:     0,
		viewportRow: 0,
	}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(fetchData(), textinput.Blink)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		key := msg.String()

		if key == "ctrl+c" {
			return m, tea.Quit
		}

		// While editing, only enter/esc exit; everything else goes to the input.
		if m.editing {
			idx := m.selectedField()
			if key == "enter" || key == "esc" {
				m.fields[idx].input.Blur()
				m.fields[idx].value = m.fields[idx].input.Value()
				m.editing = false
				return m, nil
			}
			var cmd tea.Cmd
			m.fields[idx].input, cmd = m.fields[idx].input.Update(msg)
			m.fields[idx].value = m.fields[idx].input.Value()
			return m, cmd
		}

		// Global navigation keys.
		switch key {
		case "shift+left":
			if m.subTab > 0 {
				m.subTab--
				m.gridRow = 0
				m.gridCol = 0
				m.viewportRow = 0
			}
			return m, nil
		case "shift+right":
			if m.subTab < len(subTabFullNames)-1 {
				m.subTab++
				m.gridRow = 0
				m.gridCol = 0
				m.viewportRow = 0
			}
			return m, nil
		case "tab":
			if m.page == "generate" {
				m.page = "docs"
			} else {
				m.page = "generate"
			}
			m.gridRow = 0
			m.gridCol = 0
			m.viewportRow = 0
			return m, nil
		case "q":
			return m, tea.Quit
		}

		// Generate-page-only field navigation.
		if m.page == "generate" && len(m.currentGrid()) > 0 {
			grid := m.currentGrid()
			switch key {
			case "up":
				if m.gridRow > 0 {
					m.gridRow--
					if m.gridCol >= len(grid[m.gridRow]) {
						m.gridCol = len(grid[m.gridRow]) - 1
					}
					m.adjustViewport()
				}
				return m, nil
			case "down":
				if m.gridRow < len(grid)-1 {
					m.gridRow++
					if m.gridCol >= len(grid[m.gridRow]) {
						m.gridCol = len(grid[m.gridRow]) - 1
					}
					m.adjustViewport()
				}
				return m, nil
			case "left":
				if m.gridCol > 0 {
					m.gridCol--
				}
				return m, nil
			case "right":
				if m.gridCol < len(grid[m.gridRow])-1 {
					m.gridCol++
				}
				return m, nil
			case "enter":
				idx := m.selectedField()
				if m.fields[idx].isText {
					m.editing = true
					m.fields[idx].input.Focus()
					return m, textinput.Blink
				}
				m.cycleFieldOption(idx, true)
				return m, nil
			}
		}

	case tea.WindowSizeMsg:
		// Preserve the selected field across layout changes.
		prevField := m.selectedField()
		m.width = msg.Width
		m.height = msg.Height
		grid := m.currentGrid()
		if len(grid) > 0 && prevField >= 0 {
			for r, row := range grid {
				for c, idx := range row {
					if idx == prevField {
						m.gridRow = r
						m.gridCol = c
					}
				}
			}
		}
		m.adjustViewport()

	case dataFetchedMsg:
		m.data = msg.data
		m.loading = false

		datasets := []string{}
		for k := range m.data.Datasets {
			datasets = append(datasets, k)
		}
		models := []string{}
		for k := range m.data.Models {
			models = append(models, k)
		}

		m.fields[fieldDataset].options = datasets
		m.fields[fieldModel].options = models

		if len(datasets) > 0 {
			m.fields[fieldDataset].value = datasets[0]
		}
		if len(models) > 0 {
			m.fields[fieldModel].value = models[0]
		}
	}

	return m, nil
}

func (m *model) cycleFieldOption(fieldIdx int, forward bool) {
	field := &m.fields[fieldIdx]
	if len(field.options) == 0 {
		return
	}

	idx := indexOf(field.options, field.value)
	if idx < 0 {
		idx = 0
	}
	if forward {
		idx = (idx + 1) % len(field.options)
	} else {
		idx = (idx - 1 + len(field.options)) % len(field.options)
	}
	field.value = field.options[idx]
}

func indexOf(arr []string, val string) int {
	for i, v := range arr {
		if v == val {
			return i
		}
	}
	return -1
}

func (m model) View() string {
	if m.width == 0 {
		return "Loading..."
	}
	if m.loading {
		return "\nFetching API data...\nMake sure FastAPI server is running at http://localhost:8888\n"
	}
	if m.page == "generate" {
		return m.renderGenerate()
	}
	return m.renderDocs()
}

// renderHeader renders the shared nav tabs + sub-tabs bar used on both pages.
func (m model) renderHeader() string {
	cw := m.contentWidth()

	genActive := m.page == "generate"
	genTab := m.renderNavTab("Generate", genActive)
	docsTab := m.renderNavTab("Docs", !genActive)
	tabBar := genTab + "    " + docsTab
	centeredTabs := lipgloss.NewStyle().Width(cw).Align(lipgloss.Center).Render(tabBar)

	return "\n" +
		centeredTabs + "\n" +
		dividerStyle.Render(strings.Repeat("─", cw)) + "\n" +
		m.renderSubTabs() + "\n" +
		dividerStyle.Render(strings.Repeat("─", cw)) + "\n"
}

func (m model) renderNavTab(name string, active bool) string {
	if active {
		return navTabActiveStyle.Render(name)
	}
	return navTabInactiveStyle.Render(name)
}

func (m model) renderSubTabs() string {
	cw := m.contentWidth()
	names := subTabFullNames
	if cw < 95 {
		names = subTabShortNames
	}
	var parts []string
	for i, name := range names {
		if i == m.subTab {
			parts = append(parts, subTabActiveStyle.Render(name))
		} else {
			parts = append(parts, subTabInactiveStyle.Render(name))
		}
	}
	bar := strings.Join(parts, "")
	return lipgloss.NewStyle().MaxWidth(cw).Render(bar)
}

// truncateToLines keeps only the first n newline-terminated lines of s.
func truncateToLines(s string, n int) string {
	if n <= 0 {
		return ""
	}
	count := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			count++
			if count >= n {
				return s[:i+1]
			}
		}
	}
	return s
}

// assembleScreen builds a screen of exactly m.height lines: header at top,
// footer at bottom, content in between (truncated or padded to fit).
func (m model) assembleScreen(header, content, footer string) string {
	headerLines := strings.Count(header, "\n")
	footerLines := strings.Count(footer, "\n")
	maxContentLines := m.height - headerLines - footerLines
	if maxContentLines < 0 {
		maxContentLines = 0
	}

	contentLines := strings.Count(content, "\n")
	if contentLines > maxContentLines {
		content = truncateToLines(content, maxContentLines)
		contentLines = maxContentLines
	}

	totalUsed := headerLines + contentLines + footerLines
	result := header + content
	if totalUsed < m.height {
		result += strings.Repeat("\n", m.height-totalUsed-1)
	}
	result += footer
	return result
}

func (m model) renderGenerate() string {
	// Dispatch to special tabs without navigable fields.
	if m.subTab == 0 {
		return m.renderCommandTab()
	}
	if m.subTab == len(subTabFullNames)-1 {
		return m.renderActiveTransforms()
	}

	header := m.renderHeader()
	footer := m.renderFooter()

	grid := m.currentGrid()
	labels := m.currentLabels()
	startRow, endRow := m.getVisibleRowRange()

	var fieldContent string
	for rowIdx := startRow; rowIdx < endRow; rowIdx++ {
		if rowIdx >= len(grid) {
			break
		}
		indices := grid[rowIdx]
		rowLabels := labels[rowIdx]
		if len(indices) == 1 {
			fieldContent += "\n" + labelStyle.Render(rowLabels[0]) + "\n"
			fieldContent += m.renderField(indices[0]) + "\n\n"
		} else {
			fieldContent += "\n" + m.renderRow(indices, rowLabels) + "\n"
		}
	}

	return m.assembleScreen(header, fieldContent, footer)
}

func (m model) hasAnyAugmentation() bool {
	return m.fields[fieldAugRotate].value == "true" || m.fields[fieldAugFlip].value == "true"
}

func (m model) renderCommandTab() string {
	cw := m.contentWidth()
	header := m.renderHeader()
	footer := m.renderFooter()

	var content string
	content += "\n"
	content += labelStyle.Render("COMMAND") + "\n\n"
	content += cmdStyle.Width(cw - 3).Render(m.generateCommand()) + "\n\n"
	content += buttonStyle.Width(cw).Align(lipgloss.Center).Render("Copy (Ctrl+Shift+C)") + "\n"

	return m.assembleScreen(header, content, footer)
}

func (m model) renderActiveTransforms() string {
	header := m.renderHeader()
	footer := m.renderFooter()

	vals := make(map[string]string)
	for _, f := range m.fields {
		vals[f.label] = f.value
	}

	var content string
	content += "\n"

	// Preprocessing section.
	content += sectionHeadStyle.Render("Preprocessing") + "\n\n"
	var preproc []string
	if vals["NORM MINMAX"] == "true" {
		preproc = append(preproc, "MinMax Normalization")
	}
	if vals["NORM ZSCORE"] == "true" {
		preproc = append(preproc, "Z-Score Normalization")
	}
	if vals["CROP CENTER"] == "true" {
		preproc = append(preproc, "Center Crop")
	}
	if vals["CROP RANDOM"] == "true" {
		preproc = append(preproc, "Random Crop")
	}
	if len(preproc) == 0 {
		content += docStyle.Render("  None") + "\n"
	} else {
		for _, p := range preproc {
			content += docStyle.Render("  • "+p) + "\n"
		}
	}

	content += "\n"

	// Augmentation section.
	content += sectionHeadStyle.Render("Augmentation") + "\n\n"
	if !m.hasAnyAugmentation() {
		content += docStyle.Render("  Disabled") + "\n"
	} else {
		if vals["AUG ROTATE"] == "true" {
			content += docStyle.Render(fmt.Sprintf("  • Rotate (prob: %s)", vals["AUG ROTATE PROB"])) + "\n"
		}
		if vals["AUG FLIP"] == "true" {
			content += docStyle.Render(fmt.Sprintf("  • Flip (prob: %s)", vals["AUG FLIP PROB"])) + "\n"
		}
	}

	return m.assembleScreen(header, content, footer)
}

func (m model) renderDocs() string {
	cw := m.contentWidth()
	header := m.renderHeader()
	footer := m.renderFooter()

	var content string
	content += "\n"
	content += sectionHeadStyle.Render(subTabFullNames[m.subTab]) + "\n\n"

	type docCard struct {
		title string
		desc  string
		info  []string
	}

	allDocs := [][]docCard{
		// Command (0)
		{
			{"Command", "No documentation for this tab. Switch to Generate to see the generated command.", nil},
		},
		// Models (1)
		{
			{"UNet", "Standard U-Net architecture", nil},
			{"Attention UNet", "U-Net with attention gates", nil},
			{"SegResNet", "Segmentation ResNet", nil},
			{"SwinUNETR", "Swin Transformer-based UNETR", nil},
		},
		// Dataset Classes (1)
		{
			{"Dataset", "Basic in-memory dataset", nil},
			{"CacheDataset", "Caches entire dataset in memory for fast access", nil},
			{"PersistentDataset", "Persistent cache on disk with hash-based validity checking", nil},
			{"SmartCacheDataset", "Intelligent caching with automatic replacement", nil},
		},
		// Training Options (2)
		{
			{"Epochs", "Number of training epochs", []string{"Default: 1"}},
			{"Batch Size", "Number of samples per batch", []string{"Default: 4"}},
			{"Learning Rate", "Optimizer learning rate", []string{"Default: 0.0001"}},
			{"Image Size", "Image size for resizing (applies to all dimensions)", []string{"Default: 128"}},
			{"Workers", "Number of data loading workers", []string{"Default: 0"}},
		},
		// Preprocessing (3)
		{
			{"MinMax Normalization", "Scale intensity to [0, 1] range using min-max scaling", []string{"Formula: (x - min) / (max - min)"}},
			{"Z-Score Normalization", "Normalize to zero mean and unit variance", []string{"Formula: (x - mean) / std"}},
			{"Center Crop", "Crop image from the center to target size", []string{"Applied before resize"}},
			{"Random Crop", "Randomly crop image to target size", []string{"Applied before resize", "Training: random crop", "Inference: center crop"}},
		},
		// Augmentation (4)
		{
			{"Rotate", "Random rotation transform", []string{"Range: 0-90 degrees"}},
			{"Flip", "Random flip (horizontal and/or vertical)", []string{"Includes spatial flipping"}},
			{"Probability", "Probability for each augmentation transform", []string{"Range: 0.0 to 1.0"}},
		},
		// Metrics (5)
		{
			{"Dice", "Dice coefficient metric for segmentation evaluation. Measures overlap between predicted and ground truth segmentations.", []string{"Range: 0-1 (higher is better)", "Formula: 2 * |X ∩ Y| / (|X| + |Y|)"}},
			{"IoU (Intersection over Union)", "Also known as Jaccard Index. Measures the overlap ratio between predicted and ground truth segmentations.", []string{"Range: 0-1 (higher is better)", "Formula: |X ∩ Y| / |X ∪ Y|"}},
			{"Dice + IoU", "Combines both metrics for comprehensive evaluation. Provides complementary information about segmentation quality.", []string{"Recommended for balanced metric assessment"}},
		},
		// Loss Functions (6)
		{
			{"Dice Loss", "Directly optimizes the Dice metric. Particularly effective for imbalanced datasets where the background class dominates.", []string{"Default", "Best for: Segmentation tasks with class imbalance"}},
			{"Cross Entropy Loss", "Standard cross-entropy loss for multi-class classification. Measures the difference between predicted and ground truth distributions.", []string{"Formula: -Σ(y * log(ŷ))", "Best for: Balanced classification datasets"}},
			{"Focal Loss", "Focuses training on hard negative examples. Reduces loss contribution from easy examples.", []string{"Best for: Highly imbalanced datasets with rare classes", "Gamma: Controls focus on hard examples (default: 2)"}},
		},
		// Device (8)
		{
			{"CUDA", "Use GPU acceleration", nil},
			{"CPU", "Use CPU for computation", nil},
		},
		// Active Transforms (9)
		{
			{"Active Transforms", "No documentation for this tab. Switch to Generate to see active transforms.", nil},
		},
	}

	for _, card := range allDocs[m.subTab] {
		content += m.renderDocCard(card.title, card.desc, card.info, cw)
	}

	return m.assembleScreen(header, content, footer)
}

func (m model) renderDocCard(title, desc string, info []string, cw int) string {
	inner := docTitleStyle.Render(title) + "\n"
	inner += docStyle.Render(desc)
	for _, line := range info {
		inner += "\n" + docInfoStyle.Render(line)
	}
	card := cardBorderStyle.Width(cw - 4).Render(inner)
	return card + "\n\n"
}

func (m model) renderFooter() string {
	cw := m.contentWidth()
	left := footerStyle.Render("MonaiUI")

	var helpText string
	if m.editing {
		helpText = "Type to edit  •  Enter/Esc Confirm"
	} else if m.page == "generate" {
		helpText = "↑↓←→ Navigate  •  Shift+←→ Sub-tabs  •  Tab Switch  •  q Quit"
	} else {
		helpText = "Shift+←→ Sub-tabs  •  Tab Switch  •  q Quit"
	}
	right := footerStyle.Render(helpText)

	leftWidth := lipgloss.Width(left)
	rightWidth := lipgloss.Width(right)
	padding := cw - leftWidth - rightWidth
	if padding < 1 {
		padding = 1
	}
	return left + strings.Repeat(" ", padding) + right + "\n"
}

// contentWidth returns the usable content width.
func (m model) contentWidth() int {
	w := m.width - 2
	if w < 20 {
		w = 20
	}
	return w
}

func (m model) renderRow(indices []int, labels []string) string {
	cw := m.contentWidth()

	if m.isNarrow() {
		var result string
		for i, idx := range indices {
			result += labelStyle.Render(labels[i]) + "\n"
			result += m.renderFieldWithWidth(idx, cw-4) + "\n\n"
		}
		return result
	}

	colWidth := cw / len(indices)
	var cols []string
	for i, idx := range indices {
		labelText := labels[i]
		maxLabelLen := colWidth - 6
		if len(labelText) > maxLabelLen && maxLabelLen > 5 {
			labelText = labelText[:maxLabelLen-3] + "..."
		}
		label := labelStyle.Render(labelText)
		field := m.renderFieldWithWidth(idx, colWidth-4)
		col := lipgloss.NewStyle().Width(colWidth).Render(label + "\n" + field)
		cols = append(cols, col)
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, cols...)
}

func (m model) renderField(idx int) string {
	return m.renderFieldWithWidth(idx, m.contentWidth()-4)
}

func (m model) renderFieldWithWidth(idx, innerWidth int) string {
	f := m.fields[idx]
	isSelected := m.page == "generate" && idx == m.selectedField()

	var value string
	if f.isText {
		value = f.input.View()
	} else {
		value = f.value
	}

	if len(value) == 0 {
		value = ">"
	}
	if innerWidth < 4 {
		innerWidth = 4
	}

	style := fieldBorder.Padding(0, 1).Width(innerWidth)
	if isSelected {
		style = activeField.Padding(0, 1).Foreground(darkFg).Width(innerWidth)
	}
	return style.Render(value)
}

func (m model) generateCommand() string {
	vals := make(map[string]string)
	for _, f := range m.fields {
		vals[f.label] = f.value
	}

	h := cmdHighlight.Render

	cmd := fmt.Sprintf("python run.py --dataset %s --model %s --metrics %s --loss %s",
		h(vals["DATASET"]), h(vals["MODEL"]), h(vals["METRICS"]), h(vals["LOSS"]))

	if vals["TRAIN DATASET CLASS"] != "Dataset" {
		cmd += fmt.Sprintf(" --train_dataset_class %s", h(vals["TRAIN DATASET CLASS"]))
	}
	if vals["INFERENCE DATASET CLASS"] != "Dataset" {
		cmd += fmt.Sprintf(" --inference_dataset_class %s", h(vals["INFERENCE DATASET CLASS"]))
	}

	cmd += fmt.Sprintf(" --epochs %s --batch_size %s --lr %s --img_size %s --num_workers %s --output_dir %s --device %s",
		h(vals["EPOCHS"]), h(vals["BATCH SIZE"]), h(vals["LEARNING RATE"]), h(vals["IMAGE SIZE"]),
		h(vals["WORKERS"]), h(vals["OUTPUT DIR"]), h(vals["DEVICE"]))

	var normOpts []string
	if vals["NORM MINMAX"] == "true" {
		normOpts = append(normOpts, "minmax")
	}
	if vals["NORM ZSCORE"] == "true" {
		normOpts = append(normOpts, "zscore")
	}
	if len(normOpts) > 0 {
		cmd += fmt.Sprintf(" --norm %s", h(strings.Join(normOpts, " ")))
	}

	var cropOpts []string
	if vals["CROP CENTER"] == "true" {
		cropOpts = append(cropOpts, "center")
	}
	if vals["CROP RANDOM"] == "true" {
		cropOpts = append(cropOpts, "random")
	}
	if len(cropOpts) > 0 {
		cmd += fmt.Sprintf(" --crop %s", h(strings.Join(cropOpts, " ")))
	}

	if m.hasAnyAugmentation() {
		cmd += " --augment"
		if vals["AUG ROTATE"] == "true" {
			cmd += fmt.Sprintf(" --aug_rotate --aug_rotate_prob %s", h(vals["AUG ROTATE PROB"]))
		}
		if vals["AUG FLIP"] == "true" {
			cmd += fmt.Sprintf(" --aug_flip --aug_flip_prob %s", h(vals["AUG FLIP PROB"]))
		}
	}

	return cmd
}

type dataFetchedMsg struct {
	data apiData
}

func fetchData() tea.Cmd {
	return func() tea.Msg {
		var data apiData

		resp, _ := http.Get(apiURL + "/api/datasets")
		if resp != nil {
			defer resp.Body.Close()
			body, _ := io.ReadAll(resp.Body)
			json.Unmarshal(body, &data.Datasets) //nolint:errcheck
		}

		resp, _ = http.Get(apiURL + "/api/models")
		if resp != nil {
			defer resp.Body.Close()
			body, _ := io.ReadAll(resp.Body)
			json.Unmarshal(body, &data.Models) //nolint:errcheck
		}

		return dataFetchedMsg{data: data}
	}
}

// startServer starts the FastAPI server and waits for it to be ready.
func startServer() error {
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("could not get working directory: %v", err)
	}

	serverScript := filepath.Join(cwd, "UI", "server.py")
	if _, err := os.Stat(serverScript); os.IsNotExist(err) {
		serverScript = filepath.Join(cwd, "server.py")
		if _, err := os.Stat(serverScript); os.IsNotExist(err) {
			return fmt.Errorf("server.py not found in %s/UI or %s", cwd, cwd)
		}
	}

	serverProcess = exec.Command("python3", serverScript)
	serverProcess.Stdout = io.Discard
	serverProcess.Stderr = io.Discard

	if err := serverProcess.Start(); err != nil {
		return fmt.Errorf("failed to start server: %v", err)
	}

	maxRetries := 20
	retryDelay := time.Duration(500) * time.Millisecond

	for i := 0; i < maxRetries; i++ {
		resp, err := http.Get(apiURL + "/api/datasets")
		if err == nil && resp.StatusCode == 200 {
			resp.Body.Close()
			return nil
		}
		if resp != nil {
			resp.Body.Close()
		}
		time.Sleep(retryDelay)
	}

	return fmt.Errorf("server failed to start within 10 seconds")
}

// stopServer stops the FastAPI server.
func stopServer() {
	if serverProcess != nil && serverProcess.Process != nil {
		serverProcess.Process.Kill()
		serverProcess.Wait() //nolint:errcheck
	}
}

func main() {
	if err := startServer(); err != nil {
		fmt.Fprintf(os.Stderr, "Server start failed: %v\n", err)
		fmt.Fprintf(os.Stderr, "Make sure you're in the MonaiUI project root directory.\n")
		os.Exit(1)
	}
	defer stopServer()

	m := newModel()
	p := tea.NewProgram(m, tea.WithAltScreen())

	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
