// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const clearBtn = document.getElementById('clear-btn');
const convertBtn = document.getElementById('convert-btn');
const validateBtn = document.getElementById('validate-btn');
const countrySelect = document.getElementById('country-select');
const formatBtns = document.querySelectorAll('.format-btn');

// Status elements
const emptyState = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const successState = document.getElementById('success-state');
const errorState = document.getElementById('error-state');
const warningsArea = document.getElementById('warnings-area');
const outputPath = document.getElementById('output-path');
const errorList = document.getElementById('error-list');
const warningList = document.getElementById('warning-list');
const openFolderBtn = document.getElementById('open-folder-btn');

// State
let currentFile = null;
let currentFilePath = null;
let outputFormat = 'ubl';
let lastOutputPath = null;

// Detect input format from file extension
function detectInputFormat(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  if (ext === 'json') return 'json';
  if (ext === 'xml') return 'xml';
  if (ext === 'csv') return 'csv';
  return 'json';
}

// Format file size
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Show state
function showState(state) {
  emptyState.classList.add('hidden');
  loadingState.classList.add('hidden');
  successState.classList.add('hidden');
  errorState.classList.add('hidden');
  warningsArea.classList.add('hidden');
  
  if (state === 'empty') emptyState.classList.remove('hidden');
  if (state === 'loading') loadingState.classList.remove('hidden');
  if (state === 'success') successState.classList.remove('hidden');
  if (state === 'error') errorState.classList.remove('hidden');
}


// Handle file selection
function handleFile(file, filePath) {
  currentFile = file;
  currentFilePath = filePath;
  
  // Update UI
  dropZone.classList.add('hidden');
  fileInfo.classList.remove('hidden');
  fileName.textContent = file.name;
  fileSize.textContent = formatFileSize(file.size);
  
  // Enable buttons
  convertBtn.disabled = false;
  validateBtn.disabled = false;
  
  showState('empty');
}

// Handle file from path (for macOS menu File > Open)
function handleFilePath(filePath) {
  const pathParts = filePath.split('/');
  const name = pathParts[pathParts.length - 1];
  
  currentFilePath = filePath;
  currentFile = { name, size: 0 }; // Size unknown from menu open
  
  // Update UI
  dropZone.classList.add('hidden');
  fileInfo.classList.remove('hidden');
  fileName.textContent = name;
  fileSize.textContent = 'Opened from menu';
  
  // Enable buttons
  convertBtn.disabled = false;
  validateBtn.disabled = false;
  
  showState('empty');
}

// Clear file
function clearFile() {
  currentFile = null;
  currentFilePath = null;
  
  dropZone.classList.remove('hidden');
  fileInfo.classList.add('hidden');
  
  convertBtn.disabled = true;
  validateBtn.disabled = true;
  
  showState('empty');
}

// Drop zone events
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  
  const file = e.dataTransfer.files[0];
  if (file) {
    // Get the file path from the dropped file
    const filePath = file.path;
    handleFile(file, filePath);
  }
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    const filePath = file.path;
    handleFile(file, filePath);
  }
});

clearBtn.addEventListener('click', clearFile);

// Format selection
formatBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    formatBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    outputFormat = btn.dataset.format;
  });
});

// Show errors
function showErrors(errors) {
  errorList.innerHTML = errors.map(err => `
    <div class="error-item">
      <div class="error-field">${err.field}</div>
      <div class="error-message">${err.message}</div>
    </div>
  `).join('');
  showState('error');
}

// Show warnings
function showWarnings(warnings) {
  if (warnings && warnings.length > 0) {
    warningList.innerHTML = warnings.map(w => `
      <div class="warning-item">
        <strong>${w.field}:</strong> ${w.message}
      </div>
    `).join('');
    warningsArea.classList.remove('hidden');
  }
}

// Convert button
convertBtn.addEventListener('click', async () => {
  if (!currentFilePath) return;
  
  showState('loading');
  
  const inputFormat = detectInputFormat(currentFile.name);
  const ciusCountry = countrySelect.value;
  
  try {
    const result = await window.electronAPI.convertFile({
      filePath: currentFilePath,
      inputFormat,
      outputFormat,
      ciusCountry
    });
    
    if (result.success) {
      lastOutputPath = result.outputPath;
      outputPath.textContent = result.outputFileName;
      showState('success');
      showWarnings(result.warnings);
    } else {
      showErrors(result.errors);
      showWarnings(result.warnings);
    }
  } catch (error) {
    showErrors([{ field: 'Error', message: error.message }]);
  }
});

// Validate button
validateBtn.addEventListener('click', async () => {
  if (!currentFilePath) return;
  
  showState('loading');
  
  const inputFormat = detectInputFormat(currentFile.name);
  
  try {
    const result = await window.electronAPI.validateFile({
      filePath: currentFilePath,
      inputFormat
    });
    
    if (result.isValid) {
      outputPath.textContent = 'Invoice is valid! ✓';
      showState('success');
      openFolderBtn.classList.add('hidden');
    } else {
      showErrors(result.errors);
    }
    showWarnings(result.warnings);
  } catch (error) {
    showErrors([{ field: 'Error', message: error.message }]);
  }
});

// Open folder button (Show in Finder on macOS)
openFolderBtn.addEventListener('click', () => {
  if (lastOutputPath) {
    window.electronAPI.openFolder(lastOutputPath);
  }
});

// Reset open folder button visibility on success
successState.addEventListener('transitionend', () => {
  if (!successState.classList.contains('hidden')) {
    openFolderBtn.classList.remove('hidden');
  }
});

// macOS specific - handle file opened from menu
if (window.electronAPI.onFileOpened) {
  window.electronAPI.onFileOpened((filePath) => {
    handleFilePath(filePath);
  });
}
