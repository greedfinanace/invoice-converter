const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');

// Import converters
const { jsonToUBL } = require('./lib/converters/jsonToUBL');
const { jsonToCII } = require('./lib/converters/jsonToCII');
const { jsonToPDF } = require('./lib/converters/jsonToPDF');
const { xmlToJSON } = require('./lib/parsers/xmlParser');
const { csvToJSON } = require('./lib/parsers/csvParser');
const { validateBusinessRules } = require('./lib/validators/businessRules');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#0f172a',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets', 'icon.png')
  });

  mainWindow.loadFile('renderer/index.html');
  
  // Open DevTools in development
  // mainWindow.webContents.openDevTools();
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Window controls
ipcMain.on('window-minimize', () => mainWindow.minimize());
ipcMain.on('window-maximize', () => {
  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow.maximize();
  }
});
ipcMain.on('window-close', () => mainWindow.close());

// File conversion handler
ipcMain.handle('convert-file', async (event, { filePath, inputFormat, outputFormat, ciusCountry }) => {
  try {
    // Read file
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    
    // Parse input to JSON
    let invoice;
    if (inputFormat === 'json') {
      invoice = JSON.parse(fileContent);
    } else if (inputFormat === 'xml') {
      const isUBL = fileContent.includes('urn:oasis:names:specification:ubl') || fileContent.includes('<Invoice');
      invoice = await xmlToJSON(fileContent, isUBL ? 'ubl' : 'cii');
    } else if (inputFormat === 'csv') {
      invoice = await csvToJSON(fileContent);
    }

    // Validate
    const errors = validateBusinessRules(invoice);
    const criticalErrors = errors.filter(e => e.severity === 'CRITICAL');
    
    if (criticalErrors.length > 0) {
      return { success: false, errors: criticalErrors, warnings: errors.filter(e => e.severity !== 'CRITICAL') };
    }

    // Convert
    let result;
    let extension;
    
    switch (outputFormat) {
      case 'ubl':
        result = await jsonToUBL(invoice, ciusCountry);
        extension = 'xml';
        break;
      case 'cii':
        result = await jsonToCII(invoice, ciusCountry);
        extension = 'xml';
        break;
      case 'pdf':
        result = jsonToPDF(invoice);
        extension = 'html';
        break;
      case 'json':
        result = JSON.stringify(invoice, null, 2);
        extension = 'json';
        break;
    }

    // Generate output path
    const inputDir = path.dirname(filePath);
    const inputName = path.basename(filePath, path.extname(filePath));
    const outputFileName = `${inputName}_converted_${outputFormat.toUpperCase()}.${extension}`;
    const outputPath = path.join(inputDir, outputFileName);

    // Write output file
    fs.writeFileSync(outputPath, result, 'utf-8');

    return {
      success: true,
      outputPath,
      outputFileName,
      warnings: errors.filter(e => e.severity !== 'CRITICAL')
    };

  } catch (error) {
    return {
      success: false,
      errors: [{ field: 'general', message: error.message, severity: 'CRITICAL' }]
    };
  }
});

// Validate file handler
ipcMain.handle('validate-file', async (event, { filePath, inputFormat }) => {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    
    let invoice;
    if (inputFormat === 'json') {
      invoice = JSON.parse(fileContent);
    } else if (inputFormat === 'xml') {
      const isUBL = fileContent.includes('urn:oasis:names:specification:ubl') || fileContent.includes('<Invoice');
      invoice = await xmlToJSON(fileContent, isUBL ? 'ubl' : 'cii');
    } else if (inputFormat === 'csv') {
      invoice = await csvToJSON(fileContent);
    }

    const errors = validateBusinessRules(invoice);
    
    return {
      isValid: errors.filter(e => e.severity === 'CRITICAL').length === 0,
      errors: errors.filter(e => e.severity === 'CRITICAL'),
      warnings: errors.filter(e => e.severity !== 'CRITICAL')
    };

  } catch (error) {
    return {
      isValid: false,
      errors: [{ field: 'parsing', message: error.message, severity: 'CRITICAL' }],
      warnings: []
    };
  }
});

// Open output folder
ipcMain.handle('open-folder', async (event, folderPath) => {
  shell.showItemInFolder(folderPath);
});

// Select output folder
ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });
  return result.filePaths[0];
});
