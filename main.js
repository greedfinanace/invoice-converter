const { app, BrowserWindow, ipcMain, dialog, shell, Menu } = require('electron');
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
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#0f172a',
    vibrancy: 'under-window',
    visualEffectState: 'active',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'assets', 'icon.icns')
  });

  mainWindow.loadFile('renderer/index.html');
  
  // Open DevTools in development
  // mainWindow.webContents.openDevTools();
}

// macOS App Menu
function createMenu() {
  const template = [
    {
      label: app.name,
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Invoice...',
          accelerator: 'CmdOrCtrl+O',
          click: async () => {
            const result = await dialog.showOpenDialog(mainWindow, {
              properties: ['openFile'],
              filters: [
                { name: 'Invoice Files', extensions: ['json', 'xml', 'csv'] },
                { name: 'All Files', extensions: ['*'] }
              ]
            });
            if (!result.canceled && result.filePaths.length > 0) {
              mainWindow.webContents.send('file-opened', result.filePaths[0]);
            }
          }
        },
        { type: 'separator' },
        { role: 'close' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectAll' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        { type: 'separator' },
        { role: 'front' },
        { type: 'separator' },
        { role: 'window' }
      ]
    },
    {
      role: 'help',
      submenu: [
        {
          label: 'Learn More',
          click: async () => {
            await shell.openExternal('https://en16931.eu');
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

app.whenReady().then(() => {
  createMenu();
  createWindow();
});

app.on('window-all-closed', () => {
  // On macOS, apps typically stay open until explicitly quit
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS, re-create window when dock icon is clicked
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Window controls (for custom title bar buttons if needed)
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

// Open output folder - macOS uses Finder
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
