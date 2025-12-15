const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Window controls
  minimize: () => ipcRenderer.send('window-minimize'),
  maximize: () => ipcRenderer.send('window-maximize'),
  close: () => ipcRenderer.send('window-close'),
  
  // File operations
  convertFile: (options) => ipcRenderer.invoke('convert-file', options),
  validateFile: (options) => ipcRenderer.invoke('validate-file', options),
  openFolder: (path) => ipcRenderer.invoke('open-folder', path),
  selectFolder: () => ipcRenderer.invoke('select-folder'),
});
