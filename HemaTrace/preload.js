const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Window controls (existing)
  minimize : () => ipcRenderer.send('win-minimize'),
  maximize : () => ipcRenderer.send('win-maximize'),
  close    : () => ipcRenderer.send('win-close'),

  // New: send fingerprint image to Python backend via main process
  predictBloodGroup: (imageArrayBuffer) =>
    ipcRenderer.invoke('predict-blood-group', imageArrayBuffer),
});
