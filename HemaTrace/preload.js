const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Window controls (existing)
  minimize : () => ipcRenderer.send('win-minimize'),
  maximize : () => ipcRenderer.send('win-maximize'),
  close    : () => ipcRenderer.send('win-close'),

  predictBloodGroup: (imageArrayBuffer) =>
    ipcRenderer.invoke('predict-blood-group', imageArrayBuffer),

  apiRequest: (opts) => ipcRenderer.invoke('api-request', opts),
});
