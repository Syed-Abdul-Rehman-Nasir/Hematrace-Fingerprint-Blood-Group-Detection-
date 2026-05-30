// HemaTrace — Electron main process (window, Python server, IPC)

const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');
const fs = require('fs');

const PYTHON_EXE = path.join(__dirname, '..', '.venv', 'Scripts', 'python.exe');
const SERVER_SCRIPT = path.join(__dirname, '..', 'backend', 'server.py');
const SERVER_URL = 'http://127.0.0.1:5000';

let pythonProcess = null;
let mainWindow;

function startPythonServer() {
  pythonProcess = spawn(PYTHON_EXE, [SERVER_SCRIPT], {
    cwd: path.join(__dirname, '..', 'backend'),
    stdio: ['ignore', 'pipe', 'pipe']
  });
  pythonProcess.stdout.on('data', d => console.log('[Python]', d.toString().trim()));
  pythonProcess.stderr.on('data', d => console.error('[Python ERR]', d.toString().trim()));
  pythonProcess.on('exit', code => console.log('[Python] exited', code));
}

function waitForServer(retries = 30, delayMs = 1000) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const check = () => {
      http.get(`${SERVER_URL}/health`, res => {
        if (res.statusCode === 200) resolve();
        else retry();
      }).on('error', retry);
    };
    const retry = () => {
      attempts++;
      if (attempts >= retries) reject(new Error('Python server did not start'));
      else setTimeout(check, delayMs);
    };
    check();
  });
}

function apiRequest(method, apiPath, body) {
  return new Promise((resolve) => {
    const payload = body ? JSON.stringify(body) : null;
    const options = {
      hostname: '127.0.0.1',
      port: 5000,
      path: apiPath,
      method,
      headers: payload
        ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) }
        : {}
    };
    const req = http.request(options, res => {
      let data = '';
      res.on('data', chunk => { data += chunk; });
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data) });
        } catch {
          resolve({ status: res.statusCode, data: { error: 'Invalid JSON from server' } });
        }
      });
    });
    req.on('error', err => resolve({ status: 0, data: { error: err.message } }));
    if (payload) req.write(payload);
    req.end();
  });
}

ipcMain.handle('api-request', async (_event, { method, path, body }) => {
  return apiRequest(method || 'GET', path, body);
});

// IPC handler: renderer sends ArrayBuffer of image bytes → forward to Python
ipcMain.handle('predict-blood-group', async (_event, arrayBuffer) => {
  const bytes = Buffer.from(arrayBuffer);
  const FormData = require('form-data');
  const form = new FormData();
  form.append('image', bytes, { filename: 'fingerprint.bmp',
    contentType: 'image/bmp' });

  return new Promise((resolve) => {
    const options = {
      hostname: '127.0.0.1', port: 5000, path: '/predict',
      method: 'POST', headers: form.getHeaders()
    };
    const req = http.request(options, res => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch { resolve({ error: 'Invalid JSON from server' }); }
      });
    });
    req.on('error', err => resolve({ error: err.message }));
    form.pipe(req);
  });
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 700,
    minWidth: 900,
    minHeight: 600,
    frame: false,
    titleBarStyle: 'hidden',
    icon: path.join(__dirname, 'assets', 'icons', 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    backgroundColor: '#F4F6F9',
    show: false
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  Menu.setApplicationMenu(null);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  startPythonServer();
  try {
    await waitForServer();
  } catch (e) {
    console.error('[HemaTrace]', e.message);
  }
  createWindow();
});

app.on('before-quit', () => {
  if (pythonProcess) pythonProcess.kill();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// IPC: Window Controls
ipcMain.on('win-close', () => mainWindow && mainWindow.close());
ipcMain.on('win-minimize', () => mainWindow && mainWindow.minimize());
ipcMain.on('win-maximize', () => {
  if (!mainWindow) return;
  mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize();
});
