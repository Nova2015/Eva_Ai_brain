const { app, BrowserWindow } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

let processes = [];

function startProcess(command, args, cwd) {
  const child = spawn(command, args, {
    cwd,
    shell: false,
    windowsHide: true
  });

  processes.push(child);

  child.on("error", (err) => {
    console.error(`Process error: ${err.message}`);
  });

  return child;
}

function startServices() {
  const root = "D:\\Rag project";

  startProcess(
    path.join(root, ".venv", "Scripts", "python.exe"),
    ["-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "3000"],
    root
  );

  startProcess(
    path.join(root, ".venv", "Scripts", "python.exe"),
    ["-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"],
    root
  );

  startProcess(
    path.join(root, "eva-ml-service", ".venv", "Scripts", "python.exe"),
    ["app.py"],
    path.join(root, "eva-ml-service")
  );
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  console.log("DIR:", __dirname);
  console.log("INDEX:", path.join(__dirname, "build", "index.html"));
  console.log("EXISTS:", require("fs").existsSync(path.join(__dirname, "build", "index.html")));

  win.loadFile(path.join(__dirname, "build", "index.html"));
  win.webContents.openDevTools();
}

app.whenReady().then(() => {
  startServices();

  setTimeout(createWindow, 5000);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  processes.forEach((child) => {
    try {
      child.kill();
    } catch {}
  });
});