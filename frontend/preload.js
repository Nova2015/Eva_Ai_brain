const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("evaDesktop", {
  isDesktop: true
});