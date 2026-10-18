'use strict';

/**
 * Proceso principal de Electron: abre la ventana flotante y transparente
 * donde vive el avatar 3D de Miku (VRM). La logica de Three.js/VRM y la
 * conexion por WebSocket al backend de Python viven en renderer.js,
 * dentro de la ventana (proceso de renderizado), no aqui.
 */

const { app, BrowserWindow, screen } = require('electron');
const path = require('node:path');

const ANCHO_VENTANA = 420;
const ALTO_VENTANA = 640;
const MARGEN_ESQUINA = 24;

let ventanaMiku = null;

function crearVentana() {
  const pantalla = screen.getPrimaryDisplay();
  const { width, height } = pantalla.workAreaSize;

  ventanaMiku = new BrowserWindow({
    width: ANCHO_VENTANA,
    height: ALTO_VENTANA,
    x: width - ANCHO_VENTANA - MARGEN_ESQUINA,
    y: height - ALTO_VENTANA - MARGEN_ESQUINA,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    resizable: false,
    movable: true,
    skipTaskbar: true,
    hasShadow: false,
    backgroundColor: '#00000000',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // 'screen-saver' es el nivel mas alto de always-on-top en Windows/macOS:
  // se queda flotando encima de todo, incluso de apps en pantalla completa.
  ventanaMiku.setAlwaysOnTop(true, 'screen-saver');
  ventanaMiku.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  ventanaMiku.setMenuBarVisibility(false);

  // Explicito (aunque sea el default): la ventana NO es click-through.
  // setIgnoreMouseEvents(true) haria que TODOS los clics/teclado pasen
  // de largo hacia el escritorio, incluido el input de chat -- nunca se
  // activa aqui, precisamente para que la barra de chat (index.html)
  // reciba clics y teclado con normalidad. Lo que SI decide, elemento
  // por elemento, si algo es arrastrable o interactivo es el CSS
  // "-webkit-app-region" (drag/no-drag) dentro de index.html.
  ventanaMiku.setIgnoreMouseEvents(false);

  // Reenvia console.log/warn/error del renderer (Three.js/VRM) a esta
  // terminal, para poder depurar sin tener que abrir DevTools a mano.
  const NIVELES_CONSOLA = ['log', 'warn', 'error', 'debug'];
  ventanaMiku.webContents.on('console-message', (_evento, nivel, mensaje, linea, origen) => {
    const etiqueta = NIVELES_CONSOLA[nivel] ?? 'log';
    console.log(`[renderer:${etiqueta}] ${mensaje} (${origen}:${linea})`);
  });

  ventanaMiku.loadFile(path.join(__dirname, 'index.html'));

  ventanaMiku.on('closed', () => {
    ventanaMiku = null;
  });
}

app.whenReady().then(crearVentana);

app.on('window-all-closed', () => {
  // En Windows/Linux, cerrar la unica ventana cierra la app completa
  // (a diferencia de macOS, donde las apps suelen quedarse en el dock).
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    crearVentana();
  }
});
