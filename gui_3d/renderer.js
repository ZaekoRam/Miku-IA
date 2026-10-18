/**
 * Renderer de la ventana de Miku: carga el avatar VRM con Three.js,
 * anima el parpadeo automatico y mueve la boca segun los eventos que
 * llegan del backend de Python por WebSocket.
 *
 * Corre dentro de la ventana de Electron (proceso de renderizado), con
 * nodeIntegration desactivado -- solo usa APIs de navegador (WebSocket,
 * fetch, import de modulos ES), nada de Node.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

// --- Configuracion -----------------------------------------------------------

// Coloca aqui tu modelo de Miku en formato .vrm (ver gui_3d/assets/LEEME.txt).
const RUTA_MODELO_VRM = './assets/miku.vrm';

// Debe coincidir con el servidor WebSocket que exponga el backend de
// Python (ese servidor todavia no existe; ver nota en el chat).
const URL_WEBSOCKET = 'ws://localhost:8765';
const REINTENTO_WEBSOCKET_MS = 2000;

const PARPADEO_INTERVALO_MIN_MS = 2000;
const PARPADEO_INTERVALO_MAX_MS = 5000;
const PARPADEO_DURACION_MS = 150;

const SUAVIZADO_BOCA = 12; // que tan rapido la boca alcanza el nivel objetivo (mas alto = mas brusco)

// Cuanto espera el avatar, ya en silencio tras terminar de responder,
// antes de disparar la animacion de salida (fade-out/ocultar) -- "Timer
// AFK Dinamico": se reinicia solo cada vez que llega actividad nueva
// (ver mostrarAvatar -> cancelarOcultarAvatar), asi que en realidad
// mide inactividad, no un tiempo fijo desde la primera respuesta.
const RETRASO_OCULTAR_AVATAR_MS = 5000;

const RESPIRACION_VELOCIDAD_HZ = 0.25; // ciclos por segundo (~4s por respiracion completa)
const RESPIRACION_AMPLITUD_RAD = 0.025; // que tan notoria es la inclinacion del pecho

// Gesto de asentir al activarse ("Oye Miku" -> "¿Si?"): un solo cabeceo
// hacia abajo y de regreso, no un asentimiento repetido.
const ASENTIR_DURACION_SEG = 0.5;
const ASENTIR_AMPLITUD_RAD = 0.35; // ~20 grados de inclinacion hacia abajo en el pico del gesto

const MIRADA_RANGO_X = 0.3; // cuanto se mueve el objetivo de la mirada a los lados
const MIRADA_RANGO_Y = 0.15; // cuanto se mueve arriba/abajo

// Distancia fija de la camara a la cabeza en Z: valores chicos (~1.3)
// quedan demasiado cerca (se corta pelo/coletas); esto da un plano
// medio/busto completo, con aire alrededor, en vez de un primer plano.
const DISTANCIA_CAMARA_Z = 3.0;

// --- Escena de Three.js --------------------------------------------------------

const lienzo = document.getElementById('lienzo');

const escena = new THREE.Scene();

const camara = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 20);
camara.position.set(0, 1.4, 1.0);
// Sin esto la camara mira derecho por -Z sin apuntar a la cara del
// avatar (quedaba encuadrando el pecho/falda en vez del rostro).
camara.lookAt(0, 1.35, 0);

const renderizador = new THREE.WebGLRenderer({ canvas: lienzo, alpha: true, antialias: true });
renderizador.setSize(window.innerWidth, window.innerHeight);
renderizador.setPixelRatio(window.devicePixelRatio);
renderizador.setClearColor(0x000000, 0); // fondo transparente: se ve el escritorio detras

// OrbitControls se usa SOLO por su ".target": el punto al que la camara
// mira. Se reubica en la cabeza del VRM apenas carga (ver
// encuadrarCamaraEnModelo) para que no se quede apuntando al origen.
// No queremos que el usuario pueda orbitar/mover/hacer zoom con el
// mouse (esto es un avatar de escritorio de encuadre fijo, no un
// visor 3D interactivo) -- sin desactivar el zoom, la ruedita del
// mouse agrandaba/achicaba a Miku sin querer.
const controls = new OrbitControls(camara, renderizador.domElement);
controls.enableZoom = false;
controls.enablePan = false;
controls.enableRotate = false;

escena.add(new THREE.DirectionalLight(0xffffff, 1.2).translateX(1).translateY(1).translateZ(1));
escena.add(new THREE.AmbientLight(0xffffff, 0.7));

// Objetivo de la mirada (vrm.lookAt lo sigue automaticamente en cada
// vrm.update()): un punto invisible frente a la cara que se mueve un
// poco con el cursor, para un seguimiento sutil, no un giro exagerado.
const objetivoMirada = new THREE.Object3D();
objetivoMirada.position.set(0, 1.4, 1.0);
escena.add(objetivoMirada);

window.addEventListener('pointermove', (evento) => {
  const x = (evento.clientX / window.innerWidth) * 2 - 1;
  const y = -(evento.clientY / window.innerHeight) * 2 + 1;
  const alturaCabeza = controls.target.y || 1.4;
  objetivoMirada.position.set(x * MIRADA_RANGO_X, alturaCabeza + y * MIRADA_RANGO_Y, controls.target.z + 1.0);
});

window.addEventListener('resize', () => {
  camara.aspect = window.innerWidth / window.innerHeight;
  camara.updateProjectionMatrix();
  renderizador.setSize(window.innerWidth, window.innerHeight);
});

// --- Carga del modelo VRM de Miku ----------------------------------------------

let vrmActual = null;
let huesoCabezaActual = null;
let tiempoInicioAsentir = null; // null = sin gesto activo; si no, segundo de tiempoAnimacion en que empezo

/**
 * Los VRM cargan en su pose de bind (normalmente T-pose: brazos
 * estirados en cruz) porque no traen ninguna animacion de reposo.
 * Como todavia no hay un sistema de animacion (VRMA), bajamos los
 * brazos a mano rotando los huesos del esqueleto humanoid para que
 * Miku se vea de pie de forma natural en vez de en cruz.
 */
function aplicarPoseDeReposo(vrm) {
  const humanoid = vrm.humanoid;
  if (!humanoid) {
    console.warn('[Miku-GUI] El VRM no trae esqueleto humanoid: no se puede arreglar la pose.');
    return;
  }

  const rotarHueso = (nombreHueso, gradosZ) => {
    const hueso = humanoid.getNormalizedBoneNode(nombreHueso);
    if (hueso) {
      hueso.rotation.z = THREE.MathUtils.degToRad(gradosZ);
      console.log(`[Miku-GUI] Pose: ${nombreHueso} rotado a ${gradosZ} grados (encontrado).`);
    } else {
      console.warn(`[Miku-GUI] Pose: no se encontro el hueso "${nombreHueso}" en este VRM.`);
    }
  };

  // Brazo/antebrazo izquierdo y derecho hacia abajo (espejados en Z).
  rotarHueso('leftUpperArm', 65);
  rotarHueso('rightUpperArm', -65);
  rotarHueso('leftLowerArm', 15);
  rotarHueso('rightLowerArm', -15);
}

/**
 * Encuadra la camara sobre la cara del VRM ya cargado, calculando todo
 * a partir del modelo real en vez de valores fijos -- necesario porque
 * cada VRM puede venir con una escala y un origen distintos.
 */
function encuadrarCamaraEnModelo(vrm) {
  // Fuerza a recalcular las matrices de mundo ahora mismo: todavia no
  // se renderizo ningun frame, asi que Box3 podria ver matrices viejas
  // (identidad) si no se actualizan a mano.
  vrm.scene.updateMatrixWorld(true);

  const caja = new THREE.Box3().setFromObject(vrm.scene);
  const centro = caja.getCenter(new THREE.Vector3());
  const tamano = caja.getSize(new THREE.Vector3());

  // El hueso de la cabeza del esqueleto humanoid de VRM es mas preciso
  // que estimar la altura de la cara a partir de la caja completa
  // (sobre todo si el modelo tiene pelo/accesorios que la alargan).
  // Si el modelo no trae esqueleto humanoid valido, se usa la caja como
  // respaldo.
  const huesoCabeza = vrm.humanoid?.getNormalizedBoneNode('head');
  const posicionCabeza = new THREE.Vector3();
  if (huesoCabeza) {
    huesoCabeza.getWorldPosition(posicionCabeza);
  } else {
    posicionCabeza.set(centro.x, centro.y + tamano.y * 0.35, centro.z);
  }

  // console.log con objetos THREE.Vector3 llega como "[object Object]"
  // al reenviarse por IPC a la terminal (main_electron.js); se arman
  // strings con numeros para poder depurar de verdad.
  console.log(
    `[Miku-GUI] Cabeza via ${huesoCabeza ? 'esqueleto humanoid' : 'caja (fallback, sin humanoid.head)'}`
  );
  console.log(
    `[Miku-GUI] caja min=(${caja.min.x.toFixed(3)}, ${caja.min.y.toFixed(3)}, ${caja.min.z.toFixed(3)})` +
      ` max=(${caja.max.x.toFixed(3)}, ${caja.max.y.toFixed(3)}, ${caja.max.z.toFixed(3)})`
  );
  console.log(
    `[Miku-GUI] tamano=(${tamano.x.toFixed(3)}, ${tamano.y.toFixed(3)}, ${tamano.z.toFixed(3)})` +
      ` centro=(${centro.x.toFixed(3)}, ${centro.y.toFixed(3)}, ${centro.z.toFixed(3)})`
  );
  console.log(
    `[Miku-GUI] cabeza=(${posicionCabeza.x.toFixed(3)}, ${posicionCabeza.y.toFixed(3)}, ${posicionCabeza.z.toFixed(3)})` +
      ` distancia=${DISTANCIA_CAMARA_Z} escala.vrm.scene=(${vrm.scene.scale.x}, ${vrm.scene.scale.y}, ${vrm.scene.scale.z})`
  );

  // El target de OrbitControls es lo que hace que quede anclado a la
  // cabeza en vez de resetearse al origen (0,0,0).
  controls.target.copy(posicionCabeza);
  controls.update();

  camara.position.set(posicionCabeza.x, posicionCabeza.y, posicionCabeza.z + DISTANCIA_CAMARA_Z);
  camara.lookAt(posicionCabeza);
}

const cargador = new GLTFLoader();
cargador.register((parser) => new VRMLoaderPlugin(parser));

cargador.load(
  RUTA_MODELO_VRM,
  (gltf) => {
    const vrm = gltf.userData.vrm;

    // Limpieza recomendada por three-vrm: quita vertices/joints que no
    // se usan y evita warnings de material.
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.removeUnnecessaryJoints(gltf.scene);

    vrm.scene.rotation.y = Math.PI; // los VRM miran hacia -Z; los giramos para que miren a la camara

    aplicarPoseDeReposo(vrm); // arregla el T-pose antes de medir/encuadrar

    escena.add(vrm.scene);
    vrmActual = vrm;
    huesoCabezaActual = vrm.humanoid?.getNormalizedBoneNode('head') ?? null;

    if (vrm.lookAt) {
      vrm.lookAt.target = objetivoMirada;
    }

    encuadrarCamaraEnModelo(vrm);

    console.log('[Miku-GUI] Modelo VRM cargado.');
  },
  undefined,
  (error) => {
    console.error('[Miku-GUI] No se pudo cargar el modelo VRM en', RUTA_MODELO_VRM, error);
  }
);

// --- Parpadeo automatico ---------------------------------------------------------

function programarSiguienteParpadeo() {
  const espera =
    PARPADEO_INTERVALO_MIN_MS + Math.random() * (PARPADEO_INTERVALO_MAX_MS - PARPADEO_INTERVALO_MIN_MS);
  setTimeout(parpadear, espera);
}

function parpadear() {
  const expresiones = vrmActual?.expressionManager;
  if (expresiones) {
    expresiones.setValue('blink', 1.0);
    setTimeout(() => expresiones.setValue('blink', 0.0), PARPADEO_DURACION_MS);
  }
  programarSiguienteParpadeo();
}

programarSiguienteParpadeo();

// --- Boca al hablar, controlada por el backend de Python via WebSocket -----------

let nivelBocaObjetivo = 0;
let nivelBocaActual = 0;

// Referencia al WebSocket vivo, para poder mandar mensajes (el chat de
// texto) y no solo recibirlos. null mientras no hay conexion.
let socketActual = null;

function conectarWebSocket() {
  const socket = new WebSocket(URL_WEBSOCKET);

  socket.addEventListener('open', () => {
    socketActual = socket;
    console.log('[Miku-GUI] Conectado al backend por WebSocket.');
  });

  socket.addEventListener('message', (evento) => {
    manejarMensajeBackend(evento.data);
  });

  socket.addEventListener('close', () => {
    socketActual = null;
    nivelBocaObjetivo = 0;
    setTimeout(conectarWebSocket, REINTENTO_WEBSOCKET_MS);
  });

  socket.addEventListener('error', () => {
    socket.close();
  });
}

/**
 * Formato de mensaje esperado del backend (JSON):
 *   {"tipo": "hablando_inicio"}
 *   {"tipo": "hablando_nivel", "nivel": 0.0-1.0}   // amplitud del audio en ese instante
 *   {"tipo": "hablando_fin"}
 *   {"tipo": "activar"}                            // se detecto "Oye Miku": aparece el avatar
 *   {"tipo": "dormir"}                              // "Miku duerme"/"modo juego": oculta y pausa el render
 *   {"tipo": "respuesta_completa"}                  // termino de decir TODA la respuesta -> ocultar tras un rato
 *   {"tipo": "escuchando_inicio"}                   // empezo a grabar el comando -> prende el halo
 *   {"tipo": "escuchando_fin"}                      // termino de grabar el comando -> apaga el halo
 *   {"tipo": "chat_respuesta", "texto": "..."}      // modo hibrido: respuesta de Miku a un mensaje escrito
 */
function manejarMensajeBackend(datoCrudo) {
  let mensaje;
  try {
    mensaje = JSON.parse(datoCrudo);
  } catch {
    return;
  }

  switch (mensaje.tipo) {
    case 'hablando_inicio':
      // Regla dura: si esta hablando, tiene que estar en pantalla --
      // sin esto, un turno nuevo que arranca a mitad del temporizador
      // de ocultar (programado por el 'respuesta_completa' del turno
      // ANTERIOR, ver modo conversacion) podia hacer que el avatar se
      // ocultara mientras Miku todavia estaba hablando la respuesta
      // siguiente. mostrarAvatar() cancela ese temporizador viejo y
      // asegura que este visible, sea la primera vez o no.
      mostrarAvatar();
      nivelBocaObjetivo = 0.6;
      break;
    case 'hablando_nivel':
      nivelBocaObjetivo = Math.min(1, Math.max(0, Number(mensaje.nivel) || 0));
      break;
    case 'hablando_fin':
      nivelBocaObjetivo = 0;
      break;
    case 'activar':
      // "Oye Miku" se detecto del lado de Python: aparece el avatar
      // (animacion de entrada, estilo widget) y sale de "modo reposo"
      // si estaba pausado, sin esperar a que la ventana vuelva a ser
      // visible. El gesto de asentir (ver ASENTIR_* arriba y el manejo
      // en animar()) SOLO se dispara si el mensaje trae asentir=true --
      // main.py lo manda en false cuando "oye miku" ya vino con una
      // instruccion pegada en la misma respiracion, o cuando es un
      // mensaje de texto (que siempre trae contenido): en esos casos
      // asentir se veria fuera de lugar, justo antes de responder de
      // una en vez de estar "esperando" algo.
      mostrarAvatar();
      if (mensaje.asentir) {
        tiempoInicioAsentir = tiempoAnimacion;
      }
      break;
    case 'dormir':
      // Modo juego: oculta el avatar de inmediato (sin esperar los 2s
      // normales) y pausa el render aunque la ventana siga visible --
      // el usuario puede estar jugando con Miku flotando encima, y aun
      // asi queremos frenar el render para liberar GPU.
      ocultarHaloEscuchando();
      ocultarAvatar();
      detenerAnimacion();
      break;
    case 'respuesta_completa':
      // Termino de decir TODA la respuesta: programa la salida
      // (fade-out) tras un par de segundos, no de inmediato.
      programarOcultarAvatar();
      break;
    case 'escuchando_inicio':
      mostrarHaloEscuchando();
      break;
    case 'escuchando_fin':
      ocultarHaloEscuchando();
      break;
    case 'chat_respuesta':
      agregarMensajeChat(String(mensaje.texto ?? ''), true);
      break;
    default:
      break;
  }
}

conectarWebSocket();

// --- Barra de chat de texto (modo hibrido Universidad/Silencioso) -----------------

const contenedorChat = document.getElementById('chat-container');
const entradaChat = document.getElementById('chat-input');
const botonEnviarChat = document.getElementById('chat-enviar');
const contenedorChatLog = document.getElementById('chat-log');
const botonToggleChatLog = document.getElementById('chat-log-toggle');
const PLACEHOLDER_NORMAL = entradaChat.placeholder;
const PLACEHOLDER_ESCUCHANDO = 'Escuchando...';

// El log de chat NUNCA aparece solo -- se queda armado en el DOM (para
// no perder el historial) pero oculto hasta que el usuario lo pida con
// este boton. Un clic lo despliega, otro lo vuelve a esconder.
botonToggleChatLog.addEventListener('click', () => {
  const visible = contenedorChatLog.classList.toggle('chat-log-visible');
  botonToggleChatLog.classList.toggle('chat-log-toggle-activo', visible);
});

function mostrarHaloEscuchando() {
  contenedorChat.classList.add('escuchando');
  entradaChat.placeholder = PLACEHOLDER_ESCUCHANDO;
}

function ocultarHaloEscuchando() {
  contenedorChat.classList.remove('escuchando');
  entradaChat.placeholder = PLACEHOLDER_NORMAL;
}

/**
 * Agrega un mensaje al log de chat (modo hibrido: cuando le escribis a
 * Miku en vez de hablarle, contesta por aca ademas de en voz alta -- ver
 * main.py, al_recibir_comando(via_texto=True)). esDeMiku=false para tu
 * propio mensaje (eco inmediato, sin esperar al backend).
 *
 * OJO: esto NO muestra el panel -- se queda armando el historial en el
 * DOM aunque este oculto, para que cuando el usuario lo abra con el
 * boton (ver botonToggleChatLog) ya este todo ahi.
 */
function agregarMensajeChat(texto, esDeMiku) {
  if (!texto) {
    return;
  }
  const burbuja = document.createElement('div');
  burbuja.className = `chat-mensaje ${esDeMiku ? 'chat-mensaje-miku' : 'chat-mensaje-tu'}`;
  burbuja.textContent = texto;
  contenedorChatLog.appendChild(burbuja);
  contenedorChatLog.scrollTop = contenedorChatLog.scrollHeight;
}

function enviarMensajeChat() {
  const texto = entradaChat.value.trim();
  if (!texto) {
    return;
  }

  if (!socketActual || socketActual.readyState !== WebSocket.OPEN) {
    console.warn('[Miku-GUI] No se pudo mandar el mensaje: sin conexion con el backend todavia.');
    return;
  }

  socketActual.send(JSON.stringify({ type: 'user_message', text: texto }));
  agregarMensajeChat(texto, false);
  entradaChat.value = '';
}

botonEnviarChat.addEventListener('click', enviarMensajeChat);

entradaChat.addEventListener('keydown', (evento) => {
  if (evento.key === 'Enter') {
    evento.preventDefault();
    enviarMensajeChat();
  }
});

// --- Bucle de render (con "modo reposo") -------------------------------------------

const reloj = new THREE.Clock();
let tiempoAnimacion = 0;

// id de requestAnimationFrame; null significa "el loop esta parado"
// (modo reposo). cancelAnimationFrame detiene el render por completo
// -- 0% de uso de GPU -- sin tocar el modelo VRM ni sus texturas, que
// se quedan en VRAM tal cual para poder reanudar al instante.
let idCuadroAnimacion = null;

function animar() {
  idCuadroAnimacion = requestAnimationFrame(animar);
  const delta = reloj.getDelta();
  tiempoAnimacion += delta;

  // Suaviza el movimiento de la boca en vez de que salte de golpe entre niveles.
  nivelBocaActual += (nivelBocaObjetivo - nivelBocaActual) * Math.min(1, delta * SUAVIZADO_BOCA);

  if (vrmActual) {
    vrmActual.expressionManager?.setValue('aa', nivelBocaActual);

    // Respiracion: una oscilacion senoidal leve sobre el hueso del
    // pecho (o la columna si el modelo no trae "chest"), para que no
    // se vea completamente estatica en reposo.
    const huesoPecho =
      vrmActual.humanoid?.getNormalizedBoneNode('chest') ??
      vrmActual.humanoid?.getNormalizedBoneNode('spine');
    if (huesoPecho) {
      huesoPecho.rotation.x =
        Math.sin(tiempoAnimacion * RESPIRACION_VELOCIDAD_HZ * Math.PI * 2) * RESPIRACION_AMPLITUD_RAD;
    }

    // vrm.update() tambien actualiza vrm.lookAt (sigue a objetivoMirada)
    // ademas de las springbones y expresiones -- si el VRM usa lookAt
    // por hueso, esto puede tocar la rotacion de la cabeza. El gesto de
    // asentir se aplica DESPUES (abajo), para que siempre gane sobre lo
    // que haya calculado el lookAt mientras el gesto esta activo.
    vrmActual.update(delta);

    // Gesto de asentir (activado en case 'activar'): un cabeceo suave,
    // una sola vez -- sube y baja siguiendo media curva de seno (0 en el
    // arranque, pico a la mitad, 0 otra vez al terminar), en vez de un
    // asentimiento que se repite en loop como la respiracion de arriba.
    // Signo NEGATIVO: probado en vivo que positivo inclinaba la cabeza
    // hacia ARRIBA en este modelo (convencion de eje invertida respecto
    // a lo esperado) -- negativo es lo que de verdad se ve como asentir
    // hacia abajo.
    if (huesoCabezaActual && tiempoInicioAsentir !== null) {
      const progreso = (tiempoAnimacion - tiempoInicioAsentir) / ASENTIR_DURACION_SEG;
      if (progreso >= 1) {
        huesoCabezaActual.rotation.x = 0;
        tiempoInicioAsentir = null;
      } else {
        huesoCabezaActual.rotation.x = -Math.sin(progreso * Math.PI) * ASENTIR_AMPLITUD_RAD;
      }
    }
  }

  // Reafirma hacia donde mira la camara sin tocar camara.position, para
  // mantener el plano fijo (no llamamos controls.update() aqui: eso
  // recalcularia camara.position a partir del estado interno de
  // OrbitControls y podria mover el encuadre).
  camara.lookAt(controls.target);

  renderizador.render(escena, camara);
}

function detenerAnimacion() {
  if (idCuadroAnimacion !== null) {
    cancelAnimationFrame(idCuadroAnimacion);
    idCuadroAnimacion = null;
    console.log('[Miku-GUI] Modo reposo: loop de render pausado (ventana oculta/minimizada).');
  }
}

function reanudarAnimacion() {
  if (idCuadroAnimacion === null) {
    reloj.getDelta(); // descarta el tiempo acumulado en pausa (si no, el primer frame salta de golpe)
    animar();
    console.log('[Miku-GUI] Saliendo del modo reposo: loop de render reanudado.');
  }
}

// --- Aparicion/desaparicion del avatar (estilo widget, "Oye Miku") ----------------

const contenedorAvatar = document.getElementById('avatar-contenedor');
let temporizadorOcultarAvatar = null;

function cancelarOcultarAvatar() {
  if (temporizadorOcultarAvatar !== null) {
    clearTimeout(temporizadorOcultarAvatar);
    temporizadorOcultarAvatar = null;
  }
}

function mostrarAvatar() {
  cancelarOcultarAvatar();
  contenedorAvatar.classList.add('avatar-visible');
  reanudarAnimacion();
}

function ocultarAvatar() {
  cancelarOcultarAvatar();
  contenedorAvatar.classList.remove('avatar-visible');
  contenedorChatLog.classList.remove('chat-log-visible');
  botonToggleChatLog.classList.remove('chat-log-toggle-activo');
}

function programarOcultarAvatar(retrasoMs = RETRASO_OCULTAR_AVATAR_MS) {
  cancelarOcultarAvatar();
  temporizadorOcultarAvatar = setTimeout(() => {
    temporizadorOcultarAvatar = null;
    ocultarAvatar();
  }, retrasoMs);
}

// La Page Visibility API de Chromium refleja el estado real de la
// ventana de Electron (oculta con .hide(), minimizada, tapada por
// completo por otra ventana, etc.) sin necesitar IPC con el proceso
// principal.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    detenerAnimacion();
  } else {
    reanudarAnimacion();
  }
});

animar();
