<div align="center">

<img src="docs/miku.webp" alt="Miku en el escritorio: parpadea, respira y sigue el cursor con la mirada" width="72%">

<h1>Miku IA</h1>

<p><b>«Oye Miku».</b><br>
Un asistente de voz tipo Alexa con Hatsune Miku que vive en tu escritorio y funciona sin internet.</p>

<img src="https://img.shields.io/badge/100%25_local-sin_internet-39C5BB?style=for-the-badge&labelColor=0D1720" alt="100 % local, sin internet">
<img src="https://img.shields.io/badge/estado-en_pausa-E12885?style=for-the-badge&labelColor=0D1720" alt="Estado: en pausa">

<br><br>

<img src="https://img.shields.io/badge/Python-0D1720?style=flat-square&logo=python&logoColor=39C5BB" alt="Python">
<img src="https://img.shields.io/badge/llama.cpp-0D1720?style=flat-square&logo=meta&logoColor=39C5BB" alt="llama.cpp">
<img src="https://img.shields.io/badge/Ollama-0D1720?style=flat-square&logo=ollama&logoColor=white" alt="Ollama">
<img src="https://img.shields.io/badge/Whisper-0D1720?style=flat-square&logo=openai&logoColor=white" alt="Whisper">
<img src="https://img.shields.io/badge/Kokoro_TTS-0D1720?style=flat-square" alt="Kokoro TTS">
<img src="https://img.shields.io/badge/RVC-0D1720?style=flat-square" alt="RVC">
<img src="https://img.shields.io/badge/Electron-0D1720?style=flat-square&logo=electron&logoColor=9FEAF9" alt="Electron">
<img src="https://img.shields.io/badge/Three.js_·_VRM-0D1720?style=flat-square&logo=threedotjs&logoColor=white" alt="Three.js y VRM">

</div>

<br>

> La llamas, te escucha, te contesta con su voz y aparece en una esquina de la pantalla. Todo corre en tu propia computadora: nada sale a internet.

<br>

## ✦ Cómo piensa

```mermaid
flowchart LR
    A(["«Oye Miku»"]) --> B["Oídos<br/>Whisper"]
    B --> C["Cerebro<br/>Qwen 2.5 local"]
    C --> D["Voz<br/>Kokoro"]
    D --> E["Voz de Miku<br/>RVC"]
    E --> F(["Avatar 3D<br/>Electron + VRM"])
    C -.-> G[("Memoria<br/>Obsidian")]

    classDef miku fill:#39C5BB,stroke:#0D1720,stroke-width:2px,color:#0D1720
    classDef rosa fill:#E12885,stroke:#0D1720,stroke-width:2px,color:#ffffff
    classDef noche fill:#14353F,stroke:#39C5BB,stroke-width:2px,color:#ffffff
    class A,F rosa
    class B,C,D,E miku
    class G noche
```

<br>

## ✦ Lo que hace

<table>
  <tr>
    <td width="50%" valign="top"><b>Te escucha</b><br><sub>Se activa con «Oye Miku» y entiende lo que le dices, sin conexión.</sub></td>
    <td width="50%" valign="top"><b>Contesta rápido</b><br><sub>Empieza a hablar en cuanto tiene la primera oración, mientras piensa la siguiente.</sub></td>
  </tr>
  <tr>
    <td valign="top"><b>Con su propia voz</b><br><sub>Kokoro genera la voz y RVC la convierte en la de Miku.</sub></td>
    <td valign="top"><b>Vive en tu escritorio</b><br><sub>Un avatar 3D que parpadea, respira, te sigue con la mirada y mueve la boca al hablar.</sub></td>
  </tr>
  <tr>
    <td valign="top"><b>Recuerda</b><br><sub>Lleva un diario y guarda lo importante en su propia bóveda de Obsidian.</sub></td>
    <td valign="top"><b>Se hace a un lado</b><br><sub>«Miku duerme» o «modo juego» liberan la tarjeta de video para jugar.</sub></td>
  </tr>
  <tr>
    <td colspan="2" valign="top"><b>También por texto</b><br><sub>Una barra de chat para cuando no puedes hablar en voz alta.</sub></td>
  </tr>
</table>

<br>

## ✦ Estado

En pausa. Funciona de principio a fin, pero ocupa de 10 a 14 GB de VRAM y le falta optimizarse.

<br>

---

<div align="center">
<sub>Proyecto personal de <a href="https://carloramirez.dev"><b>Carlo Ramírez</b></a> · 2026</sub><br>
<sub>Hatsune Miku es un personaje de Crypton Future Media. Este es un proyecto de fan sin fines comerciales.</sub>
</div>
