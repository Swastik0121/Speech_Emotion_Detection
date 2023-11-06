import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";

import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";

import { GLTFLoader } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/loaders/GLTFLoader.js";


const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);


let mouseX = window.innerWidth / 2;
let mouseY = window.innerHeight / 2;


let object;


let controls;


let objToRender = 'late';


const loader = new GLTFLoader();


loader.load(
  `/static/models/${objToRender}/scene.gltf`,
  function (gltf) {
    object = gltf.scene;
    object.rotation.set(0, 0, 0);
    object.scale.set(3, 3, 3);
    
    scene.add(object);
  },
  function (xhr) {
    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
  },
  function (error) {
    console.error(error);
  }
);

camera.focus = 5; 
camera.aperture = 0.005; 


const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);

document.getElementById("container3d").appendChild(renderer.domElement);

camera.position.z = objToRender === "late" ? 25 : 10;

const topLight = new THREE.DirectionalLight(0xffffff, 1); 
topLight.position.set(500, 500, 500) 
topLight.castShadow = true;
scene.add(topLight);

const ambientLight = new THREE.AmbientLight(0x333333, objToRender === "late" ? 5 : 1);
scene.add(ambientLight);



if (objToRender === "late") {
  controls = new OrbitControls(camera, renderer.domElement);
}

function animate() {
  requestAnimationFrame(animate);

  if (object && objToRender === "late") {
    object.rotation.y = -1 + mouseX / window.innerWidth * 1;
    
  }
  renderer.render(scene, camera);
}

window.addEventListener("resize", function () {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

document.onmousemove = (e) => {
  mouseX = e.clientX;
  mouseY = e.clientY;
}

animate();


document.getElementById("listenButton").addEventListener("click", function () {
  var outputElement = document.getElementById("output");
  outputElement.textContent = "Listening...";
});
