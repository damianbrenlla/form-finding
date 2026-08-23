/**
 * DBSW R260003 Form-Finding Frontend Orchestrator & Three.js Renderer
 * Author: Damian Brenlla / DBSW 2026
 */

let scene, camera, renderer, controls;
let networkGroup, reactionGroup;
let worker;

function initThreeJS() {
    const container = document.getElementById("canvas-container");
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 10, 100000);
    camera.position.set(9000, 9000, 7000);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(10000, 20000, 10000);
    scene.add(dirLight);

    // Grid Helper
    const gridHelper = new THREE.GridHelper(20000, 20, 0x444444, 0x222222);
    gridHelper.rotation.x = Math.PI / 2;
    scene.add(gridHelper);

    networkGroup = new THREE.Group();
    reactionGroup = new THREE.Group();
    scene.add(networkGroup);
    scene.add(reactionGroup);

    window.addEventListener("resize", onWindowResize);
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById("canvas-container");
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function initWorker() {
    worker = new Worker("worker_ff.js");
    
    worker.onmessage = function(e) {
        const { status, message, data } = e.data;

        if (status === "log") {
            document.getElementById("status-bar").innerText = message;
        } else if (status === "ready") {
            document.getElementById("status-bar").innerText = "DBSW Form-Finding Engine Ready.";
        } else if (status === "completed") {
            document.getElementById("status-bar").innerText = `Form Found (${data.material})`;
            renderFormFoundNetwork(data);
            displayReactions(data.reactions);
        } else if (status === "error") {
            document.getElementById("status-bar").innerText = `Error: ${message}`;
        }
    };

    worker.postMessage({ action: "init" });
}

function runFormFinding() {
    document.getElementById("status-bar").innerText = "Solving Spatial Equilibrium...";
    
    const payload = {
        preset: document.getElementById("preset").value,
        material_grade: document.getElementById("material_grade").value,
        support_preset: document.getElementById("support_preset").value,
        Lx: parseFloat(document.getElementById("Lx").value),
        Ly: parseFloat(document.getElementById("Ly").value),
        Lz: parseFloat(document.getElementById("Lz").value),
        prestress: parseFloat(document.getElementById("prestress").value),
        iterations: 400
    };

    worker.postMessage({ action: "form_find", payload });
}

function renderFormFoundNetwork(data) {
    // Clear previous mesh
    while (networkGroup.children.length > 0) networkGroup.remove(networkGroup.children[0]);
    while (reactionGroup.children.length > 0) reactionGroup.remove(reactionGroup.children[0]);

    const { nodes, edges, axial_forces } = data;

    // Color gradient setup for tension/compression mapping
    const minF = Math.min(...axial_forces);
    const maxF = Math.max(...axial_forces);

    const lineGeometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    edges.forEach((edge, idx) => {
        const p1 = nodes[edge[0]];
        const p2 = nodes[edge[1]];

        positions.push(...p1, ...p2);

        // Normalize force color (Tension = Red/Orange, Compression = Blue)
        const norm = (axial_forces[idx] - minF) / (maxF - minF || 1);
        const color = new THREE.Color();
        color.setHSL(0.6 * (1 - norm), 1.0, 0.5);

        colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    });

    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    lineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

    const lineMaterial = new THREE.LineSegmentsMaterial ? 
        new THREE.LineSegmentsMaterial({ vertexColors: true }) : 
        new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });

    const networkMesh = new THREE.LineSegments(lineGeometry, lineMaterial);
    networkGroup.add(networkMesh);
}

function displayReactions(reactions) {
    const list = document.getElementById("reaction-list");
    list.innerHTML = "";

    reactions.forEach(r => {
        const item = document.createElement("li");
        item.innerText = `Node ${r.node}: Rz=${r.Rz_kN}kN | R_total=${r.R_total_kN}kN`;
        list.appendChild(item);

        // Render 3D Reaction Vector Arrows at Fixed Supports
        const origin = new THREE.Vector3(...r.pos);
        const dir = new THREE.Vector3(r.Rx_kN, r.Ry_kN, r.Rz_kN).normalize();
        const length = Math.min(Math.max(r.R_total_kN * 50, 300), 1500);

        const arrowHelper = new THREE.ArrowHelper(dir, origin, length, 0x00d2ff, 150, 80);
        reactionGroup.add(arrowHelper);
    });
}

// Initialize on page load
window.onload = function() {
    initThreeJS();
    initWorker();
};