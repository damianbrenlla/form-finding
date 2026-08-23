/**
 * DBSW 3D Form-Finding Engine — Canvas & Axis Triad Visualizer
 * Exact Industrial Visual Styling Matching SIMP Topology Tool
 */

let scene, camera, renderer, controls;
let networkGroup, reactionGroup, axisGroup;
let worker;

function initThreeJS() {
    const container = document.getElementById("canvas-container");
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xc8c8c8); // Technical off-white canvas

    camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 10, 200000);
    camera.position.set(12000, 12000, 9000);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Technical Lighting Setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight.position.set(10000, 15000, 20000);
    scene.add(dirLight);

    // Grid Floor
    const gridHelper = new THREE.GridHelper(30000, 60, 0x999999, 0xb0b0b0);
    gridHelper.rotation.x = Math.PI / 2;
    scene.add(gridHelper);

    networkGroup = new THREE.Group();
    reactionGroup = new THREE.Group();
    scene.add(networkGroup);
    scene.add(reactionGroup);

    createAxisTriad();

    window.addEventListener("resize", onWindowResize);
    animate();
}

function createAxisTriad() {
    axisGroup = new THREE.Group();
    
    const origin = new THREE.Vector3(0, 0, 0);
    const len = 800;
    
    // X - Red, Y - Green, Z - Black
    axisGroup.add(new THREE.ArrowHelper(new THREE.Vector3(1,0,0), origin, len, 0xd32f2f, 150, 50));
    axisGroup.add(new THREE.ArrowHelper(new THREE.Vector3(0,1,0), origin, len, 0x388e3c, 150, 50));
    axisGroup.add(new THREE.ArrowHelper(new THREE.Vector3(0,0,1), origin, len, 0x111111, 150, 50));

    scene.add(axisGroup);
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
    worker = new Worker(`worker_ff.js?v=${Date.now()}`);
    
    worker.onmessage = function(e) {
        const { status, data } = e.data;

        if (status === "completed") {
            renderFormFoundNetwork(data);
            displayReactions(data.reactions);
        }
    };

    worker.postMessage({ action: "init" });
}

function runFormFinding() {
    const payload = {
        preset: document.getElementById("preset").value,
        material_grade: document.getElementById("material_grade").value,
        support_preset: document.getElementById("support_preset").value,
        Lx: parseFloat(document.getElementById("Lx").value),
        Ly: parseFloat(document.getElementById("Ly").value),
        Lz: parseFloat(document.getElementById("Lz").value),
        nx: parseInt(document.getElementById("nx").value),
        ny: parseInt(document.getElementById("ny").value),
        prestress: parseFloat(document.getElementById("prestress").value),
        iterations: 400
    };

    worker.postMessage({ action: "form_find", payload });
}

function renderFormFoundNetwork(data) {
    while (networkGroup.children.length > 0) networkGroup.remove(networkGroup.children[0]);
    while (reactionGroup.children.length > 0) reactionGroup.remove(reactionGroup.children[0]);

    const { nodes, edges, axial_forces } = data;
    const minF = Math.min(...axial_forces);
    const maxF = Math.max(...axial_forces);

    const lineGeometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    edges.forEach((edge, idx) => {
        const p1 = nodes[edge[0]];
        const p2 = nodes[edge[1]];

        positions.push(...p1, ...p2);

        // Technical Line Color: Red for Tension, Dark Gray/Black for Compression
        const norm = (axial_forces[idx] - minF) / (maxF - minF || 1);
        const color = new THREE.Color();
        color.setHSL(0.0, norm, 0.2 + 0.3 * (1 - norm));

        colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    });

    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    lineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

    const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 1.5 });
    const networkMesh = new THREE.LineSegments(lineGeometry, lineMaterial);
    
    networkGroup.add(networkMesh);
}

function displayReactions(reactions) {
    const panel = document.getElementById("reaction-panel");
    const tbody = document.querySelector("#reaction-table tbody");
    tbody.innerHTML = "";
    panel.style.display = "block";

    reactions.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${r.node}</td><td>${r.Rz_kN}</td><td>${r.R_total_kN}</td>`;
        tbody.appendChild(tr);

        // Render Red Reaction Vector Pins
        const origin = new THREE.Vector3(...r.pos);
        const dir = new THREE.Vector3(r.Rx_kN, r.Ry_kN, r.Rz_kN).normalize();
        const length = 400;

        const arrow = new THREE.ArrowHelper(dir, origin, length, 0xd32f2f, 100, 40);
        reactionGroup.add(arrow);
    });
}

window.onload = function() {
    initThreeJS();
    initWorker();
};