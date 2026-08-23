/**
 * DBSW 3D Form-Finding Engine — Canvas & UI Integration
 * Bridge WebWorker status logs to industrial sidebar layout
 */

let scene, camera, renderer, controls;
let networkGroup, reactionGroup, axisGroup;
let worker;
let engineReady = false;

function initThreeJS() {
    const container = document.getElementById("viewport");
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff); // White canvas matching SIMP tool

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 100000);
    camera.position.set(9000, 6000, 7000);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Technical Lighting Setup
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.75);
    keyLight.position.set(5000, 8000, 4000);
    scene.add(keyLight);

    // Subtle Ground Grid
    const gridHelper = new THREE.GridHelper(20000, 200, 0x000000, 0xe0e0e0);
    gridHelper.position.y = -1;
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
    const container = document.getElementById("viewport");
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function initWasmEngine() {
    worker = new Worker(`worker_ff.js?cb=${Date.now()}`);
    worker.onmessage = handleWorkerMessage;
    worker.postMessage({ action: "init" });
}

function handleWorkerMessage(e) {
    const data = e.data;
    const status = document.getElementById("status");
    const btnExec = document.getElementById("btn-exec");
    const btnStop = document.getElementById("btn-stop");
    const actionsGroup = document.getElementById("results-actions-group");

    if (data.status === "log") {
        status.style.color = "#000000";
        status.style.borderLeftColor = "#000000";
        status.innerText = data.message;

    } else if (data.status === "ready") {
        engineReady = true;
        status.style.color = "#000000";
        status.style.borderLeftColor = "#000000";
        status.innerText = "Wasm Engine Ready. Define problem parameters.";
        btnExec.disabled = false;
        btnExec.innerText = "Execute Form-Finding Solver";

    } else if (data.status === "completed") {
        btnExec.disabled = false;
        btnExec.innerText = "Execute Form-Finding Solver";
        btnStop.style.display = "none";
        actionsGroup.style.display = "flex";

        status.style.color = "#000000";
        status.style.borderLeftColor = "#000000";
        status.innerText = `Form found successfully [${data.data.material}].`;
        
        renderFormFoundNetwork(data.data);
        displayReactions(data.data.reactions);

    } else if (data.status === "error") {
        btnExec.disabled = false;
        btnExec.innerText = "Execute Form-Finding Solver";
        btnStop.style.display = "none";
        status.style.color = "#cc0000";
        status.style.borderLeftColor = "#cc0000";
        status.innerText = "Solver Error: " + data.message;
    }
}

function startOptimisation() {
    if (!engineReady) return;
    
    const status = document.getElementById("status");
    status.style.color = "#000000";
    status.style.borderLeftColor = "#000000";
    status.innerText = "Solving Spatial Equilibrium...";

    document.getElementById("btn-exec").disabled = true;
    document.getElementById("btn-stop").style.display = "block";
    document.getElementById("results-actions-group").style.display = "none";

    const payload = {
        preset: document.getElementById("preset").value,
        material_grade: document.getElementById("mat_grade").value,
        support_preset: document.getElementById("support_preset").value,
        Lx: parseFloat(document.getElementById("Lx").value),
        Ly: parseFloat(document.getElementById("Ly").value),
        Lz: parseFloat(document.getElementById("Lz").value),
        nx: parseInt(document.getElementById("nx").value),
        ny: parseInt(document.getElementById("ny").value),
        prestress: parseFloat(document.getElementById("prestress").value),
        iterations: parseInt(document.getElementById("iterations").value)
    };

    worker.postMessage({ action: "form_find", payload });
}

function stopOptimisation() {
    const status = document.getElementById("status");
    status.style.color = "#cc0000";
    status.style.borderLeftColor = "#cc0000";
    status.innerText = "Terminating and reloading Wasm engine...";

    document.getElementById("btn-stop").style.display = "none";
    document.getElementById("btn-exec").disabled = true;
    document.getElementById("btn-exec").innerText = "Loading Wasm Engine…";
    document.getElementById("results-actions-group").style.display = "none";
    engineReady = false;

    if (worker) worker.terminate();
    initWasmEngine();
}

function clearResults() {
    while (networkGroup.children.length > 0) networkGroup.remove(networkGroup.children[0]);
    while (reactionGroup.children.length > 0) reactionGroup.remove(reactionGroup.children[0]);

    document.getElementById("results-actions-group").style.display = "none";
    const status = document.getElementById("status");
    status.style.color = "#000000";
    status.style.borderLeftColor = "#000000";
    status.innerText = "Results cleared. Ready to define parameters.";
}

function renderFormFoundNetwork(data) {
    while (networkGroup.children.length > 0) networkGroup.remove(networkGroup.children[0]);

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
    while (reactionGroup.children.length > 0) reactionGroup.remove(reactionGroup.children[0]);

    reactions.forEach(r => {
        const origin = new THREE.Vector3(...r.pos);
        const dir = new THREE.Vector3(r.Rx_kN, r.Ry_kN, r.Rz_kN).normalize();
        const length = 400;

        const arrow = new THREE.ArrowHelper(dir, origin, length, 0xd32f2f, 100, 40);
        reactionGroup.add(arrow);
    });
}

function updateDomainVisuals() {
    // Visual bounding box update for preset changes
}

function exportSTL() {
    alert("Exporting form-found mesh as .STL...");
}

window.onload = function() {
    initThreeJS();
    initWasmEngine();
};
