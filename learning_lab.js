// AegisMARL: Neural Lab & Tab System Controller

let netCanvas, netCtx;
let networkPulseTime = 0;

// Tab Switcher Engine
function switchTab(tabId) {
    // Hide all tab panels
    const panels = document.querySelectorAll('.tab-panel');
    panels.forEach(panel => panel.classList.remove('active'));

    // Deactivate all navigation tabs
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => tab.classList.remove('active'));

    // Show selected panel
    document.getElementById(tabId).classList.add('active');

    // Highlight selected tab button
    if (tabId === 'tab-sim') {
        document.getElementById('btn-tab-sim').classList.add('active');
    } else if (tabId === 'tab-lab') {
        document.getElementById('btn-tab-lab').classList.add('active');
        // Lazy initialize the network drawing canvas
        setTimeout(initNetworkDiagram, 100);
    } else if (tabId === 'tab-study') {
        document.getElementById('btn-tab-study').classList.add('active');
    }
}

// --- 2. Interactive Bellman TD-Learning Sandbox ---
function updateBellmanMath() {
    const R = parseFloat(document.getElementById('slide-bellman-reward').value);
    const gamma = parseFloat(document.getElementById('slide-bellman-gamma').value);
    const alpha = parseFloat(document.getElementById('slide-bellman-alpha').value);

    // Update text labels
    document.getElementById('lbl-bellman-reward').innerText = (R >= 0 ? '+' : '') + R.toFixed(1);
    document.getElementById('lbl-bellman-gamma').innerText = gamma.toFixed(2);
    document.getElementById('lbl-bellman-alpha').innerText = alpha.toFixed(2);

    // Initial state values for math playground
    const vOld = 25.0;
    const vNext = 32.00;

    // Temporal Difference Equations
    // TD Target = R + gamma * V(S')
    const tdTarget = R + gamma * vNext;
    
    // TD Error (delta) = TD Target - V(S)
    const tdError = tdTarget - vOld;
    
    // V_new(S) = V(S) + alpha * TD Error
    const vNew = vOld + alpha * tdError;

    // Display updates in UI
    const deltaLbl = document.getElementById('lbl-calc-delta');
    deltaLbl.innerText = (tdError >= 0 ? '+' : '') + tdError.toFixed(2);
    
    // Toggle color depending on positive/negative TD error
    if (tdError >= 0) {
        deltaLbl.className = 'neon-text-teal';
    } else {
        deltaLbl.className = 'neon-text-pink';
    }

    document.getElementById('lbl-calc-vnew').innerText = vNew.toFixed(3);
}

// --- 3. Neural Network Interactive Topology Canvas ---
function initNetworkDiagram() {
    netCanvas = document.getElementById('network-canvas');
    if (!netCanvas) return;
    netCtx = netCanvas.getContext('2d');
    
    // Start drawing loop for the network diagram
    drawNetworkDiagram();
}

function drawNetworkDiagram() {
    if (!netCanvas || !document.getElementById('tab-lab').classList.contains('active')) return;

    netCtx.clearRect(0, 0, netCanvas.width, netCanvas.height);
    networkPulseTime += 0.02;

    const width = netCanvas.width;
    const height = netCanvas.height;

    // Define topology layout coordinates
    // Input layer (33 dims -> drawn as 6 key nodes representing categories)
    const inputNodes = [
        { label: 'Self HP %', x: 80, y: 70 },
        { label: 'Self Cooldown', x: 80, y: 130 },
        { label: 'Ally Knights Offsets', x: 80, y: 190 },
        { label: 'Ally Cleric Offsets', x: 80, y: 250 },
        { label: 'Goblin Target Dist', x: 80, y: 310 },
        { label: 'Goblin Active Pct', x: 80, y: 370 }
    ];

    // Hidden Layer 1 (64 dims -> drawn as 7 nodes)
    const hidden1 = [];
    const h1Count = 7;
    for (let i = 0; i < h1Count; i++) {
        hidden1.push({ x: 230, y: 60 + i * 48 });
    }

    // Hidden Layer 2 (64 dims -> drawn as 7 nodes)
    const hidden2 = [];
    const h2Count = 7;
    for (let i = 0; i < h2Count; i++) {
        hidden2.push({ x: 370, y: 60 + i * 48 });
    }

    // Output Layer (6 Actions -> drawn as 6 nodes)
    const outputs = [
        { label: 'Move North', x: 500, y: 70 },
        { label: 'Move South', x: 500, y: 130 },
        { label: 'Move East', x: 500, y: 190 },
        { label: 'Move West', x: 500, y: 250 },
        { label: 'Execute Skill', x: 500, y: 310 },
        { label: 'Idle / Wait', x: 500, y: 370 }
    ];

    // 1. Draw Synapse Connections (Lines)
    netCtx.lineWidth = 0.5;

    // Inputs -> H1
    inputNodes.forEach(iNode => {
        hidden1.forEach(hNode => {
            const grad = netCtx.createLinearGradient(iNode.x, iNode.y, hNode.x, hNode.y);
            grad.addColorStop(0, 'rgba(0, 255, 204, 0.04)');
            grad.addColorStop(1, 'rgba(59, 130, 246, 0.04)');
            netCtx.strokeStyle = grad;
            netCtx.beginPath();
            netCtx.moveTo(iNode.x, iNode.y);
            netCtx.lineTo(hNode.x, hNode.y);
            netCtx.stroke();
        });
    });

    // H1 -> H2
    hidden1.forEach(h1Node => {
        hidden2.forEach(h2Node => {
            const grad = netCtx.createLinearGradient(h1Node.x, h1Node.y, h2Node.x, h2Node.y);
            grad.addColorStop(0, 'rgba(59, 130, 246, 0.04)');
            grad.addColorStop(1, 'rgba(16, 185, 129, 0.04)');
            netCtx.strokeStyle = grad;
            netCtx.beginPath();
            netCtx.moveTo(h1Node.x, h1Node.y);
            netCtx.lineTo(h2Node.x, h2Node.y);
            netCtx.stroke();
        });
    });

    // H2 -> Outputs
    hidden2.forEach(hNode => {
        outputs.forEach(oNode => {
            const grad = netCtx.createLinearGradient(hNode.x, hNode.y, oNode.x, oNode.y);
            grad.addColorStop(0, 'rgba(16, 185, 129, 0.04)');
            grad.addColorStop(1, 'rgba(0, 255, 204, 0.04)');
            netCtx.strokeStyle = grad;
            netCtx.beginPath();
            netCtx.moveTo(hNode.x, hNode.y);
            netCtx.lineTo(oNode.x, oNode.y);
            netCtx.stroke();
        });
    });

    // 2. Draw Moving Pulses (Signal propagation simulation)
    netCtx.fillStyle = 'rgba(0, 255, 204, 0.2)';
    inputNodes.forEach((iNode, idx) => {
        const hNode = hidden1[(idx + Math.floor(networkPulseTime)) % h1Count];
        const progress = (networkPulseTime % 1.0);
        const pulseX = iNode.x + (hNode.x - iNode.x) * progress;
        const pulseY = iNode.y + (hNode.y - iNode.y) * progress;
        
        netCtx.beginPath();
        netCtx.arc(pulseX, pulseY, 2, 0, Math.PI * 2);
        netCtx.fill();
    });

    hidden2.forEach((hNode, idx) => {
        const oNode = outputs[(idx + Math.floor(networkPulseTime * 1.5)) % outputs.length];
        const progress = ((networkPulseTime * 1.3) % 1.0);
        const pulseX = hNode.x + (oNode.x - hNode.x) * progress;
        const pulseY = hNode.y + (oNode.y - hNode.y) * progress;

        netCtx.fillStyle = 'rgba(16, 185, 129, 0.2)';
        netCtx.beginPath();
        netCtx.arc(pulseX, pulseY, 2, 0, Math.PI * 2);
        netCtx.fill();
    });

    // 3. Draw Nodes (Input Layer)
    inputNodes.forEach(node => {
        netCtx.fillStyle = 'var(--bg-card-glass)';
        netCtx.strokeStyle = 'var(--cyber-teal)';
        netCtx.lineWidth = 1.5;
        
        netCtx.beginPath();
        netCtx.arc(node.x, node.y, 10, 0, Math.PI * 2);
        netCtx.fill();
        netCtx.stroke();
        
        // Inner glowing core
        netCtx.fillStyle = 'rgba(0, 255, 204, 0.2)';
        netCtx.beginPath();
        netCtx.arc(node.x, node.y, 4, 0, Math.PI * 2);
        netCtx.fill();

        // Node labels
        netCtx.fillStyle = 'var(--text-muted)';
        netCtx.font = '9px Outfit';
        netCtx.textAlign = 'right';
        netCtx.fillText(node.label, node.x - 18, node.y + 3);
    });

    // H1 Hidden Layer Nodes
    hidden1.forEach(node => {
        netCtx.fillStyle = 'rgba(10, 16, 32, 0.8)';
        netCtx.strokeStyle = 'rgba(59, 130, 246, 0.4)';
        netCtx.lineWidth = 1;
        netCtx.beginPath();
        netCtx.arc(node.x, node.y, 6, 0, Math.PI * 2);
        netCtx.fill();
        netCtx.stroke();

        netCtx.fillStyle = 'rgba(59, 130, 246, 0.3)';
        netCtx.beginPath();
        netCtx.arc(node.x, node.y, 2, 0, Math.PI * 2);
        netCtx.fill();
    });

    // H2 Hidden Layer Nodes
    hidden2.forEach(node => {
        netCtx.fillStyle = 'rgba(10, 16, 32, 0.8)';
        netCtx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
        netCtx.lineWidth = 1;
        netCtx.beginPath();
        netCtx.arc(node.x, node.y, 6, 0, Math.PI * 2);
        netCtx.fill();
        netCtx.stroke();

        netCtx.fillStyle = 'rgba(16, 185, 129, 0.3)';
        netCtx.beginPath();
        netCtx.arc(node.x, node.y, 2, 0, Math.PI * 2);
        netCtx.fill();
    });

    // Output Layer Nodes
    outputs.forEach((node, idx) => {
        netCtx.fillStyle = 'var(--bg-card-glass)';
        
        // Highlight active output node visually based on time pulses
        const isActive = idx === Math.floor(networkPulseTime * 0.7) % outputs.length;
        netCtx.strokeStyle = isActive ? 'var(--cyber-green)' : 'rgba(0, 255, 204, 0.3)';
        netCtx.lineWidth = isActive ? 2 : 1.2;

        netCtx.beginPath();
        netCtx.arc(node.x, node.y, 10, 0, Math.PI * 2);
        netCtx.fill();
        netCtx.stroke();

        netCtx.fillStyle = isActive ? 'var(--cyber-green)' : 'rgba(0, 255, 204, 0.1)';
        netCtx.beginPath();
        netCtx.arc(node.x, node.y, isActive ? 5 : 3, 0, Math.PI * 2);
        netCtx.fill();

        // Node labels
        netCtx.fillStyle = isActive ? '#fff' : 'var(--text-muted)';
        netCtx.font = isActive ? 'bold 10px Outfit' : '9px Outfit';
        netCtx.textAlign = 'left';
        netCtx.fillText(node.label, node.x + 18, node.y + 3);
    });

    // Trigger next frame
    requestAnimationFrame(drawNetworkDiagram);
}

// Initial update of sandbox computational result
updateBellmanMath();
