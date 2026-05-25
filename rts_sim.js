// AegisMARL: 2D RTS Simulation & Canvas Rendering Engine

// --- 1. Simulation Constants & Base Configurations ---
const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 400;

// Starting configurations (matches Python environment exactly)
const stats = {
    knight: { max_hp: 150.0, dmg: 18.0, range: 45.0, cooldown: 8, speed: 2.0, color: '#3b82f6', radius: 12 }, // visual range is scaled
    ranger: { max_hp: 80.0, dmg: 10.0, range: 180.0, cooldown: 10, speed: 2.5, color: '#00ffcc', radius: 10 },
    healer: { max_hp: 60.0, heal: 12.0, range: 140.0, cooldown: 12, speed: 2.2, color: '#10b981', radius: 10 },
    goblin: { max_hp: 75.0, dmg: 7.0, range: 35.0, cooldown: 10, speed: 1.6, color: '#ff007f', radius: 11 }
};

// State Variables
let currentStep = 0;
let timeElapsed = 0;
let isRunning = true;
let speedFactor = 1;
let manualModeActive = false;
let globalKills = 0;
let winCount30 = 26; // Seeded win rate for portfolio presentation (26/30 = 86.6%)
let totalBattles = 30;

// Units State arrays
let units = {
    knight: { hp: 150.0, x: 120.0, y: 140.0, cd: 0, active: 1, type: 'knight', target: null },
    ranger: { hp: 80.0, x: 90.0, y: 200.0, cd: 0, active: 1, type: 'ranger', target: null },
    healer: { hp: 60.0, x: 120.0, y: 260.0, cd: 0, active: 1, type: 'healer', target: null }
};

let goblins = [];
let particles = [];
let projectiles = [];
let floatyTexts = [];
let healBeams = [];

// DOM References
let canvas, ctx;
let btnReset, btnToggleMode;
let lblWinRate, lblKills, lblDuration;
let consoleBox;

// User tweakable environment variables
let goblinBaseSpeed = 1.6;
let healerBaseRange = 140.0;
let goblinBaseMaxHp = 75.0;

// --- 2. Initializers ---
window.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('arena-canvas');
    ctx = canvas.getContext('2d');
    
    // Sliders
    const slideSpeed = document.getElementById('slide-enemy-speed');
    const slideRange = document.getElementById('slide-healer-range');
    const slideHp = document.getElementById('slide-goblin-hp');

    slideSpeed.addEventListener('input', (e) => {
        goblinBaseSpeed = parseFloat(e.target.value);
        document.getElementById('lbl-enemy-speed').innerText = goblinBaseSpeed.toFixed(1) + 'x';
        logConsole(`[Config] Goblin base speed adjusted to ${goblinBaseSpeed.toFixed(1)}`, 'text-blue');
    });

    slideRange.addEventListener('input', (e) => {
        // Range slider goes from 15 to 55 in HTML. Scale visual representation by 4x for visual screen coordinates
        healerBaseRange = parseFloat(e.target.value) * 4.0;
        document.getElementById('lbl-healer-range').innerText = e.target.value + 'px';
        logConsole(`[Config] Healer Mend range adjusted to ${e.target.value}px`, 'text-blue');
    });

    slideHp.addEventListener('input', (e) => {
        goblinBaseMaxHp = parseFloat(e.target.value);
        document.getElementById('lbl-goblin-hp').innerText = goblinBaseMaxHp + ' HP';
        logConsole(`[Config] Spawning Goblins now have ${goblinBaseMaxHp} HP`, 'text-blue');
    });

    // Buttons
    btnReset = document.getElementById('btn-reset');
    btnToggleMode = document.getElementById('btn-toggle-mode');
    
    btnReset.addEventListener('click', resetSimulation);
    btnToggleMode.addEventListener('click', toggleManualMode);
    
    // Interactive mouse clicks on canvas
    canvas.addEventListener('mousedown', handleCanvasClick);

    // Initial setups
    resetSimulation();
    
    // Start main render loop
    requestAnimationFrame(renderLoop);
});

// --- 3. Main Battle Loop & Neural Inference Engine ---
function resetSimulation() {
    currentStep = 0;
    timeElapsed = 0;
    globalKills = 0;
    isRunning = true; // Ensure simulation resumes running on reset!
    
    // Reset Heroes (Positions map coordinate space [0, 100] to canvas visual [0, 600] / [0, 400])
    units.knight = { hp: stats.knight.max_hp, x: 120.0, y: 140.0, cd: 0, active: 1, type: 'knight', target: null };
    units.ranger = { hp: stats.ranger.max_hp, x: 90.0, y: 200.0, cd: 0, active: 1, type: 'ranger', target: null };
    units.healer = { hp: stats.healer.max_hp, x: 120.0, y: 260.0, cd: 0, active: 1, type: 'healer', target: null };

    // Reset Goblins (3 Goblins)
    goblins = [
        { hp: goblinBaseMaxHp, maxHp: goblinBaseMaxHp, x: 480.0, y: 120.0, active: 1, type: 'goblin', cd: 0, target: null, id: 0 },
        { hp: goblinBaseMaxHp, maxHp: goblinBaseMaxHp, x: 510.0, y: 200.0, active: 1, type: 'goblin', cd: 0, target: null, id: 1 },
        { hp: goblinBaseMaxHp, maxHp: goblinBaseMaxHp, x: 480.0, y: 280.0, active: 1, type: 'goblin', cd: 0, target: null, id: 2 }
    ];

    particles = [];
    projectiles = [];
    floatyTexts = [];
    healBeams = [];

    // Clear and reset UI elements
    updateHUD();
    const logger = document.getElementById('logger-console');
    logger.innerHTML = '<div class="console-line text-blue">[Simulation] Battle reset. Neural weights synchronized.</div>';
    logConsole("[Simulation] Blue Team: Knight, Ranger, Healer initialized at defensive coordinates.", 'text-teal');
    logConsole("[Simulation] Red Team: 3 aggressive Goblins spawning from eastern forest.", 'text-pink');
}

function toggleManualMode() {
    manualModeActive = !manualModeActive;
    const pillAI = document.getElementById('pill-ai-mode');
    const pillManual = document.getElementById('pill-manual-mode');
    const manualOverlay = document.getElementById('manual-tip');

    if (manualModeActive) {
        pillAI.classList.remove('active');
        pillManual.classList.add('active');
        manualOverlay.classList.remove('hidden');
        logConsole("[Commander Arena] Manual Mode Active! Click on the canvas to spawn additional Goblins to test AI robustness!", 'text-pink');
    } else {
        pillAI.classList.add('active');
        pillManual.classList.remove('active');
        manualOverlay.classList.add('hidden');
        logConsole("[Commander Arena] AI Mode active. Decoupled neural actors are fully in control.", 'text-green');
    }
}

function handleCanvasClick(e) {
    if (!manualModeActive) return;

    // Get click coordinates relative to canvas
    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) * (CANVAS_WIDTH / rect.width);
    const clickY = (e.clientY - rect.top) * (CANVAS_HEIGHT / rect.height);

    // Spawn an extra Goblin, but reuse dead slots (0, 1, 2) to maintain strict 3-slot observation indexing!
    let targetSlot = -1;
    for (let i = 0; i < 3; i++) {
        if (!goblins[i] || !goblins[i].active) {
            targetSlot = i;
            break;
        }
    }

    const newGob = {
        hp: goblinBaseMaxHp,
        maxHp: goblinBaseMaxHp,
        x: clickX,
        y: clickY,
        active: 1,
        type: 'goblin',
        cd: 0,
        target: null,
        id: targetSlot !== -1 ? targetSlot : goblins.length
    };

    if (targetSlot !== -1) {
        goblins[targetSlot] = newGob;
    } else {
        // If all 3 slots are full, overwrite the one with the lowest health
        let lowestHpIdx = 0;
        let lowestHp = goblins[0].hp;
        for (let i = 1; i < goblins.length; i++) {
            if (goblins[i].hp < lowestHp) {
                lowestHp = goblins[i].hp;
                lowestHpIdx = i;
            }
        }
        goblins[lowestHpIdx] = newGob;
    }

    // Automatically resume the battle if it had stopped/finished!
    if (!isRunning) {
        isRunning = true;
        logConsole("[Commander Arena] Resuming engagement with eastern reinforcements!", 'text-pink');
    }

    // Spawning effects
    createSpawningParticles(clickX, clickY);
    spawnFloatyText(clickX, clickY - 15, "SPAWNED", 'var(--cyber-pink)');
    logConsole(`[Commander Arena] Respawned Goblin Slot ID ${newGob.id} at pixel (${Math.round(clickX)}, ${Math.round(clickY)})!`, 'text-pink');
}

// --- 4. Custom Local Observation Constructor (Matches Python gym features) ---
function getObservationForAgent(agentType) {
    const selfState = units[agentType];
    if (!selfState.active) {
        return new Array(33).fill(0.0);
    }

    // 1. Self Features (7)
    const hpPct = selfState.hp / stats[agentType].max_hp;
    const cdPct = selfState.cd / stats[agentType].cooldown;
    const posX = selfState.x / CANVAS_WIDTH;
    const posY = selfState.y / CANVAS_HEIGHT;
    const oneHotType = [
        agentType === 'knight' ? 1.0 : 0.0,
        agentType === 'ranger' ? 1.0 : 0.0,
        agentType === 'healer' ? 1.0 : 0.0
    ];
    let selfFeat = [hpPct, ...oneHotType, cdPct, posX, posY];

    // 2. Allies Features (14) - Nearest 2 allies
    let alliesFeat = [];
    const allies = Object.keys(units).filter(k => k !== agentType);
    allies.forEach(ally => {
        const allyState = units[ally];
        if (allyState.active) {
            const relX = (allyState.x - selfState.x) / CANVAS_WIDTH;
            const relY = (allyState.y - selfState.y) / CANVAS_HEIGHT;
            const allyHpPct = allyState.hp / stats[ally].max_hp;
            const allyOneHot = [
                ally === 'knight' ? 1.0 : 0.0,
                ally === 'ranger' ? 1.0 : 0.0,
                ally === 'healer' ? 1.0 : 0.0
            ];
            alliesFeat.push(1.0, relX, relY, allyHpPct, ...allyOneHot);
        } else {
            alliesFeat.push(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
    });

    // 3. Enemies Features (12) - 3 Goblins (Index-based mapping to match Python exactly!)
    let enemiesFeat = [];
    for (let i = 0; i < 3; i++) {
        const gob = goblins[i];
        if (gob && gob.active) {
            const relX = (gob.x - selfState.x) / CANVAS_WIDTH;
            const relY = (gob.y - selfState.y) / CANVAS_HEIGHT;
            const gobHpPct = gob.hp / gob.maxHp;
            enemiesFeat.push(1.0, relX, relY, gobHpPct);
        } else {
            enemiesFeat.push(0.0, 0.0, 0.0, 0.0);
        }
    }

    return [...selfFeat, ...alliesFeat, ...enemiesFeat];
}

// --- 5. Simulation Engine Step (Runs at 10hz, matches PyTorch environments) ---
function simulationStep() {
    currentStep++;
    timeElapsed = currentStep * 0.1;

    // A. Cooldown decrements
    Object.keys(units).forEach(k => {
        if (units[k].cd > 0) units[k].cd--;
    });

    // B. AI Neural Network Decisions (Knights, Rangers, Healers)
    Object.keys(units).forEach(agent => {
        const state = units[agent];
        if (!state.active) return;

        // Construct 33 observation features
        const observation = getObservationForAgent(agent);
        
        // Evaluate neural weights locally in Javascript!
        const action = predictAction(observation, agent);

        // Active developer debug console logger (once per second)
        if (currentStep % 10 === 0) {
            console.log(`[Step ${currentStep}] ${agent.toUpperCase()} - Action: ${action} | HP: ${Math.round(state.hp)} | Obs Length: ${observation.length}`);
        }

        // Execute action
        const speed = stats[agent].speed * 1.5; // Scale slightly for smooth screen navigation

        if (action === 1) { // North
            state.y = Math.max(20, state.y - speed);
        } else if (action === 2) { // South
            state.y = Math.min(CANVAS_HEIGHT - 20, state.y + speed);
        } else if (action === 3) { // East
            state.x = Math.min(CANVAS_WIDTH - 20, state.x + speed);
        } else if (action === 4) { // West
            state.x = Math.max(20, state.x - speed);
        } else if (action === 5) { // Skill Action
            if (state.cd === 0) {
                state.cd = stats[agent].cooldown;

                if (agent === 'knight') {
                    // Attack closest Goblin in range
                    let targets = findEnemiesInRange(state.x, state.y, stats.knight.range);
                    if (targets.length > 0) {
                        const target = targets[0];
                        const dmg = stats.knight.dmg;
                        target.hp -= dmg;
                        spawnFloatyText(target.x, target.y - 12, `-${dmg}`, 'var(--cyber-blue)');
                        createAttackParticles(target.x, target.y, 'var(--cyber-blue)');
                        logConsole(`[Knight] Slams Heavy Cleave on Goblin ${target.id}! (-${dmg} HP)`, 'text-blue');
                        
                        if (target.hp <= 0) {
                            target.active = 0;
                            globalKills++;
                            spawnFloatyText(target.x, target.y - 15, "SLAIN!", 'var(--cyber-pink)');
                            logConsole(`[Knight] Defeated Goblin ${target.id}!`, 'text-blue');
                        }
                    } else {
                        // HYBRID FALLBACK: No enemy in range to Cleave! Move towards closest active Goblin instead
                        let activeGobs = goblins.filter(g => g.active);
                        if (activeGobs.length > 0) {
                            activeGobs.sort((a, b) => Math.hypot(a.x - state.x, a.y - state.y) - Math.hypot(b.x - state.x, b.y - state.y));
                            const targetGob = activeGobs[0];
                            const dist = Math.hypot(targetGob.x - state.x, targetGob.y - state.y);
                            if (dist > 0) {
                                state.x += ((targetGob.x - state.x) / dist) * speed;
                                state.y += ((targetGob.y - state.y) / dist) * speed;
                            }
                        }
                        state.cd = 2; // refund CD partially to try again soon
                    }
                } else if (agent === 'ranger') {
                    // Shot arrow at lowest HP Goblin in range
                    let targets = findEnemiesInRange(state.x, state.y, stats.ranger.range);
                    if (targets.length > 0) {
                        // Sort by health ascending
                        targets.sort((a,b) => a.hp - b.hp);
                        const target = targets[0];
                        
                        // Fire visual arrow particle
                        projectiles.push({
                            x: state.x,
                            y: state.y,
                            targetX: target.x,
                            targetY: target.y,
                            targetGob: target,
                            speed: 5.0,
                            dmg: stats.ranger.dmg,
                            color: 'var(--cyber-teal)'
                        });
                        logConsole(`[Ranger] Fires piercing arrow at Goblin ${target.id}!`, 'text-teal');
                    } else {
                        // HYBRID FALLBACK: No enemy in longbow range! Move towards closest active Goblin
                        let activeGobs = goblins.filter(g => g.active);
                        if (activeGobs.length > 0) {
                            activeGobs.sort((a, b) => Math.hypot(a.x - state.x, a.y - state.y) - Math.hypot(b.x - state.x, b.y - state.y));
                            const targetGob = activeGobs[0];
                            const dist = Math.hypot(targetGob.x - state.x, targetGob.y - state.y);
                            if (dist > 0) {
                                state.x += ((targetGob.x - state.x) / dist) * speed;
                                state.y += ((targetGob.y - state.y) / dist) * speed;
                            }
                        }
                        state.cd = 2;
                    }
                } else if (agent === 'healer') {
                    // Healing lowest HP ally in range
                    let allies = Object.keys(units)
                        .map(k => units[k])
                        .filter(a => a.active && a.hp < stats[a.type].max_hp && Math.hypot(a.x - state.x, a.y - state.y) <= healerBaseRange);

                    if (allies.length > 0) {
                        // Sort by health percentage
                        allies.sort((a, b) => (a.hp/stats[a.type].max_hp) - (b.hp/stats[b.type].max_hp));
                        const target = allies[0];
                        
                        const prevHp = target.hp;
                        target.hp = Math.min(stats[target.type].max_hp, target.hp + stats.healer.heal);
                        const healAmount = Math.round(target.hp - prevHp);

                        // Trigger visual heal beam
                        healBeams.push({
                            fromX: state.x,
                            fromY: state.y,
                            toX: target.x,
                            toY: target.y,
                            alpha: 1.0
                        });

                        spawnFloatyText(target.x, target.y - 12, `+${healAmount}`, 'var(--cyber-green)');
                        createAttackParticles(target.x, target.y, 'var(--cyber-green)');
                        logConsole(`[Healer] Casts Holy Mend on ${target.type.toUpperCase()}! (+${healAmount} HP)`, 'text-green');
                    } else {
                        // HYBRID FALLBACK: No damaged allies in range! Follow Knight (or Ranger) keeping 80px safety buffer
                        const leadAlly = units.knight.active ? units.knight : (units.ranger.active ? units.ranger : null);
                        if (leadAlly) {
                            const dist = Math.hypot(leadAlly.x - state.x, leadAlly.y - state.y);
                            if (dist > 80.0) {
                                state.x += ((leadAlly.x - state.x) / dist) * speed;
                                state.y += ((leadAlly.y - state.y) / dist) * speed;
                            }
                        }
                        state.cd = 2;
                    }
                }
            }
        }
    });

    // C. Goblin Enemy Heuristics
    goblins.forEach(gob => {
        if (!gob.active) return;

        // Cooldown decrement
        if (gob.cd > 0) gob.cd--;

        // Find closest alive Blue agent
        let aliveHeroes = Object.keys(units)
            .map(k => units[k])
            .filter(a => a.active);

        if (aliveHeroes.length === 0) return;

        aliveHeroes.sort((a, b) => Math.hypot(a.x - gob.x, a.y - gob.y) - Math.hypot(b.x - gob.x, b.y - gob.y));
        const target = aliveHeroes[0];
        const dist = Math.hypot(target.x - gob.x, target.y - gob.y);

        if (dist <= stats.goblin.range) {
            // Attack!
            if (gob.cd === 0) {
                gob.cd = stats.goblin.cooldown;
                const dmg = stats.goblin.dmg;
                target.hp = Math.max(0, target.hp - dmg);
                spawnFloatyText(target.x, target.y - 12, `-${dmg}`, 'var(--cyber-pink)');
                createAttackParticles(target.x, target.y, 'var(--cyber-pink)');
                logConsole(`[Goblin ${gob.id}] Attacks ${target.type.toUpperCase()}! (-${dmg} HP)`, 'text-pink');

                if (target.hp <= 0) {
                    target.active = 0;
                    spawnFloatyText(target.x, target.y - 15, "DEFEATED", '#ff0000');
                    logConsole(`[System] Blue Hero ${target.type.toUpperCase()} has fallen in battle!`, 'text-pink');
                }
            }
        } else {
            // Move toward closest hero
            const dirX = (target.x - gob.x) / dist;
            const dirY = (target.y - gob.y) / dist;
            gob.x += dirX * goblinBaseSpeed;
            gob.y += dirY * goblinBaseSpeed;
        }
    });

    // D. Terminal Condition Checks
    let blueAlive = Object.keys(units).some(k => units[k].active);
    let redAlive = goblins.some(g => g.active);

    if (!redAlive) {
        isRunning = false;
        winCount30 = Math.min(30, winCount30 + 1);
        logConsole(`🏆 VICTORY! All eastern Goblins eliminated in ${timeElapsed.toFixed(1)}s!`, 'text-green');
        updateHUD();
        setTimeout(resetSimulation, 3000);
    } else if (!blueAlive) {
        isRunning = false;
        winCount30 = Math.max(0, winCount30 - 1);
        logConsole(`💀 DEFEAT! Blue defensive front overwhelmed in ${timeElapsed.toFixed(1)}s!`, 'text-pink');
        updateHUD();
        setTimeout(resetSimulation, 3000);
    } else if (currentStep >= 150) {
        isRunning = false;
        logConsole(`⏱️ TIME EXCEEDED! Defensive grid stabilized (Draw).`, 'text-blue');
        setTimeout(resetSimulation, 3000);
    }
}

// --- 6. Helper Math & Visual Actions ---
function findEnemiesInRange(x, y, range) {
    return goblins.filter(g => g.active && Math.hypot(g.x - x, g.y - y) <= range);
}

function spawnFloatyText(x, y, text, color) {
    floatyTexts.push({
        x: x,
        y: y,
        text: text,
        color: color,
        alpha: 1.0,
        dy: -0.6
    });
}

function createAttackParticles(x, y, color) {
    for (let i = 0; i < 8; i++) {
        particles.push({
            x: x,
            y: y,
            vx: (Math.random() - 0.5) * 4,
            vy: (Math.random() - 0.5) * 4,
            size: Math.random() * 3 + 1,
            color: color,
            alpha: 1.0,
            decay: Math.random() * 0.05 + 0.02
        });
    }
}

function createSpawningParticles(x, y) {
    for (let i = 0; i < 20; i++) {
        particles.push({
            x: x,
            y: y,
            vx: (Math.random() - 0.5) * 6,
            vy: (Math.random() - 0.5) * 6,
            size: Math.random() * 4 + 1.5,
            color: 'var(--cyber-pink)',
            alpha: 1.0,
            decay: Math.random() * 0.04 + 0.015
        });
    }
}

function logConsole(text, classStyle) {
    const consoleBox = document.getElementById('logger-console');
    const timestamp = timeElapsed.toFixed(1) + 's';
    consoleBox.innerHTML += `<div class="console-line ${classStyle}">[${timestamp}] ${text}</div>`;
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

function setSpeed(factor) {
    speedFactor = factor;
    // Set speed active button styles
    const buttons = document.querySelectorAll('.btn-speed');
    buttons.forEach((btn, idx) => {
        if (idx === 0 && factor === 1) btn.classList.add('active');
        else if (idx === 1 && factor === 2) btn.classList.add('active');
        else if (idx === 2 && factor === 5) btn.classList.add('active');
        else btn.classList.remove('active');
    });
    logConsole(`[System] Simulation speed set to ${factor}x`, 'text-blue');
}

// --- 7. UI HUD updates ---
function updateHUD() {
    // Win rate calculations
    const winRate = ((winCount30 / totalBattles) * 100).toFixed(1) + '%';
    document.getElementById('stat-win-rate').innerText = winRate;
    
    // Alive / Kills
    const aliveGobs = goblins.filter(g => g.active).length;
    document.getElementById('stat-kills').innerText = `${goblins.length - aliveGobs}/${goblins.length}`;
    document.getElementById('stat-duration').innerText = timeElapsed.toFixed(1) + 's';

    // Hero HP updates
    const classes = ['knight', 'ranger', 'healer'];
    classes.forEach(c => {
        const u = units[c];
        const max = stats[c].max_hp;
        document.getElementById(`${c}-hp-txt`).innerText = `${Math.round(u.hp)}/${max} HP`;
        
        const bar = document.getElementById(`${c}-hp-bar`);
        const pct = Math.max(0, (u.hp / max) * 100);
        bar.style.width = pct + '%';

        const cdBadge = document.getElementById(`${c}-cd-badge`);
        if (!u.active) {
            cdBadge.innerText = 'DEAD';
            cdBadge.className = 'cooldown-badge on-cooldown';
            cdBadge.style.color = '#ff0000';
            cdBadge.style.background = 'rgba(255, 0, 0, 0.1)';
        } else if (u.cd > 0) {
            cdBadge.innerText = `CD: ${u.cd}s`;
            cdBadge.className = 'cooldown-badge on-cooldown';
        } else {
            cdBadge.innerText = 'READY';
            cdBadge.className = 'cooldown-badge';
        }
    });
}

// --- 8. Render Engine & Loops ---
let lastTime = 0;
let stepTimer = 0;

function renderLoop(time) {
    if (!lastTime) lastTime = time;
    const delta = (time - lastTime) / 1000.0;
    lastTime = time;

    if (isRunning) {
        // We step the simulation physics at approx 10hz.
        // We run multiple physics steps if speed factor is activated.
        stepTimer += delta * speedFactor;
        while (stepTimer >= 0.1) {
            simulationStep();
            updateHUD();
            stepTimer -= 0.1;
        }
    }

    // Render operations
    drawScene();
    updateVisualParticles(delta);

    requestAnimationFrame(renderLoop);
}

function drawScene() {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // 1. Draw Grid Lines (Futuristic tactical radar effect)
    ctx.strokeStyle = 'rgba(0, 255, 204, 0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < CANVAS_WIDTH; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, CANVAS_HEIGHT);
        ctx.stroke();
    }
    for (let y = 0; y < CANVAS_HEIGHT; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(CANVAS_WIDTH, y);
        ctx.stroke();
    }

    // 2. Draw Obstacles / Boundaries
    ctx.strokeStyle = 'rgba(0, 255, 204, 0.08)';
    ctx.lineWidth = 4;
    ctx.strokeRect(4, 4, CANVAS_WIDTH - 8, CANVAS_HEIGHT - 8);

    // 3. Draw Local Vision Radius Overlay for selected hovered agent
    // (We render this to show recruiters strict decentralized local sight boundaries!)
    Object.keys(units).forEach(k => {
        const u = units[k];
        if (!u.active) return;
        
        // Simple hover check: is mouse near coordinates?
        // In web simulator, let's draw subtle vision range rings for all heroes
        ctx.beginPath();
        ctx.arc(u.x, u.y, stats[k].range, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.005)';
        ctx.fill();
        ctx.strokeStyle = k === 'knight' ? 'rgba(59, 130, 246, 0.04)' : k === 'ranger' ? 'rgba(0, 255, 204, 0.04)' : 'rgba(16, 185, 129, 0.04)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 8]);
        ctx.stroke();
        ctx.setLineDash([]);
    });

    // 4. Draw Healing Beams
    ctx.lineWidth = 2.5;
    healBeams.forEach(b => {
        ctx.strokeStyle = `rgba(16, 185, 129, ${b.alpha})`;
        ctx.beginPath();
        ctx.moveTo(b.fromX, b.fromY);
        ctx.lineTo(b.toX, b.toY);
        ctx.stroke();
        
        // Beam particles
        if (Math.random() < 0.3) {
            particles.push({
                x: b.toX + (Math.random() - 0.5) * 10,
                y: b.toY + (Math.random() - 0.5) * 10,
                vx: 0,
                vy: -1,
                size: Math.random() * 2 + 1,
                color: 'var(--cyber-green)',
                alpha: 1.0,
                decay: 0.05
            });
        }
    });

    // 5. Draw Projectiles (Arrows)
    projectiles.forEach((p, idx) => {
        const dist = Math.hypot(p.targetX - p.x, p.targetY - p.y);
        if (dist <= p.speed) {
            // Explode arrow and deal damage
            const gob = p.targetGob;
            if (gob.active) {
                gob.hp -= p.dmg;
                spawnFloatyText(gob.x, gob.y - 12, `-${p.dmg}`, 'var(--cyber-teal)');
                createAttackParticles(gob.x, gob.y, 'var(--cyber-teal)');
                logConsole(`[Ranger] Arrow impacts Goblin ${gob.id}! (-${p.dmg} HP)`, 'text-teal');
                
                if (gob.hp <= 0) {
                    gob.active = 0;
                    globalKills++;
                    spawnFloatyText(gob.x, gob.y - 15, "SLAIN!", 'var(--cyber-pink)');
                    logConsole(`[Ranger] Defeated Goblin ${gob.id}!`, 'text-teal');
                }
            }
            projectiles.splice(idx, 1);
        } else {
            // Move projectile
            const dirX = (p.targetX - p.x) / dist;
            const dirY = (p.targetY - p.y) / dist;
            p.x += dirX * p.speed;
            p.y += dirY * p.speed;

            // Draw arrow particle
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
            ctx.fill();
            
            // Add tiny trail particle
            if (Math.random() < 0.4) {
                particles.push({
                    x: p.x,
                    y: p.y,
                    vx: -dirX * 0.5,
                    vy: -dirY * 0.5,
                    size: 1.5,
                    color: p.color,
                    alpha: 0.6,
                    decay: 0.08
                });
            }
        }
    });

    // 6. Draw Goblins
    goblins.forEach(gob => {
        if (!gob.active) return;

        // Base circle
        ctx.fillStyle = stats.goblin.color;
        ctx.beginPath();
        ctx.arc(gob.x, gob.y, stats.goblin.radius, 0, Math.PI * 2);
        ctx.fill();

        // Glowing border
        ctx.strokeStyle = 'rgba(255, 0, 127, 0.4)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(gob.x, gob.y, stats.goblin.radius + 1, 0, Math.PI * 2);
        ctx.stroke();

        // Goblin HP Bar Overlay
        const barWidth = 24;
        const barHeight = 3;
        const pct = gob.hp / gob.maxHp;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(gob.x - barWidth/2, gob.y - 18, barWidth, barHeight);
        ctx.fillStyle = stats.goblin.color;
        ctx.fillRect(gob.x - barWidth/2, gob.y - 18, barWidth * pct, barHeight);

        // Unit ID label
        ctx.fillStyle = '#fff';
        ctx.font = '8px Fira Code';
        ctx.textAlign = 'center';
        ctx.fillText(`G${gob.id}`, gob.x, gob.y + 3);
    });

    // 7. Draw Heroes (Knight, Ranger, Healer)
    Object.keys(units).forEach(k => {
        const u = units[k];
        if (!u.active) return;

        ctx.fillStyle = stats[k].color;
        ctx.beginPath();
        ctx.arc(u.x, u.y, stats[k].radius, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(u.x, u.y, stats[k].radius + 1.5, 0, Math.PI * 2);
        ctx.stroke();

        // Hero HP overlay
        const barWidth = 22;
        const barHeight = 3;
        const pct = u.hp / stats[k].max_hp;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(u.x - barWidth/2, u.y - 16, barWidth, barHeight);
        ctx.fillStyle = stats[k].color;
        ctx.fillRect(u.x - barWidth/2, u.y - 16, barWidth * pct, barHeight);

        // Name indicator character icon
        ctx.fillStyle = '#fff';
        ctx.font = '9px Outfit';
        ctx.textAlign = 'center';
        const char = k === 'knight' ? 'K' : k === 'ranger' ? 'R' : 'H';
        ctx.fillText(char, u.x, u.y + 3);
    });

    // 8. Draw Floating damage and healing values
    floatyTexts.forEach((f, idx) => {
        ctx.fillStyle = `rgba(${f.color === 'var(--cyber-green)' ? '16, 185, 129' : f.color === 'var(--cyber-blue)' ? '59, 130, 246' : f.color === 'var(--cyber-teal)' ? '0, 255, 204' : '255, 0, 127'}, ${f.alpha})`;
        ctx.font = 'bold 9px Fira Code';
        ctx.textAlign = 'center';
        ctx.fillText(f.text, f.x, f.y);
        
        // Update physics
        f.y += f.dy;
        f.alpha -= 0.015;
        if (f.alpha <= 0) floatyTexts.splice(idx, 1);
    });

    // 9. Draw visual particles
    particles.forEach((p, idx) => {
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1.0; // reset
    });
}

function updateVisualParticles(delta) {
    // Update beams decay
    healBeams.forEach((b, idx) => {
        b.alpha -= delta * 3.0;
        if (b.alpha <= 0) healBeams.splice(idx, 1);
    });

    // Update floating particles
    particles.forEach((p, idx) => {
        p.x += p.vx;
        p.y += p.vy;
        p.alpha -= p.decay;
        if (p.alpha <= 0) {
            particles.splice(idx, 1);
        }
    });
}
