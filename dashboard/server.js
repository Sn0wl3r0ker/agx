/**
 * AGX ROS Dashboard — Node.js Backend
 * Express + WebSocket, controls sibling Docker containers
 */

const express = require('express');
const http = require('http');
const { WebSocketServer } = require('ws');
const { spawn, execSync, exec } = require('child_process');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

const PORT = process.env.DASHBOARD_PORT || 8080;
const PROJECT = process.env.PROJECT_NAME || 'agx_ros';

// ─── Configuration ───────────────────────────────────────────────────────────

const SERVICES = {
    planning: { name: 'ROS 2 高階規劃', container: 'planning', icon: '🧭' },
    foxglove: { name: '資料視覺化', container: 'foxglove', icon: '📊' },
    vlm: { name: 'Isaac ROS 視覺加速', container: 'isaac_ros', icon: '👁️' },
    nanollm: { name: 'Nano LLM', container: 'nanollm', icon: '🤖' },
};

const TASKS = {
    planning: {
        label: 'Planning Tasks (ROS 2)',
        container: 'planning',
        items: {
            plan_lidar: { name: 'Lidar 啟動測試', cmd: 'ros2 launch urg_node2 urg_node2.launch.py', icon: '📡' },
            plan_slam: { name: 'SLAM Bringup', cmd: 'ros2 launch car_control slam_bringup.launch.py', icon: '🗺️' },
            plan_keyboard: { name: 'Keyboard Control', cmd: 'ros2 run teleop_twist_keyboard teleop_twist_keyboard', icon: '⌨️' },
            plan_savemap: { name: 'Save Map', cmd: 'ros2 run nav2_map_server map_saver_cli -f /root/ros2_ws/src/car_control/config/my_map', icon: '💾' },
        },
    },
};

// ─── Docker helpers ──────────────────────────────────────────────────────────

function runCmd(cmd, timeout = 30000) {
    return new Promise((resolve) => {
        exec(cmd, { timeout }, (err, stdout, stderr) => {
            resolve({
                ok: !err,
                stdout: (stdout || '').trim(),
                stderr: (stderr || '').trim(),
            });
        });
    });
}

async function getContainerStatus() {
    const fmt = '{{.Names}}|{{.Status}}|{{.State}}';
    const r = await runCmd(
        `docker ps -a --filter "label=com.docker.compose.project=${PROJECT}" --format "${fmt}"`
    );
    const statuses = {};
    if (r.ok && r.stdout) {
        for (const line of r.stdout.split('\n')) {
            const [name, status, state] = line.split('|');
            if (name) statuses[name] = { status, state };
        }
    }
    return statuses;
}

async function getTmuxSessions() {
    const r = await runCmd('docker exec planning bash -c "tmux ls -F \\"#{session_name}\\" 2>/dev/null || true"');
    if (r.ok && r.stdout) {
        return r.stdout.split('\n').filter(s => s.startsWith('agx_') || s.startsWith('plan_'));
    }
    // Fallback: try on host
    const r2 = await runCmd('tmux ls -F "#{session_name}" 2>/dev/null || true');
    if (r2.stdout) {
        return r2.stdout.split('\n').filter(s => s.startsWith('agx_') || s.startsWith('plan_'));
    }
    return [];
}

const fs = require('fs');

const MODE = process.env.DEPLOY_MODE || 'pc';
console.log(`[Info] Deploy mode: ${MODE}`);

function composeCmd(folder) {
    let cmd = `docker compose -f /project/${folder}/docker-compose.yaml`;
    const override = `/project/${folder}/docker-compose.${MODE}.yaml`;
    if (fs.existsSync(override)) {
        cmd += ` -f ${override}`;
    }
    return cmd + ` -p ${PROJECT}`;
}

// ─── Persistent ROS publisher for cmd_vel ────────────────────────────────────

let cmdVelProcess = null;

const PY_PUB = `
import rclpy, sys, json
from geometry_msgs.msg import Twist
rclpy.init()
node = rclpy.create_node('web_teleop')
pub = node.create_publisher(Twist, '/cmd_vel', 10)
t = Twist()
sys.stdout.write('READY\\n')
sys.stdout.flush()
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        t.linear.x = float(d.get('lx', 0))
        t.angular.z = float(d.get('az', 0))
        pub.publish(t)
    except:
        pass
`.trim();

function ensureCmdVelProcess() {
    if (cmdVelProcess && cmdVelProcess.exitCode === null) return; // still alive

    const encoded = Buffer.from(PY_PUB).toString('base64');
    cmdVelProcess = spawn('docker', [
        'exec', '-i', 'planning', 'bash', '-c',
        `source /opt/ros/humble/setup.bash && python3 -c "$(echo ${encoded} | base64 -d)"`,
    ]);

    cmdVelProcess.stdout.on('data', (d) => {
        const msg = d.toString().trim();
        if (msg === 'READY') console.log('[cmd_vel] Persistent ROS publisher ready.');
    });

    cmdVelProcess.stderr.on('data', (d) => {
        console.error('[cmd_vel]', d.toString().trim());
    });

    cmdVelProcess.on('close', (code) => {
        console.log(`[cmd_vel] Process exited (code ${code})`);
        cmdVelProcess = null;
    });
}

function sendCmdVel(lx, az) {
    ensureCmdVelProcess();
    if (cmdVelProcess && cmdVelProcess.exitCode === null) {
        cmdVelProcess.stdin.write(JSON.stringify({ lx, az }) + '\n');
        return true;
    }
    return false;
}

// ─── REST API ────────────────────────────────────────────────────────────────

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

app.get('/api/status', async (_req, res) => {
    const [containers, sessions] = await Promise.all([
        getContainerStatus(),
        getTmuxSessions(),
    ]);
    res.json({ containers, sessions });
});

app.get('/api/config', (_req, res) => {
    res.json({ services: SERVICES, tasks: TASKS });
});

app.post('/api/service/:action', async (req, res) => {
    const { action } = req.params;
    const { folder } = req.body;

    if (!folder || !SERVICES[folder]) {
        return res.json({ ok: false, error: 'Invalid service' });
    }

    const cmds = {
        up: `${composeCmd(folder)} up -d`,
        down: `${composeCmd(folder)} down`,
        build: `${composeCmd(folder)} build`,
        rebuild: `${composeCmd(folder)} up -d --build --force-recreate`,
    };

    if (!cmds[action]) return res.json({ ok: false, error: 'Unknown action' });

    const r = await runCmd(cmds[action], action === 'build' || action === 'rebuild' ? 600000 : 120000);
    res.json(r);
});

app.post('/api/service-all/:action', async (_req, res) => {
    const { action } = _req.params;
    let composeFile = MODE === 'agx' ? 'docker-compose.yaml' : 'docker-compose.pc.yaml';
    const rootCompose = `docker compose -f /project/${composeFile}`;
    const cmds = {
        up: `${rootCompose} -p ${PROJECT} up -d`,
        down: `${rootCompose} -p ${PROJECT} down --remove-orphans`,
    };
    if (!cmds[action]) return res.json({ ok: false, error: 'Unknown action' });
    const r = await runCmd(cmds[action], 120000);
    res.json(r);
});

app.post('/api/task/launch', async (req, res) => {
    const { group, task } = req.body;
    if (!group || !TASKS[group] || !TASKS[group].items[task]) {
        return res.json({ ok: false, error: 'Invalid task' });
    }

    const container = TASKS[group].container;
    const cmd = TASKS[group].items[task].cmd;

    // Check if tmux session already exists
    const sessions = await getTmuxSessions();
    if (sessions.includes(task)) {
        return res.json({ ok: true, stdout: `Task '${task}' already running.` });
    }

    // Launch inside container's tmux (interactive shell so ~/.bashrc loads ROS)
    await runCmd(`docker exec -d ${container} tmux new-session -d -s ${task} bash -ic "${cmd}; exec bash"`);
    res.json({ ok: true, stdout: `Task '${task}' launched.` });
});

app.post('/api/task/stop', async (req, res) => {
    const { task } = req.body;
    if (task === 'all') {
        const sessions = await getTmuxSessions();
        for (const s of sessions) {
            await runCmd(`docker exec planning tmux kill-session -t ${s} 2>/dev/null`);
        }
        return res.json({ ok: true, stdout: `Stopped ${sessions.length} task(s).` });
    }
    // Try container
    await runCmd(`docker exec planning tmux kill-session -t ${task} 2>/dev/null`);
    res.json({ ok: true });
});

app.get('/api/logs', async (req, res) => {
    const { folder, lines = 50 } = req.query;
    let r;
    if (folder && SERVICES[folder]) {
        r = await runCmd(`docker logs --tail ${lines} ${SERVICES[folder].container}`, 10000);
    } else {
        r = await runCmd(`docker compose -f /project/docker-compose.yaml -p ${PROJECT} logs --tail ${lines}`, 10000);
    }
    res.json(r);
});

app.get('/api/task/logs', async (req, res) => {
    const { task } = req.query;
    if (!task) return res.json({ ok: false, error: 'Task not specified' });
    const r = await runCmd(`docker exec planning tmux capture-pane -t ${task} -p -S -50 2>/dev/null`);
    res.json(r);
});

app.post('/api/dashboard/shutdown', async (_req, res) => {
    res.json({ ok: true, stdout: 'Dashboard shutting down...' });
    // Give response time to flush, then stop own container
    setTimeout(() => {
        exec('docker stop dashboard', () => process.exit(0));
    }, 500);
});

// ─── WebSocket ───────────────────────────────────────────────────────────────

wss.on('connection', (ws) => {
    console.log('[ws] Client connected');

    ws.on('message', (raw) => {
        try {
            const msg = JSON.parse(raw);
            if (msg.type === 'cmd_vel') {
                const ok = sendCmdVel(msg.lx || 0, msg.az || 0);
                ws.send(JSON.stringify({ type: 'cmd_vel_ack', ok }));
            }
        } catch (e) {
            // ignore
        }
    });

    // Push status updates every 3s
    const interval = setInterval(async () => {
        try {
            const [containers, sessions] = await Promise.all([
                getContainerStatus(),
                getTmuxSessions(),
            ]);
            ws.send(JSON.stringify({ type: 'status', containers, sessions }));
        } catch (e) {
            // ignore
        }
    }, 3000);

    ws.on('close', () => {
        clearInterval(interval);
        console.log('[ws] Client disconnected');
    });
});

// ─── Start ───────────────────────────────────────────────────────────────────

server.listen(PORT, '0.0.0.0', () => {
    console.log(`
╔══════════════════════════════════════════╗
║        🚀 AGX ROS Dashboard             ║
╠══════════════════════════════════════════╣
║  Port:    ${String(PORT).padEnd(30)}║
║  URL:     ${'http://localhost:' + PORT}${' '.repeat(Math.max(0, 30 - ('http://localhost:' + PORT).length))}║
╚══════════════════════════════════════════╝
  `);
});
