/**
 * 前端 SPA 路由 — 基于 hash 的简单路由
 */

const Router = {
    routes: {},

    register(path, handler) {
        this.routes[path] = handler;
    },

    navigate(path) {
        window.location.hash = path;
    },

    getCurrentPath() {
        return window.location.hash.slice(1) || '/';
    },

    async handle() {
        const path = this.getCurrentPath();
        const handler = this.routes[path] || this.routes['/404'];
        if (handler) {
            await handler();
        }
    },

    start() {
        window.addEventListener('hashchange', () => this.handle());
        this.handle();
    },
};

// ── 页面渲染函数 ──

async function renderHome() {
    const app = document.getElementById('app');
    app.innerHTML = `
        <div class="hero">
            <h1>智能问答系统</h1>
            <p class="hero-desc">上传你的问答模型，在线评测、对比打分，登上排行榜</p>
            <div class="hero-actions">
                <a href="#/qa" class="btn btn-primary">快速体验</a>
                <a href="#/ranking" class="btn btn-outline">排行榜</a>
            </div>
        </div>
        <div class="features">
            <div class="feature-card">
                <h3>🤖 检索式问答</h3>
                <p>上传 FAQ 数据，构建 TF-IDF 索引，秒级响应</p>
            </div>
            <div class="feature-card">
                <h3>🧠 生成式问答</h3>
                <p>上传 LoRA 权重，基于千问模型微调，智能生成</p>
            </div>
            <div class="feature-card">
                <h3>📊 排行榜</h3>
                <p>多模型对比打分，全民评测，真实效果一目了然</p>
            </div>
        </div>
    `;
}

async function renderLogin() {
    const app = document.getElementById('app');
    app.innerHTML = `
        <div class="form-page">
            <h2>登录</h2>
            <form id="loginForm">
                <div class="form-group">
                    <label>用户名</label>
                    <input type="text" id="loginUsername" required>
                </div>
                <div class="form-group">
                    <label>密码</label>
                    <input type="password" id="loginPassword" required>
                </div>
                <div id="loginError" class="form-error hidden"></div>
                <button type="submit" class="btn btn-primary btn-full">登录</button>
            </form>
            <p class="form-footer">没有账号？<a href="#/register">注册</a></p>
        </div>
    `;
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = e.target.querySelector('button');
        btn.disabled = true; btn.textContent = '登录中...';
        try {
            await Auth.login(
                document.getElementById('loginUsername').value,
                document.getElementById('loginPassword').value,
            );
            Router.navigate('/dashboard');
        } catch (err) {
            document.getElementById('loginError').textContent = err.message;
            document.getElementById('loginError').classList.remove('hidden');
        }
        btn.disabled = false; btn.textContent = '登录';
    });
}

async function renderRegister() {
    const app = document.getElementById('app');
    app.innerHTML = `
        <div class="form-page">
            <h2>注册</h2>
            <form id="registerForm">
                <div class="form-group">
                    <label>用户名</label>
                    <input type="text" id="regUsername" required minlength="2">
                </div>
                <div class="form-group">
                    <label>密码</label>
                    <input type="password" id="regPassword" required minlength="6">
                </div>
                <div id="regError" class="form-error hidden"></div>
                <button type="submit" class="btn btn-primary btn-full">注册</button>
            </form>
            <p class="form-footer">已有账号？<a href="#/login">登录</a></p>
        </div>
    `;
    document.getElementById('registerForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = e.target.querySelector('button');
        btn.disabled = true; btn.textContent = '注册中...';
        try {
            await Auth.register(
                document.getElementById('regUsername').value,
                document.getElementById('regPassword').value,
            );
            Router.navigate('/dashboard');
        } catch (err) {
            document.getElementById('regError').textContent = err.message;
            document.getElementById('regError').classList.remove('hidden');
        }
        btn.disabled = false; btn.textContent = '注册';
    });
}

async function renderDashboard() {
    if (!Auth.isLoggedIn()) { Router.navigate('/login'); return; }
    const app = document.getElementById('app');
    app.innerHTML = `
        <div class="dashboard">
            <div class="dash-header">
                <h2>我的模型</h2>
                <button class="btn btn-primary" onclick="showCreateModel()">+ 新建模型</button>
            </div>
            <div id="createModelForm" class="create-form hidden">
                <h3>新建模型</h3>
                <form id="modelForm">
                    <div class="form-group"><label>模型名称</label><input type="text" id="modelName" required></div>
                    <div class="form-group">
                        <label>类型</label>
                        <select id="modelType">
                            <option value="retrieval">检索式 (FAQ数据)</option>
                            <option value="generative">生成式 (LoRA权重)</option>
                        </select>
                    </div>
                    <div class="form-group"><label>描述</label><textarea id="modelDesc" rows="2"></textarea></div>
                    <div class="form-group">
                        <label><input type="checkbox" id="modelPublic" checked> 公开</label>
                    </div>
                    <div id="modelError" class="form-error hidden"></div>
                    <button type="submit" class="btn btn-primary">创建</button>
                    <button type="button" class="btn btn-outline" onclick="hideCreateModel()">取消</button>
                </form>
            </div>
            <div id="modelList" class="model-grid"><p>加载中...</p></div>
        </div>
    `;
    document.getElementById('modelForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        try {
            await API.createModel({
                name: document.getElementById('modelName').value,
                type: document.getElementById('modelType').value,
                description: document.getElementById('modelDesc').value,
                public: document.getElementById('modelPublic').checked,
            });
            hideCreateModel();
            loadModelList();
        } catch (err) {
            document.getElementById('modelError').textContent = err.message;
            document.getElementById('modelError').classList.remove('hidden');
        }
    });
    loadModelList();
}

async function loadModelList() {
    const container = document.getElementById('modelList');
    try {
        const data = await API.listMyModels();
        if (data.models.length === 0) {
            container.innerHTML = '<p class="empty">还没有模型，点击上方按钮创建</p>';
            return;
        }
        container.innerHTML = data.models.map(m => `
            <div class="model-card">
                <div class="model-type ${m.type}">${m.type === 'retrieval' ? '📄 检索式' : '🧠 生成式'}</div>
                <h4>${m.name}</h4>
                <p>${m.description || '无描述'}</p>
                <div class="model-meta">
                    <span>⭐ ${m.avg_score.toFixed(2)}</span>
                    <span>📥 ${m.downloads}</span>
                    <span>${m.public ? '🌍 公开' : '🔒 私有'}</span>
                </div>
                <div class="model-actions">
                    <a href="#/qa?model=${m.id}" class="btn-sm">测试</a>
                    ${!m.file_path ? `<input type="file" id="upload_${m.id}" style="display:none" onchange="uploadModelFile(${m.id})">
                        <button class="btn-sm btn-outline" onclick="document.getElementById('upload_${m.id}').click()">上传文件</button>` : ''}
                    <button class="btn-sm btn-danger" onclick="deleteModel(${m.id})">删除</button>
                </div>
            </div>
        `).join('');
    } catch (err) {
        container.innerHTML = `<p class="error">加载失败: ${err.message}</p>`;
    }
}

window.showCreateModel = () => document.getElementById('createModelForm').classList.remove('hidden');
window.hideCreateModel = () => document.getElementById('createModelForm').classList.add('hidden');

window.uploadModelFile = async (modelId) => {
    const input = document.getElementById(`upload_${modelId}`);
    const file = input.files[0];
    if (!file) return;
    try {
        await API.uploadModelFile(modelId, file);
        loadModelList();
    } catch (err) { alert(err.message); }
};

window.deleteModel = async (modelId) => {
    if (!confirm('确定删除此模型？')) return;
    try {
        await API.deleteModel(modelId);
        loadModelList();
    } catch (err) { alert(err.message); }
};

async function renderQA() {
    const app = document.getElementById('app');
    const params = new URLSearchParams(window.location.hash.split('?')[1] || '');
    const selectedModelId = params.get('model');

    // 加载模型列表
    let models = [];
    try {
        const mine = await API.listMyModels();
        const pub = await API.listPublicModels();
        models = [...mine.models, ...pub.models];
        // 去重
        const seen = new Set();
        models = models.filter(m => { const k = m.id; if (seen.has(k)) return false; seen.add(k); return true; });
    } catch {}

    app.innerHTML = `
        <div class="qa-page">
            <h2>问答测试</h2>
            <div class="qa-controls">
                <select id="qaModelSelect">
                    <option value="">-- 选择模型 --</option>
                    ${models.map(m => `<option value="${m.id}" ${m.id == selectedModelId ? 'selected' : ''}>${m.name} (${m.type})</option>`).join('')}
                </select>
                <input type="text" id="qaInput" placeholder="输入问题..." autofocus>
                <button class="btn btn-primary" onclick="doAsk()">提问</button>
            </div>
            <div id="qaResult" class="qa-result hidden"></div>
        </div>
    `;
    document.getElementById('qaInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') doAsk(); });
    if (selectedModelId && document.getElementById('qaModelSelect').value) {
        document.getElementById('qaInput').focus();
    }
}

window.doAsk = async () => {
    const modelId = document.getElementById('qaModelSelect').value;
    const question = document.getElementById('qaInput').value.trim();
    if (!modelId || !question) return;

    const resultEl = document.getElementById('qaResult');
    resultEl.classList.remove('hidden');
    resultEl.innerHTML = '<div class="loading"><div class="spinner"></div> 思考中...</div>';

    try {
        const data = await API.ask(parseInt(modelId), question);
        resultEl.innerHTML = `
            <div class="answer-card">
                <p class="answer-text">${data.answer}</p>
                <div class="answer-meta">
                    <span class="badge ${data.confidence > 0.7 ? 'high' : 'mid'}">${(data.confidence*100).toFixed(0)}%</span>
                    <span class="badge category">${data.model_name}</span>
                    <span class="source">${data.latency_ms}ms</span>
                </div>
            </div>
        `;
    } catch (err) {
        resultEl.innerHTML = `<div class="form-error">${err.message}</div>`;
    }
};

async function renderRanking() {
    const app = document.getElementById('app');
    app.innerHTML = `
        <div class="ranking-page">
            <h2>🏆 排行榜</h2>
            <div class="ranking-tabs">
                <button class="tab active" onclick="loadRanking('all','avg_score')">综合</button>
                <button class="tab" onclick="loadRanking('all','downloads')">热度</button>
                <button class="tab" onclick="loadRanking('retrieval','avg_score')">检索式</button>
                <button class="tab" onclick="loadRanking('generative','avg_score')">生成式</button>
            </div>
            <div id="rankingList"><p>加载中...</p></div>
        </div>
    `;
    loadRanking('all', 'avg_score');
}

window.loadRanking = async (type, sort) => {
    const container = document.getElementById('rankingList');
    container.innerHTML = '<p>加载中...</p>';
    document.querySelectorAll('.ranking-tabs .tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.ranking-tabs .tab:nth-child(${['all','downloads','retrieval','generative'].indexOf(type) + 1})`)?.classList.add('active');

    try {
        const data = await API.getRanking({ model_type: type, sort_by: sort });
        if (data.items.length === 0) {
            container.innerHTML = '<p class="empty">暂无数据</p>'; return;
        }
        container.innerHTML = `
            <table class="ranking-table">
                <thead><tr><th>#</th><th>用户</th><th>模型</th><th>类型</th><th>评分</th><th>调用</th></tr></thead>
                <tbody>
                    ${data.items.map((item, i) => `
                        <tr class="${i < 3 ? 'top-' + (i+1) : ''}">
                            <td class="rank">${item.rank}</td>
                            <td>${item.username}</td>
                            <td><a href="#/models/${item.model_id}">${item.model_name}</a></td>
                            <td>${item.model_type === 'retrieval' ? '检索式' : '生成式'}</td>
                            <td>⭐ ${item.avg_score.toFixed(2)}</td>
                            <td>${item.call_count}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (err) {
        container.innerHTML = `<p class="error">${err.message}</p>`;
    }
};

// ── 注册路由 ──

Router.register('/', renderHome);
Router.register('/login', renderLogin);
Router.register('/register', renderRegister);
Router.register('/dashboard', renderDashboard);
Router.register('/qa', renderQA);
Router.register('/ranking', renderRanking);
