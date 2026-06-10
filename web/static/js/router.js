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
        const hash = window.location.hash.slice(1) || '/';
        // 分离路径和查询参数，只取路径部分匹配路由
        return hash.split('?')[0];
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
                        <select id="modelType" onchange="updateTypeHint()">
                            <option value="retrieval">检索式 — TF-IDF匹配，CPU秒级响应</option>
                            <option value="generative">生成式 — 千问LoRA微调，需GPU</option>
                        </select>
                        <div id="typeHint" class="form-hint">上传 JSON 问答数据文件</div>
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
            
            <!-- 编辑模型弹窗 -->
            <div id="editModal" class="modal-overlay hidden">
                <div class="modal-content">
                    <h3>✏️ 编辑模型</h3>
                    <form id="editForm">
                        <input type="hidden" id="editId">
                        <div class="form-group">
                            <label>模型名称</label>
                            <input type="text" id="editName" required>
                        </div>
                        <div class="form-group">
                            <label>描述</label>
                            <textarea id="editDesc" rows="2"></textarea>
                        </div>
                        <div class="form-group">
                            <label><input type="checkbox" id="editPublic"> 公开</label>
                        </div>
                        <div id="editError" class="form-error hidden"></div>
                        <button type="submit" class="btn btn-primary">保存</button>
                        <button type="button" class="btn btn-outline" onclick="hideEditModal()">取消</button>
                    </form>
                </div>
            </div>

            <div id="modelList" class="model-grid"><p>加载中...</p></div>
        </div>
    `;
    setupEditModal();
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
            <div class="model-card" data-model-id="${m.id}">
                <div class="card-header">
                    <span class="model-type ${m.type}">${m.type === 'retrieval' ? '📄 检索式' : '🧠 生成式'}</span>
                    <span class="model-status-badge ${m.status}">${m.status === 'active' ? '✅ 正常' : '⛔ 停用'}</span>
                </div>
                <h4>${m.name}</h4>
                <p>${m.description || '无描述'}</p>
                <div class="file-status ${m.file_path ? 'uploaded' : 'pending'}">
                    ${m.file_path
                        ? '📎 已上传: ' + decodeURIComponent(m.file_path.split('/').pop().split('\\\\').pop())
                        : m.type === 'retrieval' ? '⏳ 等待上传 FAQ 数据（.json）' : '⏳ 等待上传 LoRA 权重（.zip）'}
                </div>
                <div class="model-meta">
                    <span>⭐ ${Number(m.avg_score || 0).toFixed(2)}</span>
                    <span>📥 ${m.downloads || 0}</span>
                    <span>${m.public ? '🌍 公开' : '🔒 私有'}</span>
                    <span>🆔 #${m.id}</span>
                </div>
                <div class="model-actions">
                    <a href="#/qa?model=${m.id}" class="btn-sm ${m.file_path ? 'btn-primary' : ''}">🧪 测试</a>
                    <input type="file" id="upload_${m.id}" style="display:none"
                           accept="${m.type === 'retrieval' ? '.json' : '.zip'}"
                           onchange="uploadModelFile(${m.id})">
                    <button class="btn-sm btn-outline btn-upload" data-upload-id="${m.id}">
                        ${m.file_path ? '🔄 重新上传' : '📤 上传文件'}
                    </button>
                    <span class="file-hint">${m.type === 'retrieval' ? '支持 .json' : '支持 .zip'}</span>
                    <button class="btn-sm btn-outline btn-edit" data-edit-id="${m.id}">✏️ 编辑</button>
                    <button class="btn-sm btn-danger btn-del" data-del-id="${m.id}">🗑️ 删除</button>
                </div>
            </div>
        `).join('');

        // 事件委托：点击删除/上传按钮
        container.querySelectorAll('.btn-del').forEach(btn => {
            btn.onclick = (e) => {
                const id = parseInt(e.target.dataset.delId);
                if (confirm('确定删除此模型？')) {
                    window.deleteModel(id).catch(err => alert(err.message));
                }
            };
        });
        container.querySelectorAll('.btn-edit').forEach(btn => {
            btn.onclick = (e) => {
                const id = parseInt(e.target.dataset.editId);
                showEditModal(id);
            };
        });
        container.querySelectorAll('.btn-upload').forEach(btn => {
            btn.onclick = (e) => {
                const id = parseInt(e.target.dataset.uploadId);
                const input = document.getElementById(`upload_${id}`);
                if (input) input.click();
            };
        });
    } catch (err) {
        container.innerHTML = `<p class="error">加载失败: ${err.message}</p>`;
    }
}

window.showCreateModel = () => document.getElementById('createModelForm').classList.remove('hidden');
window.hideCreateModel = () => document.getElementById('createModelForm').classList.add('hidden');

window.showEditModal = async (modelId) => {
    try {
        const m = await API.getModel(modelId);
        document.getElementById('editId').value = m.id;
        document.getElementById('editName').value = m.name;
        document.getElementById('editDesc').value = m.description || '';
        document.getElementById('editPublic').checked = m.public;
        document.getElementById('editModal').classList.remove('hidden');
    } catch (err) { alert(err.message); }
};

window.hideEditModal = () => {
    document.getElementById('editModal').classList.add('hidden');
};

function setupEditModal() {
    document.getElementById('editForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = e.target.querySelector('button[type=submit]');
        btn.disabled = true; btn.textContent = '保存中...';
        try {
            await API.updateModel(
                parseInt(document.getElementById('editId').value),
                {
                    name: document.getElementById('editName').value,
                    description: document.getElementById('editDesc').value,
                    public: document.getElementById('editPublic').checked,
                }
            );
            hideEditModal();
            loadModelList();
        } catch (err) { alert(err.message); }
        btn.disabled = false; btn.textContent = '保存';
    });
}


window.updateTypeHint = function() {
    const sel = document.getElementById('modelType');
    const hint = document.getElementById('typeHint');
    if (sel && hint) {
        hint.textContent = sel.value === 'retrieval'
            ? '上传 JSON 问答数据文件（含 question + answer 字段）'
            : '上传 LoRA 权重 zip 包（含 adapter_config.json）';
    }
};

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

    let models = [];
    try {
        const pub = await API.listPublicModels();
        models = pub.models || [];
    } catch {}  // 未登录也能看公开模型

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
