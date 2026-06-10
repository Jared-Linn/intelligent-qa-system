/**
 * API 客户端 — 封装所有后端接口调用
 */

const API = {
    BASE: '',  // 同域

    // ── 认证 ──
    async register(username, password) {
        return this._post('/api/auth/register', { username, password });
    },

    async login(username, password) {
        return this._post('/api/auth/login', { username, password });
    },

    async getMe() {
        return this._get('/api/auth/me');
    },

    // ── 模型 ──
    async listMyModels() {
        return this._get('/api/models');
    },

    async createModel(data) {
        return this._post('/api/models', data);
    },

    async uploadModelFile(modelId, file) {
        const formData = new FormData();
        formData.append('file', file);
        const token = Auth.getToken();
        const resp = await fetch(`${this.BASE}/api/models/${modelId}/upload`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` },
            body: formData,
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || '上传失败');
        return resp.json();
    },

    async listPublicModels(type) {
        const query = type ? `?model_type=${type}` : '';
        return this._get(`/api/models/public${query}`);
    },

    async getModel(id) {
        return this._get(`/api/models/${id}`);
    },

    async deleteModel(id) {
        return this._delete(`/api/models/${id}`);
    },

    // ── 问答 ──
    async ask(modelId, question) {
        return this._post('/api/qa/ask', { model_id: modelId, question });
    },

    async compare(modelIds, question) {
        return this._post('/api/qa/compare', { model_ids: modelIds, question });
    },

    // ── 评分 ──
    async submitRating(data) {
        return this._post('/api/ratings', data);
    },

    async getModelRatings(modelId) {
        return this._get(`/api/ratings/${modelId}`);
    },

    // ── 排行榜 ──
    async getRanking(params = {}) {
        const q = new URLSearchParams(params).toString();
        return this._get(`/api/ranking?${q}`);
    },

    // ── 内部方法 ──
    _tokenHeader() {
        const token = Auth.getToken();
        return token ? { 'Authorization': `Bearer ${token}` } : {};
    },

    async _get(url) {
        const resp = await fetch(`${this.BASE}${url}`, {
            headers: { ...this._tokenHeader() },
        });
        if (resp.status === 401) { Auth.logout(); window.location.hash = '#/login'; }
        if (!resp.ok) throw new Error((await resp.json()).detail || `请求失败 (${resp.status})`);
        return resp.json();
    },

    async _post(url, data) {
        const resp = await fetch(`${this.BASE}${url}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...this._tokenHeader(),
            },
            body: data ? JSON.stringify(data) : undefined,
        });
        if (resp.status === 401) { Auth.logout(); window.location.hash = '#/login'; }
        if (!resp.ok) throw new Error((await resp.json()).detail || `请求失败 (${resp.status})`);
        return resp.json();
    },

    async _delete(url) {
        const resp = await fetch(`${this.BASE}${url}`, {
            method: 'DELETE',
            headers: { ...this._tokenHeader() },
        });
        if (resp.status === 401) { Auth.logout(); window.location.hash = '#/login'; }
        if (!resp.ok) throw new Error((await resp.json()).detail || '删除失败');
        return resp.json();
    },
};
