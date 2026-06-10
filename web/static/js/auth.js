/**
 * 认证管理 — JWT 存储与状态管理
 */

const Auth = {
    TOKEN_KEY: 'qa_token',
    USER_KEY: 'qa_user',

    setToken(token) {
        localStorage.setItem(this.TOKEN_KEY, token);
    },

    getToken() {
        return localStorage.getItem(this.TOKEN_KEY);
    },

    setUser(user) {
        localStorage.setItem(this.USER_KEY, JSON.stringify(user));
    },

    getUser() {
        try {
            return JSON.parse(localStorage.getItem(this.USER_KEY));
        } catch {
            return null;
        }
    },

    isLoggedIn() {
        return !!this.getToken();
    },

    logout() {
        localStorage.removeItem(this.TOKEN_KEY);
        localStorage.removeItem(this.USER_KEY);
        this._updateUI();
    },

    async login(username, password) {
        const data = await API.login(username, password);
        this.setToken(data.access_token);
        this.setUser(data.user);
        this._updateUI();
        return data.user;
    },

    async register(username, password) {
        const data = await API.register(username, password);
        this.setToken(data.access_token);
        this.setUser(data.user);
        this._updateUI();
        return data.user;
    },

    async loadUser() {
        if (!this.isLoggedIn()) return null;
        try {
            const user = await API.getMe();
            this.setUser(user);
            return user;
        } catch {
            this.logout();
            return null;
        }
    },

    _updateUI() {
        const loggedIn = this.isLoggedIn();
        document.querySelectorAll('.auth-only').forEach(el => el.style.display = loggedIn ? '' : 'none');
        document.querySelectorAll('.guest-only').forEach(el => el.style.display = loggedIn ? 'none' : '');
        const userEl = document.getElementById('userInfo');
        if (userEl) {
            const user = this.getUser();
            userEl.textContent = loggedIn && user ? user.username : '';
        }
    },
};
