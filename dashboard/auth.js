/**
 * Frontend authentication module for the Fake News Detection dashboard.
 *
 * Provides:
 *  - Token storage in localStorage
 *  - fetchWithAuth() wrapper that attaches the Bearer token
 *  - login / register / logout helpers
 *  - Auto-redirect to /dashboard/login.html when unauthenticated
 */
(function () {
  'use strict';

  const TOKEN_KEY = 'fnd_token';
  const USER_KEY = 'fnd_user';

  function getApiBase() {
    const params = new URLSearchParams(window.location.search);
    return (params.get('api') || window.FND_API_BASE || 'http://127.0.0.1:8000').replace(/\/$/, '');
  }

  function getToken() {
    return localStorage.getItem(TOKEN_KEY);
  }

  function setToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
  }

  function getUser() {
    try {
      return JSON.parse(localStorage.getItem(USER_KEY));
    } catch (_) {
      return null;
    }
  }

  function setUser(user) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }

  function clearAuth() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  }

  function isLoggedIn() {
    return !!getToken();
  }

  function requireAuth() {
    if (!isLoggedIn()) {
      window.location.href = '/login';
      return false;
    }
    return true;
  }

  /**
   * Fetch wrapper that attaches the Authorization header.
   * On 401, clears the token and redirects to login.
   */
  async function fetchWithAuth(url, options) {
    options = options || {};
    options.headers = options.headers || {};
    const token = getToken();
    if (token) {
      options.headers['Authorization'] = 'Bearer ' + token;
    }
    // Default Content-Type for JSON bodies
    if (options.body && typeof options.body === 'string' && !options.headers['Content-Type']) {
      options.headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(url, options);
    if (res.status === 401) {
      clearAuth();
      window.location.href = '/login';
      throw new Error('Session expired. Please log in again.');
    }
    return res;
  }

  /**
   * POST /auth/login with email/password form data.
   * Stores the token and user info on success.
   */
  async function login(email, password) {
    const body = new URLSearchParams();
    body.append('username', email);   // OAuth2 form uses "username" field
    body.append('password', password);

    const res = await fetch(getApiBase() + '/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: body.toString(),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Login failed');
    }
    const data = await res.json();
    setToken(data.access_token);
    setUser({ id: data.user_id, username: data.username });
    return data;
  }

  /**
   * POST /auth/register with username, email, password JSON body.
   */
  async function register(username, email, password) {
    const res = await fetch(getApiBase() + '/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Registration failed');
    }
    return await res.json();
  }

  /**
   * Clear token and redirect to login page.
   */
  function logout() {
    clearAuth();
    window.location.href = '/login';
  }

  /**
   * Get current user info from /auth/me.
   */
  async function getCurrentUser() {
    try {
      const res = await fetchWithAuth(getApiBase() + '/auth/me');
      if (res.ok) return await res.json();
    } catch (_) {}
    return null;
  }

  // ── Interceptor: patch global fetch to always attach token ──
  const _origFetch = window.fetch;
  window.fetch = function (input, init) {
    init = init || {};
    init.headers = init.headers || {};
    const token = getToken();
    if (token) {
      // If headers is a Headers object, use set; otherwise plain object
      if (init.headers instanceof Headers) {
        init.headers.set('Authorization', 'Bearer ' + token);
      } else {
        init.headers['Authorization'] = 'Bearer ' + token;
      }
    }
    return _origFetch.call(this, input, init).then(function (res) {
      if (res.status === 401) {
        clearAuth();
        window.location.href = '/login';
      }
      return res;
    });
  };

  // ── Public API ──
  window.FndAuth = {
    getToken: getToken,
    getUser: getUser,
    isLoggedIn: isLoggedIn,
    requireAuth: requireAuth,
    fetchWithAuth: fetchWithAuth,
    login: login,
    register: register,
    logout: logout,
    getCurrentUser: getCurrentUser,
    getApiBase: getApiBase,
  };
})();
