/**
 * AuthGuard.js — add this to src/components/
 * Protects all routes — redirects to /login if no token
 *
 * Usage in App.js:
 *   import AuthGuard from './components/AuthGuard';
 *   <Route path="/chat" element={<AuthGuard><Chat /></AuthGuard>} />
 */
import { Navigate } from 'react-router-dom';
import { ensureLocalAuth } from '../localAuth';

const isAdminWorkspaceMode = () => (
  typeof window !== 'undefined' &&
  window.location.port === '8003' &&
  process.env.REACT_APP_ADMIN_MODE === 'true'
);

export default function AuthGuard({ children, adminOnly = false }) {
  const adminWorkspace = isAdminWorkspaceMode();

  if (ensureLocalAuth()) {
    if (adminOnly && !adminWorkspace) return <Navigate to="/chat" replace />;
    return children;
  }

  const token = localStorage.getItem('aria_token');
  const user = (() => {
    try {
      return JSON.parse(localStorage.getItem('aria_user') || '{}');
    } catch {
      return {};
    }
  })();
  const role = (user.role || 'user').toLowerCase();
  const username = String(user.username || user.name || user.email || '').toLowerCase();

  const isAdmin =
    role === 'admin' ||
    role === 'administrator' ||
    role === 'superadmin' ||
    role === 'super_admin' ||
    username === 'admin' ||
    username === 'admin@eva.ai' ||
    user.isAdmin === true ||
    user.is_admin === true ||
    localStorage.getItem('aria_is_admin') === 'true';

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  if (adminOnly && !adminWorkspace) {
    return <Navigate to="/chat" replace />;
  }

  if (adminOnly && !isAdmin) {
    return (
      <div style={{ padding: 24, color: 'var(--text, #f8fafc)' }}>
        <h1 style={{ margin: 0, fontSize: 24 }}>Admin access required</h1>
      </div>
    );
  }

  return children;
}
