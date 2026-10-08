import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getMe } from '../api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState({
    id: null,
    name: 'Squinta Demo User',
    email: 'demo@squinta.local',
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    getMe()
      .then((profile) => { if (active) setUser(profile); })
      .catch(() => {
        // The backend may still be starting; keep the UI available in demo mode.
        if (active) setUser({ id: null, name: 'Squinta Demo User', email: 'demo@squinta.local' });
      })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  const logout = useCallback(async () => {
    // Authentication is intentionally disabled in local demo mode.
  }, []);

  const value = { user, loading, login: () => {}, logout, isAuthenticated: true };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within an AuthProvider');
  return context;
}

export default useAuth;
