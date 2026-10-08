import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { createGuestSession, getMe, login as apiLogin, logout as apiLogout } from '../api';

const AuthContext = createContext(null);
const GUEST_ID_KEY = 'squinta.guest_id';

function getGuestId() {
  let guestId = localStorage.getItem(GUEST_ID_KEY);
  if (!guestId) {
    // UUIDs keep each browser profile's guest data separate without asking
    // the user to register or sign in.
    guestId = typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (char) => {
          const random = Math.random() * 16 | 0;
          return (char === 'x' ? random : (random & 0x3 | 0x8)).toString(16);
        });
    localStorage.setItem(GUEST_ID_KEY, guestId);
  }
  return guestId;
}

function normalizeUser(user) {
  if (!user) return null;
  return {
    ...user,
    // Existing Google accounts may have no explicit guest flag, while the
    // locally generated guest address always uses this reserved suffix.
    isGuest: user.is_guest ?? user.isGuest ?? Boolean(user.email?.endsWith('@guest.squinta.local')),
  };
}

export function AuthProvider({ children }) {
  // Keep supporting the optional Google OAuth callback, but never require it.
  // Read the token before React Router can replace the callback URL.
  const [initialToken] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token');
    if (token) {
      localStorage.setItem('token', token);
      window.history.replaceState({}, '', window.location.pathname);
    }
    return token;
  });

  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function restoreSession() {
      try {
        // The callback token is already stored by initialToken's initializer.
        void initialToken;
        const token = localStorage.getItem('token');
        if (token) {
          try {
            const currentUser = await getMe();
            if (!cancelled) setUser(normalizeUser(currentUser));
            // Release the loading screen for restored Google and guest tokens.
            if (!cancelled) setLoading(false);
            return;
          } catch {
            // Expired/invalid tokens fall back to this browser's guest session.
            localStorage.removeItem('token');
          }
        }

        const session = await createGuestSession(getGuestId());
        localStorage.setItem('token', session.token);
        if (!cancelled) setUser(normalizeUser(session.user));
      } catch {
        if (!cancelled) setUser(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    restoreSession();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(() => {
    // Optional: connect Google when using Google Photos, not to enter Squinta.
    apiLogin();
  }, []);

  const logout = useCallback(async () => {
    // "Log out" from a connected Google account means switching back to guest
    // mode, not being sent to a login wall.
    if (user && !user.isGuest) {
      try {
        await apiLogout();
      } catch {
        // Clear the local Google session even if the remote logout fails.
      }
    }
    localStorage.removeItem('token');
    try {
      const session = await createGuestSession(getGuestId());
      localStorage.setItem('token', session.token);
      setUser(normalizeUser(session.user));
    } catch {
      setUser(null);
    }
  }, [user]);

  const value = {
    user,
    loading,
    login,
    logout,
    isAuthenticated: !!user,
    isGuest: !!user?.isGuest,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default useAuth;
