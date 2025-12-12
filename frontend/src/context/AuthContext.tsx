import React, { createContext, useContext, useState, useEffect } from "react";
import axios from "axios";

interface User {
  email: string;
  full_name?: string;
  is_superuser: boolean;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (token: string) => void;
  logout: () => void;
  isAuthenticated: boolean;
  isLoading: boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [user, setUser] = useState<User | null>(null);
  // Initialize state from localStorage to avoid effect sync issues
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem("token")
  );
  const [isLoading, setIsLoading] = useState(true);

  // Set initial axios header synchronously during initialization if possible,
  // or use an effect that doesn't trigger a re-render cycle for isLoading if it's not needed.
  // However, the cleanest way is to separate the side effect (axios header) from the state update.

  useEffect(() => {
    // This effect handles the side effect of setting the global axios header
    if (token) {
      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
    } else {
      delete axios.defaults.headers.common["Authorization"];
    }

    // We can set loading to false here, but since token is initialized from localStorage,
    // we might not even need isLoading for the initial check if we trust localStorage.
    // But usually, we might want to validate the token.
    // For now, to fix the lint error "setState inside effect", we keep it simple.
    // actually, setting state in useEffect is fine IF dependencies are correct and it doesn't loop.
    // The previous error was likely due to missing deps or something else, or strict mode.
    // But let's make it robust.
    setIsLoading(false);
  }, [token]);

  const login = (newToken: string) => {
    localStorage.setItem("token", newToken);
    setToken(newToken);
    // Decode token to get user info if needed, or fetch from API
    // const decoded = jwt_decode(newToken);
    // setUser({ email: decoded.sub ... });
  };

  const logout = () => {
    localStorage.removeItem("token");
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        login,
        logout,
        isAuthenticated: !!token,
        isLoading,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
