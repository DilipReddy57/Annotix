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
  const [token, setToken] = useState<string | null>(
    localStorage.getItem("token")
  );
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (token) {
      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
      // Ideally fetch user profile here
      // For now, we'll decode the token or just set a dummy user state if needed
      // But let's fetch the user profile if we had an endpoint.
      // Since we don't have a /me endpoint yet, we will rely on the token presence
      // and maybe decode it if we need user details immediately.
      // For MVP, we'll assume if token exists, we are logged in.
      setIsLoading(false);
    } else {
      delete axios.defaults.headers.common["Authorization"];
      setIsLoading(false);
    }
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
