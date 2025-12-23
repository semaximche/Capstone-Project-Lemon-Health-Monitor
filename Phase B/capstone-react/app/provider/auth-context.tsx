import { createContext, useContext, useMemo, useState } from "react";
import { useNavigate } from "react-router";

interface AuthContextType {
    token: string | null;
    login: (newToken: string) => void;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    let navigate = useNavigate();
    const [token, setToken] = useState<string | null>(null);
    const login = (newToken: string) => {
        setToken(newToken);
        navigate("/dashboard");
    };
    const logout = () => {
        setToken(null);
        navigate("/signin");
    };
    const value = useMemo(() => ({token, login, logout}), [token]);

    return (
       <AuthContext.Provider value={value}>
        <div>token: {token}</div>
        {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    return useContext(AuthContext);
};