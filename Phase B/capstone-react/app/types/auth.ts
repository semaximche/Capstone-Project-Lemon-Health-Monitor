// authentication context
interface AuthContextType {
    // server login schema
    accessToken: String | null;

    // client status and functions
    status: String | null;
    onLogin: (request: AuthLoginRequest) => void;
    onLogout: () => void;
}

// server login request schema
interface AuthLoginRequest {
    username: string;
    password: string;
}

// server login response schema
interface AuthLoginResponse {
    accessToken: string;
}