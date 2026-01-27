import { createContext, useContext, useMemo, useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { apiEndpoint, getWebSocketUrl } from "~/lib/api-config";

interface AnalysisData {
    id?: string;
    user_id: string;
    presigned_url?: string;
    description?: string;
    summary?: string;
    image?: string; // Base64 encoded image
}

interface NotificationItem {
    id: string;
    user_id: string;
    analysis_id: string;
    read: boolean;
    timestamp: number;
    raw?: any;
}

interface AuthContextType {
    token: string | null;
    login: (newToken: string) => void;
    logout: () => void;
    notifications: NotificationItem[];
    unreadCount: number;
    openNotification: (n: NotificationItem) => void;
    markAllRead: () => void;
    analyses: Record<string, AnalysisData>;
    selectedAnalysis: AnalysisData | null;
    setSelectedAnalysis: (analysis: AnalysisData | null) => void;
}

const AuthContext = createContext<AuthContextType>({
    token: null,
    login: (newToken: string) => {},
    logout: () => {},
    notifications: [],
    unreadCount: 0,
    openNotification: (n: NotificationItem) => {},
    markAllRead: () => {},
    analyses: {},
    selectedAnalysis: null,
    setSelectedAnalysis: () => {},
});

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
    const navigate = useNavigate();
    const [token, setToken] = useState<string | null>(() => {
        try {
            return localStorage.getItem('token');
        } catch {
            return null;
        }
    });

    useEffect(() => {
        if (token) {
            try { localStorage.setItem('token', token); } catch {}
        } else {
            try { localStorage.removeItem('token'); } catch {}
        }
    }, [token]);

    const login = (newToken: string) => {
        setToken(newToken);
        navigate("/analysis");
    };

    const logout = () => {
        setToken(null);
        navigate("/signin");
    };

    // notifications
    const [notifications, setNotifications] = useState<NotificationItem[]>([]);
    const [analyses, setAnalyses] = useState<Record<string, AnalysisData>>({});
    const [selectedAnalysis, setSelectedAnalysis] = useState<AnalysisData | null>(null);

    const unreadCount = useMemo(() => notifications.filter(n => !n.read).length, [notifications]);

    // decode user id from JWT if present
    const getUserIdFromToken = (tok: string | null) => {
        if (!tok) return null;
        try {
            const parts = tok.split('.');
            if (parts.length < 2) return null;
            const payload = JSON.parse(atob(parts[1]));
            return payload.sub || payload.user_id || payload.id || payload.uid || null;
        } catch (e) { return null; }
    };

    useEffect(() => {
        let ws: WebSocket | null = null;
        const userId = getUserIdFromToken(token);
        if (!userId) return;

        const wsUrl = getWebSocketUrl(`/ws/notifications/${userId}`);
        try {
            ws = new WebSocket(wsUrl);
        } catch (e) {
            console.error('WebSocket connection failed', e);
            return;
        }

        ws.addEventListener('open', () => {
            console.log('notifications websocket connected');
        });

        ws.addEventListener('message', (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                // server might send either the notification directly or wrapped
                const payload = msg.notification || msg || {};
                if (!payload || !payload.analysis_id) return;

                const id = String(Date.now()) + Math.random().toString(16).slice(2,8);
                const item: NotificationItem = {
                    id,
                    user_id: payload.user_id || userId,
                    analysis_id: String(payload.analysis_id),
                    read: false,
                    timestamp: Date.now(),
                    raw: payload,
                };

                setNotifications(prev => [item, ...prev]);

                // if full analysis details included, cache them
                if (payload.presigned_url || payload.description || payload.summary || payload.image) {
                    setAnalyses(prev => ({ ...prev, [String(payload.analysis_id)]: {
                        id: String(payload.analysis_id),
                        user_id: payload.user_id || userId,
                        presigned_url: payload.presigned_url,
                        description: payload.description,
                        summary: payload.summary,
                        image: payload.image,
                    }}));
                }
            } catch (e) {
                console.error('invalid notification message', e);
            }
        });

        ws.addEventListener('close', () => {
            console.log('notifications websocket closed');
        });

        return () => { if (ws) ws.close(); };
    }, [token]);

    const markAllRead = () => setNotifications(prev => prev.map(n => ({ ...n, read: true })));

    const openNotification = async (n: NotificationItem) => {
        // mark read
        setNotifications(prev => prev.map(it => it.id === n.id ? { ...it, read: true } : it));

        const aid = n.analysis_id;
        
        // If analysis is already cached, use it
        if (analyses[aid]) {
            setSelectedAnalysis(analyses[aid]);
            navigate('/dashboard');
            return;
        }

        // Fetch analysis details from API if not cached
        try {
            const resp = await fetch(apiEndpoint(`me/analysis/${aid}`), {
                headers: token ? { Authorization: `Bearer ${token}`, Accept: 'application/json' } : { Accept: 'application/json' },
            });
            
            if (resp.ok) {
                const data = await resp.json();
                const ad: AnalysisData = {
                    id: data.id || aid,
                    user_id: data.user_id || n.user_id,
                    presigned_url: data.presigned_url || data.url,
                    description: data.description || '',
                    summary: data.summary || '',
                    image: data.image, // Base64 encoded image
                };
                setAnalyses(prev => ({ ...prev, [aid]: ad }));
                setSelectedAnalysis(ad);
            } else {
                console.warn('failed to fetch analysis', resp.status);
                // Still set minimal data so dashboard can attempt to fetch
                setSelectedAnalysis({
                    id: aid,
                    user_id: n.user_id,
                    description: '',
                    summary: '',
                });
            }
        } catch (e) {
            console.error('error fetching analysis', e);
            // Still set minimal data so dashboard can attempt to fetch
            setSelectedAnalysis({
                id: aid,
                user_id: n.user_id,
                description: '',
                summary: '',
            });
        }

        navigate('/dashboard');
    };

    const value = useMemo(() => ({ token, login, logout, notifications, unreadCount, openNotification, markAllRead, analyses, selectedAnalysis, setSelectedAnalysis }), [token, notifications, analyses, selectedAnalysis]);

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    return useContext(AuthContext);
};