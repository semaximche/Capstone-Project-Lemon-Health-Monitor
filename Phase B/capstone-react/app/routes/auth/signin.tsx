import { useRef, useState } from "react";
import { handleLogin } from "~/lib/user-auth";
import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import Alert from "~/components/alert";

export default function Signin() {
    const [success, setSuccess] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const usernameRef = useRef<HTMLInputElement>(null);
    const passwordRef = useRef<HTMLInputElement>(null);

    const handleClick = async () => {
        setError(null);
        if(usernameRef.current && passwordRef.current) {
            try {
                setLoading(true);
                const response = await handleLogin(usernameRef.current.value, passwordRef.current.value);
                if(response) {
                    login(response);
                    setSuccess(true);
                }
            } catch (e: any) {
                setError(e?.message || 'Login failed');
            } finally {
                setLoading(false);
            }
        }
    }

        return (
                <div className="relative min-h-screen flex items-center justify-center p-6">
                    <button onClick={() => navigate('/')} aria-label="Home" className="absolute top-2 left-2 sm:top-4 sm:left-4 z-50 px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-base rounded-xl glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 shadow-lg hover-lift transition-all">Home</button>
                <div className="w-full max-w-md glass-panel rounded-2xl shadow-2xl border border-cyan-500/20 p-8 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
                    <div className="relative z-10">
          <h2 className="text-2xl font-display font-bold text-gradient-cyan mb-4">Welcome back</h2>
          <p className="text-sm text-cyan-200/80 mb-6">Sign in to access your dashboard and analysis tools.</p>

          <input ref={usernameRef} className="w-full border border-cyan-500/20 glass-panel placeholder-cyan-400/60 text-cyan-100 p-3 rounded-xl mb-3 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent" placeholder="Email or username" />

          <input ref={passwordRef} type="password" className="w-full border border-cyan-500/20 glass-panel placeholder-cyan-400/60 text-cyan-100 p-3 rounded-xl mb-4 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent" placeholder="Password" />

          {error && <div className="mb-3"><Alert type="error">{error}</Alert></div>}
          {success && <div className="mb-3"><Alert type="success">Signed in successfully.</Alert></div>}
          <button onClick={handleClick} disabled={loading} className="w-full bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white py-3 rounded-xl font-semibold disabled:opacity-60 mb-3 shadow-lg neon-glow hover-lift transition-all duration-300">
              {loading ? 'Signing in…' : 'Sign in'}
          </button>

                    <div className="flex justify-center gap-3">
                        <button onClick={() => navigate('/register')} className="text-sm text-cyan-300 hover:text-cyan-200 hover:underline transition-colors">Create an account</button>
                    </div>
                    </div>
        </div>
        </div>
    )
}
