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
                <div className="relative min-h-screen flex items-center justify-center bg-gradient-to-b from-green-900 via-black to-green-900 p-6">
                    <button onClick={() => navigate('/')} aria-label="Home" className="absolute top-4 left-4 z-50 px-4 py-2 text-base rounded-lg bg-white/10 text-emerald-100 hover:bg-white/20 shadow-lg">Home</button>
                <div className="w-full max-w-md bg-white/5 backdrop-blur rounded-xl shadow-lg border border-white/10 p-8">
          <h2 className="text-2xl font-bold text-white mb-4">Welcome back</h2>
          <p className="text-sm text-gray-300 mb-6">Sign in to access your dashboard and analysis tools.</p>

          <input ref={usernameRef} className="w-full border border-white/20 bg-white/5 placeholder-gray-300 text-white p-3 rounded mb-3 focus:outline-none focus:ring-2 focus:ring-green-500" placeholder="Email or username" />

          <input ref={passwordRef} type="password" className="w-full border border-white/20 bg-white/5 placeholder-gray-300 text-white p-3 rounded mb-4 focus:outline-none focus:ring-2 focus:ring-green-500" placeholder="Password" />

          {error && <div className="mb-3"><Alert type="error">{error}</Alert></div>}
          {success && <div className="mb-3"><Alert type="success">Signed in successfully.</Alert></div>}
          <button onClick={handleClick} disabled={loading} className="w-full bg-green-600 text-white py-3 rounded font-medium hover:bg-green-500 disabled:opacity-60 mb-3">
              {loading ? 'Signing in…' : 'Sign in'}
          </button>

                    <div className="flex justify-center gap-3">
                        <button onClick={() => navigate('/register')} className="text-sm text-green-300 hover:underline">Create an account</button>
                    </div>
        </div>
        </div>
    )
}
