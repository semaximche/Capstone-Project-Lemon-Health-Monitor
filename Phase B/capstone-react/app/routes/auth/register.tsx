import { useRef, useState } from "react";
import { handleRegister } from "~/lib/user-auth";
import { useNavigate } from "react-router";
import Alert from "~/components/alert";

export default function Register() {
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<boolean>(false);
    const [loading, setLoading] = useState(false);
    const usernameRef = useRef<HTMLInputElement>(null);
    const emailRef = useRef<HTMLInputElement>(null);
    const passwordRef = useRef<HTMLInputElement>(null);
    const navigate = useNavigate();

    const handleClick = async () => {
        setError(null);
        if (usernameRef.current && emailRef.current && passwordRef.current) {
            try {
                setLoading(true);
                const resp = await handleRegister(
                    usernameRef.current.value,
                    emailRef.current.value,
                    passwordRef.current.value
                );
                setSuccess(true);
                // If backend returned token, you could auto-login. For now, redirect to signin.
                navigate('/signin');
            } catch (err: any) {
                setError(err?.message || 'Registration failed');
            } finally {
                setLoading(false);
            }
        }
    };

        return (
                <div className="relative min-h-screen flex items-center justify-center p-6">
                    <button onClick={() => navigate('/')} aria-label="Home" className="absolute top-2 left-2 sm:top-4 sm:left-4 z-50 px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-base rounded-xl glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 shadow-lg hover-lift transition-all">Home</button>
                <div className="w-full max-w-md glass-panel rounded-2xl shadow-2xl border border-cyan-500/20 p-8 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
                    <div className="relative z-10">
            <h2 className="text-2xl font-display font-bold text-gradient-cyan mb-4">Create an account</h2>
            <p className="text-sm text-cyan-200/80 mb-6">Enter details to register and start analyzing images.</p>

            <input ref={usernameRef} className="w-full border border-cyan-500/20 glass-panel placeholder-cyan-400/60 text-cyan-100 p-3 rounded-xl mb-3 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent" placeholder="Username" />
            <input ref={emailRef} className="w-full border border-cyan-500/20 glass-panel placeholder-cyan-400/60 text-cyan-100 p-3 rounded-xl mb-3 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent" placeholder="Email" />
            <input ref={passwordRef} type="password" className="w-full border border-cyan-500/20 glass-panel placeholder-cyan-400/60 text-cyan-100 p-3 rounded-xl mb-4 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent" placeholder="Password" />

            {error && <div className="mb-3"><Alert type="error">{error}</Alert></div>}
            {success && <div className="mb-3"><Alert type="success">Registration successful. Redirecting to sign in...</Alert></div>}

            <button onClick={handleClick} disabled={loading} className="w-full bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white py-3 rounded-xl font-semibold disabled:opacity-60 mb-3 shadow-lg neon-glow hover-lift transition-all duration-300">
                {loading ? 'Creating account…' : 'Create account'}
            </button>

                        <div className="flex justify-center gap-3">
                            <button onClick={() => navigate('/signin')} className="text-sm text-cyan-300 hover:text-cyan-200 hover:underline transition-colors">Already have an account? Sign in</button>
                        </div>
                    </div>
        </div>
        </div>
    );
}
