import { useRef, useState } from "react";
import { handleLogin } from "~/lib/user-auth";
import { useAuth } from "~/provider/auth-context";

export default function Signin() {
    const [success, setSuccess] = useState<boolean>(false);
    const { login } = useAuth();

    const usernameRef = useRef<HTMLInputElement>(null);
    const passwordRef = useRef<HTMLInputElement>(null);

    const handleClick = () => {
        if(usernameRef.current && passwordRef.current) {
            console.log('logging in with ', usernameRef.current.value, passwordRef.current.value);
            handleLogin(usernameRef.current.value, passwordRef.current.value).then(function (response) {
                login(response);
                setSuccess(true);
            })
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center p-6">
        <div className="w-full max-w-md glass-panel rounded-2xl shadow-2xl border border-cyan-500/20 p-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
            <div className="relative z-10">
        <h2 className="text-2xl font-display font-bold text-gradient-cyan mb-4">Login</h2>

        <input ref={usernameRef} className="w-full border border-cyan-500/20 glass-panel placeholder-cyan-400/60 text-cyan-100 p-3 rounded-xl mb-3 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent" placeholder="Email" />

        <input ref={passwordRef} type="password" className="w-full border border-cyan-500/20 glass-panel placeholder-cyan-400/60 text-cyan-100 p-3 rounded-xl mb-4 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent" placeholder="Password" />

        <button onClick={handleClick} className="w-full bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white py-3 rounded-xl font-semibold shadow-lg neon-glow hover-lift transition-all duration-300">
            Login
        </button>
            </div>
        </div>
        </div>
    )
}