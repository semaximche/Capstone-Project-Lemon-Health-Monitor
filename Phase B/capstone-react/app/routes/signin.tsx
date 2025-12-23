import { useRef } from "react";
import { handleLogin } from "~/lib/user-auth";
import { useAuth } from "~/provider/auth-context";

export default function Signin() {
    const { login } = useAuth();

    const usernameRef = useRef<HTMLInputElement>(null);
    const passwordRef = useRef<HTMLInputElement>(null);

    const handleClick = () => {
        if(usernameRef.current && passwordRef.current) {
            console.log('logging in with ', usernameRef.current.value, passwordRef.current.value);
            const data = handleLogin(usernameRef.current.value, passwordRef.current.value);
            login(data)
        }
    }

    return (
        <div className="max-w-sm mx-auto mt-20 p-6 bg-black rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Login</h2>

        <input ref={usernameRef} className="w-full border p-2 rounded mb-3" placeholder="Email" />

        <input ref={passwordRef} type="password" className="w-full border p-2 rounded mb-4" placeholder="Password" />

        <button onClick={handleClick} className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-500 transition">
            Login
        </button>
        </div>
    )
}