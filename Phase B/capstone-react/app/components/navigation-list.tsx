import { useNavigate, useLocation } from "react-router";
import { useAuth } from "~/provider/auth-context";

export function NavigationList() {
  const navigate = useNavigate();
  const location = useLocation();
  const { token } = useAuth();

  // Navigation items for signed out users
  const signedOutNavItems = [
    { path: "/signin", label: "Sign In", icon: "🔐" },
    { path: "/register", label: "Register", icon: "📝" },
    { path: "/about", label: "About", icon: "ℹ️" },
  ];

  // Navigation items for signed in users
  const signedInNavItems = [
    { path: "/", label: "Home", icon: "🏠" },
    { path: "/analysis", label: "Analysis", icon: "🔬" },
    { path: "/about", label: "About", icon: "ℹ️" },
  ];

  // Use appropriate nav items based on auth status
  const navItems = token ? signedInNavItems : signedOutNavItems;

  const isActive = (path: string) => {
    if (path === "/") {
      return location.pathname === "/";
    }
    return location.pathname.startsWith(path);
  };

  return (
    <ul className="space-y-2">
      {navItems.map((item) => (
        <li key={item.path}>
          <button
            onClick={() => navigate(item.path)}
            className={`w-full text-left rounded-xl px-5 py-4 transition-all duration-300 flex items-center gap-4 hover-lift ${
              isActive(item.path)
                ? "bg-gradient-to-r from-cyan-500/20 to-teal-500/20 text-cyan-100 border border-cyan-500/30 neon-glow-cyan shadow-lg"
                : "text-cyan-200/70 hover:bg-cyan-500/10 hover:text-cyan-100 border border-transparent hover:border-cyan-500/20"
            }`}
          >
            <span className="text-2xl">{item.icon}</span>
            <span className="font-display font-medium text-base">{item.label}</span>
          </button>
        </li>
      ))}
      <li>
        <button
          onClick={() => window.open("https://github.com/semaximche/Capstone-Project-Lemon-Health-Monitor", "_blank")}
          className="w-full text-left rounded-xl px-5 py-4 transition-all duration-300 flex items-center gap-4 text-cyan-200/70 hover:bg-cyan-500/10 hover:text-cyan-100 border border-transparent hover:border-cyan-500/20 hover-lift"
        >
          <span className="text-2xl">💻</span>
          <span className="font-display font-medium text-base">GitHub</span>
        </button>
      </li>
    </ul>
  );
}