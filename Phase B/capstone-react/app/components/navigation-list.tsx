import { useNavigate, useLocation } from "react-router";

export function NavigationList() {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Home", icon: "🏠" },
    { path: "/signin", label: "Sign In", icon: "🔐" },
    { path: "/register", label: "Register", icon: "📝" },
    { path: "/analysis", label: "Analysis", icon: "🔬" },
    { path: "/about", label: "About", icon: "ℹ️" },
  ];

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
            className={`w-full text-left rounded-lg px-4 py-3 transition-all duration-200 flex items-center gap-3 ${
              isActive(item.path)
                ? "bg-emerald-600 text-white shadow-lg"
                : "text-emerald-100 hover:bg-emerald-800/50 hover:text-white"
            }`}
          >
            <span className="text-xl">{item.icon}</span>
            <span className="font-medium">{item.label}</span>
          </button>
        </li>
      ))}
      <li>
        <button
          onClick={() => window.open("https://github.com/semaximche/Capstone-Project-Lemon-Health-Monitor", "_blank")}
          className="w-full text-left rounded-lg px-4 py-3 transition-all duration-200 flex items-center gap-3 text-emerald-100 hover:bg-emerald-800/50 hover:text-white"
        >
          <span className="text-xl">💻</span>
          <span className="font-medium">GitHub</span>
        </button>
      </li>
    </ul>
  );
}