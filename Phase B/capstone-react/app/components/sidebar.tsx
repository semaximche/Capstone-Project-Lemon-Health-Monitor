import { useNavigate } from "react-router";
import { NavigationList } from "./navigation-list";

export function Sidebar({open, onToggleSidebar}: {open: boolean, onToggleSidebar: () => void}) {
  const navigate = useNavigate();

  return (
    <aside className={`fixed w-72 h-full glass-panel border-r border-cyan-500/20 transition-all duration-300 z-50 shadow-2xl ${open ? "translate-x-0" : "-translate-x-full"}`}>
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-950/20 to-slate-950/20" />
      <nav className="flex items-start flex-col p-6 relative z-10">
        <div className="flex items-center justify-between w-full mb-8">
          <h2 className="text-2xl font-display font-bold text-gradient-cyan">Menu</h2>
          <button 
            className="px-3 py-1.5 rounded-lg hover:bg-cyan-500/10 text-cyan-300 hover:text-cyan-100 text-xl font-bold transition-all hover-lift" 
            onClick={onToggleSidebar}
            aria-label="Close sidebar"
          >
            <span>&#10005;</span>
          </button>
        </div>
        <NavigationList/>
      </nav>
    </aside>
  );
}