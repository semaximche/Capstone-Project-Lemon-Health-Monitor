import { useNavigate } from "react-router";
import { NavigationList } from "./navigation-list";

export function Sidebar({open, onToggleSidebar}: {open: boolean, onToggleSidebar: () => void}) {
  const navigate = useNavigate();

  return (
    <aside className={`fixed w-64 h-full border-r border-emerald-700 bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/90 transition-all z-50 ${open ? "translate-x-0" : "-translate-x-full"}`}>
      <nav className="flex items-start flex-col p-4">
        <div className="flex items-center justify-between w-full mb-6">
          <h2 className="text-lg font-semibold text-emerald-100">Menu</h2>
          <button 
            className="px-3 py-1 rounded-md hover:bg-emerald-700/50 text-emerald-100 hover:text-white text-xl font-bold transition-colors" 
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