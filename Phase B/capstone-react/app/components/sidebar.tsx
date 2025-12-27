import { useNavigate } from "react-router";
import { NavigationList } from "./navigation-list";

export function Sidebar({open, onToggleSidebar}: {open: boolean, onToggleSidebar: () => void}) {
  const navigate = useNavigate();

  return (
    <aside className={`fixed w-64 h-full border-r bg-gray-900 transition-all ${open ? "translate-x-0" : "-translate-x-full"}`}>
      <nav className="flex items-start flex-col p-4">
        <button className="mb-6 px-2 rounded-md hover:bg-gray-600 text-xl font-bold" onClick={onToggleSidebar}>
            <span>&#10005;</span>
        </button>
        <NavigationList/>
      </nav>
    </aside>
  );
}