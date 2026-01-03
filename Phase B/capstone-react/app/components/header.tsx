import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import { useState } from "react";

export function Header({ onToggleSidebar }: {onToggleSidebar:() => void}) {
  const { token, logout } = useAuth();
  const navigate = useNavigate();
  const { notifications, unreadCount, openNotification } = useAuth();
  const [showNotif, setShowNotif] = useState(false);

  return (
    <header className="flex h-14 items-center justify-between px-4 shadow-sm bg-gradient-to-r from-green-700 to-green-900 text-white">
      <div className="flex items-center gap-3">
        <button
          aria-label="Toggle sidebar"
          className="rounded-md p-2 hover:bg-white/10 focus:outline-none focus:ring"
          onClick={onToggleSidebar}
        >
          <div className="space-y-1">
            <span className="block h-0.5 w-5 bg-white/80" />
            <span className="block h-0.5 w-5 bg-white/80" />
            <span className="block h-0.5 w-5 bg-white/80" />
          </div>
        </button>

        <span className="text-lg font-semibold">Lemon Disease Detection</span>
      </div>

      <div className="flex items-center gap-3">
            {token ? (
          <>
            <button onClick={() => navigate('/analysis')} className="text-sm px-3 py-1 rounded bg-white/10 hover:bg-white/20">Analysis</button>

            <div className="relative">
              <button
                onClick={() => setShowNotif(v => !v)}
                className="relative p-2 rounded hover:bg-white/10"
                aria-label="Notifications"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6 6 0 10-12 0v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
                {unreadCount > 0 && (
                  <span className="absolute -top-0.5 -right-0.5 inline-flex items-center justify-center rounded-full bg-red-600 text-white text-xs w-5 h-5">{unreadCount}</span>
                )}
              </button>

              {showNotif && (
                <div className="absolute right-0 mt-2 w-80 max-h-80 overflow-auto rounded bg-white text-black shadow-lg z-50">
                  <div className="p-2 border-b">Notifications</div>
                  {notifications.length === 0 && <div className="p-2 text-sm">No notifications</div>}
                  {notifications.map(n => (
                    <div key={n.id} className={`p-2 cursor-pointer hover:bg-gray-100 ${n.read ? 'opacity-70' : ''}`} onClick={() => { openNotification(n); setShowNotif(false); }}>
                      <div className="text-sm font-medium">Analysis {n.analysis_id}</div>
                      <div className="text-xs text-gray-600">{new Date(n.timestamp).toLocaleString()}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <button onClick={() => logout()} className="text-sm px-3 py-1 rounded bg-red-600 hover:bg-red-500">Logout</button>
          </>
        ) : (
          <>
            <button onClick={() => navigate('/signin')} className="text-sm px-3 py-1 rounded bg-white/10 hover:bg-white/20">Sign in</button>
            <button onClick={() => navigate('/register')} className="text-sm px-3 py-1 rounded bg-white/10 hover:bg-white/20">Register</button>
          </>
        )}
      </div>
    </header>
  );
}
