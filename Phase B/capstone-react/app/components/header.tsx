import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import { useState } from "react";

export function Header({ onToggleSidebar }: {onToggleSidebar:() => void}) {
  const { token, logout } = useAuth();
  const navigate = useNavigate();
  const { notifications, unreadCount, openNotification } = useAuth();
  const [showNotif, setShowNotif] = useState(false);

  return (
    <header className="fixed top-0 left-0 right-0 flex h-16 items-center justify-between px-6 shadow-2xl glass-panel border-b border-cyan-500/20 z-40">
      <div className="flex items-center gap-4">
        <button
          aria-label="Toggle sidebar"
          className="rounded-lg p-2 hover:bg-cyan-500/10 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 transition-all"
          onClick={onToggleSidebar}
        >
          <div className="space-y-1.5">
            <span className="block h-0.5 w-6 bg-cyan-300 rounded-full transition-all" />
            <span className="block h-0.5 w-6 bg-cyan-300 rounded-full transition-all" />
            <span className="block h-0.5 w-6 bg-cyan-300 rounded-full transition-all" />
          </div>
        </button>

        <button 
          onClick={() => navigate('/')}
          className="text-sm sm:text-base md:text-xl font-display font-bold text-gradient-cyan hover:scale-105 transition-transform cursor-pointer truncate max-w-[120px] sm:max-w-none"
        >
          <span className="hidden sm:inline">Lemon Disease Detection</span>
          <span className="sm:hidden">Lemon Health</span>
        </button>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
            {token ? (
          <>
            <button 
              onClick={() => navigate('/analysis')} 
              className="hidden sm:inline-block text-xs sm:text-sm px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 font-medium transition-all hover-lift"
            >
              Analysis
            </button>
            <button 
              onClick={() => navigate('/analyses')} 
              className="hidden md:inline-block text-xs sm:text-sm px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 font-medium transition-all hover-lift"
            >
              My Analyses
            </button>

            <div className="relative">
              <button
                onClick={() => setShowNotif(v => !v)}
                className="relative p-2.5 rounded-lg hover:bg-cyan-500/10 transition-all hover-lift"
                aria-label="Notifications"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-cyan-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6 6 0 10-12 0v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 inline-flex items-center justify-center rounded-full bg-gradient-to-r from-red-500 to-pink-500 text-white text-xs font-bold w-5 h-5 neon-glow">{unreadCount}</span>
                )}
              </button>

              {showNotif && (
                <div className="absolute right-0 mt-2 w-[calc(100vw-2rem)] sm:w-80 max-h-96 overflow-auto rounded-xl glass-panel border-cyan-500/30 shadow-2xl z-50">
                  <div className="p-4 border-b border-cyan-500/20 font-display font-semibold text-cyan-100">Notifications</div>
                  {notifications.length === 0 && <div className="p-4 text-sm text-cyan-300/60">No notifications</div>}
                  {notifications.map(n => (
                    <div 
                      key={n.id} 
                      className={`p-4 cursor-pointer hover:bg-cyan-500/10 transition-colors border-b border-cyan-500/10 last:border-0 ${n.read ? 'opacity-60' : ''}`} 
                      onClick={() => { openNotification(n); setShowNotif(false); }}
                    >
                      <div className="text-sm font-medium text-cyan-100">Analysis {n.analysis_id}</div>
                      <div className="text-xs text-cyan-300/60 font-mono mt-1">{new Date(n.timestamp).toLocaleString()}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <button 
              onClick={() => logout()} 
              className="text-xs sm:text-sm px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-500 hover:to-pink-500 text-white font-medium shadow-lg transition-all hover-lift"
            >
              Logout
            </button>
          </>
        ) : (
          <>
            <button 
              onClick={() => navigate('/signin')} 
              className="text-xs sm:text-sm px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 font-medium transition-all hover-lift"
            >
              Sign in
            </button>
            <button 
              onClick={() => navigate('/register')} 
              className="text-xs sm:text-sm px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white font-medium shadow-lg neon-glow transition-all hover-lift"
            >
              Register
            </button>
          </>
        )}
      </div>
    </header>
  );
}
