export function Header({ onToggleSidebar }: {onToggleSidebar:() => void}) {
  return (
    <header className="flex h-14 items-center justify-between border-gray-300 border-b px-4 shadow-sm">
      <button
          aria-label="Toggle sidebar"
          className="rounded-md p-2 hover:bg-gray-600 focus:outline-none focus:ring"
          onClick={onToggleSidebar}
        >
          <div className="space-y-1">
            <span className="block h-0.5 w-5 bg-gray-300" />
            <span className="block h-0.5 w-5 bg-gray-300" />
            <span className="block h-0.5 w-5 bg-gray-300" />
          </div>
        </button>

        <span className="text-lg font-semibold">Lemon Disease Detection</span>
    </header>
  );
} 