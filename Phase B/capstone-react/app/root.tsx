import { isRouteErrorResponse, Links, Meta, Outlet, Scripts, ScrollRestoration, } from "react-router";

import type { Route } from "./+types/root";
import "./app.css";
import { AuthProvider } from "./provider/auth-context";
import { useState } from "react";
import { Header } from "~/components/header";
import { Sidebar } from "~/components/sidebar";

// link and preload google fonts
export const links: Route.LinksFunction = () => [
  { rel: "preconnect", href: "https://fonts.googleapis.com" },
  { rel: "preconnect", href: "https://fonts.gstatic.com", crossOrigin: "anonymous", },
  { rel: "stylesheet", href: "https://fonts.googleapis.com/css2?family=Heebo:wght@100..900&family=Noto+Color+Emoji&family=Roboto+Mono:ital,wght@0,100..700;1,100..700&display=swap", },
];

// root layout
export function Layout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Links />
      </head>
      <body className="bg-white dark:bg-gray-950 text-black dark:text-white">
        <AuthProvider>
          <Sidebar open={sidebarOpen} onToggleSidebar={() => setSidebarOpen(false)} />
          <Header onToggleSidebar={() => setSidebarOpen(true)} />
          <main className="pt-16">
            {children}
          </main>
        </AuthProvider>
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

// root app
export default function App() {
  return <Outlet />;
}

// error page
export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  let message = "Oops!";
  let details = "An unexpected error occurred.";
  let stack: string | undefined;

  if (isRouteErrorResponse(error)) {
    message = error.status === 404 ? "404" : "Error";
    details =
      error.status === 404
        ? "The requested page could not be found."
        : error.statusText || details;
  } else if (import.meta.env.DEV && error && error instanceof Error) {
    details = error.message;
    stack = error.stack;
  }

  return (
    <main className="pt-16 p-4 container mx-auto">
      <h1>{message}</h1>
      <p>{details}</p>
      {stack && (
        <pre className="w-full p-4 overflow-x-auto">
          <code>{stack}</code>
        </pre>
      )}
    </main>
  );
}
