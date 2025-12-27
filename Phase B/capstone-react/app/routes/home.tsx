import { Header } from "~/components/header";
import type { Route } from "./+types/home";
import { Sidebar } from "~/components/sidebar";
import { useState } from "react";
import { useNavigate } from "react-router";
import { NavigationList } from "~/components/navigation-list";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <div>
      <Sidebar open={sidebarOpen} onToggleSidebar={() => {setSidebarOpen(false)}} />
      <Header onToggleSidebar={() => {setSidebarOpen(true)}} />
      <main className="flex text-center justify-center m-5">
        <div className="border p-4 rounded-md">
          <h1 className="font-bold text-xl p-2">
            Lemon Disease Detection
          </h1>
          <NavigationList/>
        </div>
      </main>
    </div>
  );
}
