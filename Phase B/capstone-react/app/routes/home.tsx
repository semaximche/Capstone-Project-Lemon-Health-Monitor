import type { Route } from "./+types/home";
import Welcome from "~/components/welcome";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export default function Home() {
  return (
    <main className="flex text-center justify-center m-5">
      <div className="w-full max-w-4xl">
        <Welcome />
      </div>
    </main>
  );
}
