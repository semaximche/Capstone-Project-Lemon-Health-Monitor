import { useNavigate } from "react-router";

export function NavigationList() {
  const navigate = useNavigate();

  return (
    <ul className="space-y-2">
      <li className="rounded px-3 py-1 hover:bg-gray-500">
        <button onClick={() => {navigate("/signin")}}>
          Login
        </button>
      </li>
      <li className="rounded px-3 py-1 hover:bg-gray-500">
        <button onClick={() => {navigate("/register")}}>
          Register
        </button>
      </li>
      <li className="rounded px-3 py-1 hover:bg-gray-500">
        <button onClick={() => {navigate("/analysis")}}>
          Analysis
      </button>
    </li>
    <li className="rounded px-3 py-1 hover:bg-gray-500">
      <button onClick={() => {window.open("https://github.com/semaximche/Capstone-Project-Lemon-Health-Monitor")}}>
          Github
        </button>
      </li>
    </ul>
  );
}